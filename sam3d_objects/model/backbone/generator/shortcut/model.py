# Copyright (c) Meta Platforms, Inc. and affiliates.
import random
from typing import Callable, Sequence, Union
import torch
import numpy as np
from functools import partial
import optree
import math

from sam3d_objects.model.backbone.generator.base import Base
from sam3d_objects.data.utils import right_broadcasting
from sam3d_objects.data.utils import tree_tensor_map, tree_reduce_unique
from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching, _get_device
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import copy
import os

import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os


# https://arxiv.org/pdf/2410.12557
# ss_generator
class ShortCut(FlowMatching):
    def __init__(
        self,
        no_shortcut=False,
        self_consistency_prob=0.25,
        shortcut_loss_weight=1.0,
        self_consistency_cfg_strength=3.0,
        ratio_cfg_samples_in_self_consistency_target=0.5,
        fm_in_shortcut_target_prob=0.0,
        fm_eps_max=0,
        batch_mode=False,
        cfg_modalities=["shape"],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.no_shortcut = no_shortcut
        self.self_consistency_prob = self_consistency_prob
        self.shortcut_loss_weight = shortcut_loss_weight
        self.self_consistency_cfg_strength = self_consistency_cfg_strength
        self.ratio_cfg_samples_in_self_consistency_target = ratio_cfg_samples_in_self_consistency_target
        self.fm_in_shortcut_target_prob = fm_in_shortcut_target_prob
        self.fm_eps_max = fm_eps_max
        self.batch_mode = batch_mode
        self.cfg_modalities = cfg_modalities
        self.k_map = None

    def _generate_d(self, x1):
        """
        Generate shortcut step sizes d with binary-time schedule.
        
        This method ensures deterministic behavior for distributed training:
        - Exactly self_consistency_prob fraction of samples will have d > 0 (self-consistency)
        - Remaining samples will have d = 0 (flow matching)
        - All distributed ranks will have consistent counts, preventing deadlocks
        
        Args:
            x1: Input tensor or tree of tensors
            
        Returns:
            d: Tensor of step sizes with shape [batch_size]
        """
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device

        # Use binary-time schedule: d ∈ {1/2^i for i in range(8)}
        base = [1 / 2**i for i in range(8)]
        
        # Deterministic approach: exactly self_consistency_prob fraction will have d>0
        # This ensures all distributed ranks have consistent behavior
        if self.batch_mode:
            num_self_consistency_samples = int(random.random() < self.self_consistency_prob) * batch_size
        else:
            num_self_consistency_samples = int(batch_size * self.self_consistency_prob)
        num_flow_matching_samples = batch_size - num_self_consistency_samples
        
        # Create deterministic d values
        d = torch.zeros(batch_size, device=device)
        
        if num_self_consistency_samples > 0:
            # Randomly select d values for self-consistency samples
            selected_elements = random.choices(base, k=num_self_consistency_samples)
            d[:num_self_consistency_samples] = torch.FloatTensor(selected_elements).to(device)
        
        # Shuffle the d values to randomize which samples get which d values
        # This maintains the deterministic count while randomizing positions
        shuffle_indices = torch.randperm(batch_size, device=device)
        d = d[shuffle_indices]

        return d

    @torch.no_grad()
    def compute_self_consistency_target(self, x_t, t, d, *args_conditionals, **kwargs_conditionals):
        """
        Compute self-consistency target for shortcut model's self-consistency objective.
        
        This method uses a mixed approach where:
        - First 25% of samples (num_cfg_samples) use CFG blending with strength 7.0
        - Remaining 75% of samples use conditional-only targets
        
        Safety guarantees:
        - Ensures at least 1 sample in CFG part (num_cfg_samples >= 1 for batch_size >= 2)
        - For batch_size < 2, falls back to all conditional-only (no CFG)
        - Handles edge cases where batch size is too small for mixed approach
        
        The process involves:
        1. Forward all samples through conditional model to get s_t_cond and s_td_cond
        2. Forward first num_cfg_samples through unconditional model to get s_t_uncond and s_td_uncond  
        3. Apply CFG blending: (1 + strength) * cond - strength * uncond for first num_cfg_samples
        4. Concatenate CFG results with conditional-only results for remaining samples
        5. Average the two velocities to get final self-consistency target
        """
        # CFG strength for self-consistency target computation
        self_consistency_cfg_strength = self.self_consistency_cfg_strength
        
        # Mixed approach: configurable ratio of CFG:conditional-only samples
        batch_size = x_t.shape[0] if not isinstance(x_t, dict) else next(iter(x_t.values())).shape[0]
        if self.batch_mode:
            num_cfg_samples = int(random.random() < self.ratio_cfg_samples_in_self_consistency_target) * batch_size
        else:
            num_cfg_samples = int(batch_size * self.ratio_cfg_samples_in_self_consistency_target)  # Configurable ratio for CFG
        num_cond_only_samples = batch_size - num_cfg_samples  # Remaining for conditional-only
        use_fm_in_shortcut_target = random.random() < self.fm_in_shortcut_target_prob

        # Handle edge case where batch_size < 2 (fallback to all conditional-only)
        # if batch_size < 2:
        #     num_cfg_samples = 0
        #     num_cond_only_samples = batch_size
        
        
        # ### DEBUG ###############################
        # num_cfg_samples = 0
        # num_cond_only_samples = batch_size
        # # ### DEBUG ###############################
        
        # Step 1: Get velocity predictions at current time t
        # Forward all samples through conditional model
        s_t_cond = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            d=d * self.time_scale if not use_fm_in_shortcut_target else d * self.time_scale * 0,
            p_unconditional=0.0,
            **kwargs_conditionals,
        )
        
        # Handle CFG and conditional-only parts
        if num_cfg_samples > 0:
            # Forward first num_cfg_samples through unconditional model
            if isinstance(x_t, dict):
                x_t_cfg = {k: v[:num_cfg_samples] for k, v in x_t.items()}
            else:
                x_t_cfg = x_t[:num_cfg_samples]
            
            s_t_uncond = self.reverse_fn(
                x_t_cfg,
                t[:num_cfg_samples] * self.time_scale,
                *(arg[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(arg) else arg for arg in args_conditionals),
                d=d[:num_cfg_samples] * self.time_scale if not use_fm_in_shortcut_target else d[:num_cfg_samples] * self.time_scale * 0,
                p_unconditional=1.0,
                **{k: v[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(v) else v for k, v in kwargs_conditionals.items()},
            )
            
            # Apply CFG blending for first num_cfg_samples using our standard formula
            s_t_cfg = tree_tensor_map(
                lambda cond, uncond: (1 + self_consistency_cfg_strength) * cond - self_consistency_cfg_strength * uncond,
                tree_tensor_map(lambda x: x[:num_cfg_samples], s_t_cond), s_t_uncond
            )
            
            # Combine CFG results with conditional-only results for remaining samples
            if num_cond_only_samples > 0:
                s_t = tree_tensor_map(
                    lambda cfg, cond: torch.cat([cfg, cond[num_cfg_samples:]], dim=0),
                    s_t_cfg, s_t_cond
                )
            else:
                # All samples use CFG
                s_t = s_t_cond
                if isinstance(s_t_cond, dict):
                    for modality in self.cfg_modalities:
                        s_t[modality] = s_t_cfg[modality]
                else:
                    s_t = s_t_cfg
        else:
            # All samples use conditional-only (fallback for very small batches)
            s_t = s_t_cond
        
        # Step 2: Take a step of size d using current velocity
        x_td = tree_tensor_map(lambda x, v: x + v * d[..., None, None], x_t, s_t)
        
        # Step 3: Get velocity predictions at time t+d
        # Forward all samples through conditional model at t+d
        s_td_cond = self.reverse_fn(
            x_td,
            (t + d) * self.time_scale,
            *args_conditionals,
            d=d * self.time_scale if not use_fm_in_shortcut_target else d * self.time_scale * 0,
            p_unconditional=0.0,
            **kwargs_conditionals,
        )
        
        # Handle CFG and conditional-only parts at t+d
        if num_cfg_samples > 0:
            # Forward first num_cfg_samples through unconditional model at t+d
            if isinstance(x_td, dict):
                x_td_cfg = {k: v[:num_cfg_samples] for k, v in x_td.items()}
            else:
                x_td_cfg = x_td[:num_cfg_samples]
            
            s_td_uncond = self.reverse_fn(
                x_td_cfg,
                (t + d)[:num_cfg_samples] * self.time_scale,
                *(arg[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(arg) else arg for arg in args_conditionals),
                d=d[:num_cfg_samples] * self.time_scale if not use_fm_in_shortcut_target else d[:num_cfg_samples] * self.time_scale * 0,
                p_unconditional=1.0,
                **{k: v[:num_cfg_samples] if not self.batch_mode and torch.is_tensor(v) else v for k, v in kwargs_conditionals.items()},
            )
            
            # Apply CFG blending for first num_cfg_samples at t+d using our standard formula
            s_td_cfg = tree_tensor_map(
                lambda cond, uncond: (1 + self_consistency_cfg_strength) * cond - self_consistency_cfg_strength * uncond,
                tree_tensor_map(lambda x: x[:num_cfg_samples], s_td_cond), s_td_uncond
            )
            
            # Combine CFG results with conditional-only results for remaining samples at t+d
            if num_cond_only_samples > 0:
                s_td = tree_tensor_map(
                    lambda cfg, cond: torch.cat([cfg, cond[num_cfg_samples:]], dim=0),
                    s_td_cfg, s_td_cond
                )
            else:
                # All samples use CFG
                s_td = s_td_cond
                if isinstance(s_td_cond, dict):
                    for modality in self.cfg_modalities:
                        s_td[modality] = s_td_cfg[modality]
                else:
                    s_td = s_td_cfg
        else:
            # All samples use conditional-only (fallback for very small batches)
            s_td = s_td_cond
        
        # Step 4: Compute self-consistency target as average of two velocities
        s_target = tree_tensor_map(lambda a, b: (a + b).detach() / 2, s_t, s_td)
        
        return s_target

    def _generate_t_and_d(self, x1):
        """
        Generate t and d together according to shortcut models paper.
        
        According to the paper: "During training, we first sample d, then sample t only at the discrete
        points for which the shortcut model will be queried, i.e. multiples of d. We train the 
        self-consistency objective only at these timesteps."
        
        This ensures that when d > 0 (self-consistency samples), t is sampled at multiples of d.
        When d = 0 (flow matching samples), t can be sampled normally.
        """
        first_tensor = optree.tree_flatten(x1)[0][0]
        batch_size = first_tensor.shape[0]
        device = first_tensor.device
        
        # First sample d
        d = self._generate_d(x1)
        
        # Then sample t based on d
        t = torch.zeros(batch_size, device=device)
        
        # For flow matching samples (d = 0), sample t normally
        flow_matching_mask = (d == 0)
        if flow_matching_mask.any():
            num_flow_samples = flow_matching_mask.sum().item()
            t_flow = self.training_time_sampler_fn(
                size=(num_flow_samples,),
                generator=self.random_generator,
            ).to(device)
            t[flow_matching_mask] = t_flow
        
        # For self-consistency samples (d > 0), sample t at multiples of d
        self_consistency_mask = (d > 0)
        if self_consistency_mask.any():
            d_nonzero = d[self_consistency_mask]
            # Sample how many multiples of d to use for each sample
            # We want t to be k*d where k is a random integer such that t ∈ [0, 1-d]
            # This ensures t + d ≤ 1
            max_multiples = torch.floor((1.0 - d_nonzero) / d_nonzero).long()
            # Ensure max_multiples is at least 0 to avoid empty range
            max_multiples = torch.clamp(max_multiples, min=0)
            
            # For each sample, randomly choose k from [0, max_multiples] - vectorized
            # Generate random values [0, 1) for all samples
            random_vals = torch.rand_like(d_nonzero)
            # Scale to [0, max_multiples + 1) and floor to get integers [0, max_multiples]
            k_values = torch.floor(random_vals * (max_multiples.float() + 1))
            # Compute t = k * d for all samples
            t_self_consistency = k_values * d_nonzero
            
            t[self_consistency_mask] = t_self_consistency
        
        return t, d

    def loss(self, x1: torch.Tensor, *args_conditionals, **kwargs_conditionals):
        """Compute shortcut model loss with mixed flow matching and self-consistency objectives"""
        # t, d = self._generate_t_and_d(x1)
        t = self._generate_t(x1)
        d = self._generate_d(x1)
        x0 = self._generate_x0(x1)
        x_t = self._generate_xt(x0, x1, t)
        
        # Determine which samples use flow matching vs  self-consistency
        flow_matching_indices = (d == 0).nonzero(as_tuple=False).squeeze(-1)  # 75% of the time use d=0 (flow matching), 25% use self-consistency
        self_consistency_indices = (d > 0).nonzero(as_tuple=False).squeeze(-1)
        d[d == 0] = torch.rand_like(d[d == 0]) * self.fm_eps_max
        
        # Clear autocast cache for gradient computation
        torch.clear_autocast_cache()
        
        # Get model prediction
        s = self.reverse_fn(
            x_t,
            t * self.time_scale,
            *args_conditionals,
            d=2 * d * self.time_scale,
            **kwargs_conditionals,
        )

        # Compute component losses separately by selecting relevant indices
        flow_matching_loss_val = torch.tensor(0.0, device=d.device, dtype=torch.float32)
        self_consistency_loss_val = torch.tensor(0.0, device=d.device, dtype=torch.float32)
        
        # Flow matching component (for d=0 samples)
        if len(flow_matching_indices) > 0:
            # Select samples where d=0 and compute flow matching target only for these samples
            x0_flow = tree_tensor_map(lambda x: x[flow_matching_indices], x0)
            x1_flow = tree_tensor_map(lambda x: x[flow_matching_indices], x1)
            s_flow = tree_tensor_map(lambda x: x[flow_matching_indices], s)
            
            # Compute flow matching target only for selected samples
            flow_matching_target = self._generate_target(x0_flow, x1_flow)
            
            flow_matching_loss = optree.tree_broadcast_map(
                lambda fn, weight, pred, targ: weight * fn(pred, targ),
                self.loss_fn,
                self.loss_weights,
                s_flow,
                flow_matching_target,
            )
            flow_matching_loss_val = sum(optree.tree_flatten(flow_matching_loss)[0])
        
        # Shortcut self-consistency component (for d>0 samples)
        if len(self_consistency_indices) > 0:
            # Select samples where d>0 and compute self-consistency target only for these samples
            x_t_shortcut = tree_tensor_map(lambda x: x[self_consistency_indices], x_t)
            t_shortcut = t[self_consistency_indices]
            d_shortcut = d[self_consistency_indices]
            s_shortcut = tree_tensor_map(lambda x: x[self_consistency_indices], s)
            
            # Create conditional arguments for selected samples
            if self.batch_mode:
                args_conditionals_shortcut = args_conditionals
                kwargs_conditionals_shortcut = kwargs_conditionals
            else:
                args_conditionals_shortcut = tuple(
                    tree_tensor_map(lambda x: x[self_consistency_indices], arg) if torch.is_tensor(arg) else arg
                    for arg in args_conditionals
                )
                kwargs_conditionals_shortcut = {
                    k: (tree_tensor_map(lambda x: x[self_consistency_indices], v) if torch.is_tensor(v) else v)
                    for k, v in kwargs_conditionals.items()
                }
            
            # Compute self-consistency target only for selected samples
            self_consistency_target = self.compute_self_consistency_target(
                x_t_shortcut, t_shortcut, d_shortcut, 
                *args_conditionals_shortcut, **kwargs_conditionals_shortcut
            )
            
            self_consistency_loss = optree.tree_broadcast_map(
                lambda fn, weight, pred, targ: weight * fn(pred, targ),
                self.loss_fn,
                self.loss_weights,
                s_shortcut,
                self_consistency_target,
            )
            self_consistency_loss_val = sum(optree.tree_flatten(self_consistency_loss)[0])
        
        # Total loss is the sum of both components (linear combination)
        total_loss = flow_matching_loss_val + self.shortcut_loss_weight * self_consistency_loss_val
        
        # Create detailed loss breakdown
        detail_losses = {
            "flow_matching_loss": flow_matching_loss_val,
            "self_consistency_loss": self_consistency_loss_val,
        }
        return total_loss, detail_losses
        
    def _prepare_t_and_d(self, steps=None):
        """Prepare time sequence and step size for inference"""
        steps = self.inference_steps if steps is None else steps
        t_seq = np.linspace(0, 1, steps + 1)

        if self.no_shortcut:
            d = 0
        else:
            # Use uniform step size for inference
            d = 1 / steps

        if self.rescale_t:
            t_seq = t_seq / (1 + (self.rescale_t - 1) * (1 - t_seq))

        if self.reversed_timestamp:
            t_seq = 1 - t_seq

        return t_seq, d

    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate samples using shortcut model"""
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq, d = self._prepare_t_and_d()
     

        for x_t, t,v in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            d,
            *args_conditionals,
            **kwargs_conditionals,
        ):
         
            yield t, x_t, v,()
    
    
    def _generate_dynamics(
        self,
        x_t,
        t,
        d,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate dynamics for ODE solver"""
        t = torch.tensor(
            [t * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d = torch.tensor(
            [d * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )

        output = self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
        return output
    


    def generate(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        for _, xt, v ,_ in self.generate_iter(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        ):
           
            pass
        return xt


from faster_utils_ss import (
    derivative_approximation as faster_derivative_approximation,
    faster_cal_type,
    faster_formula,
    faster_init,
    faster_step_init,
)
class ShortCut_faster(ShortCut):
    def __init__(
        self,
        no_shortcut=False,
        self_consistency_prob=0.25,
        shortcut_loss_weight=1.0,
        self_consistency_cfg_strength=3.0,
        ratio_cfg_samples_in_self_consistency_target=0.5,
        fm_in_shortcut_target_prob=0.0,
        fm_eps_max=0,
        batch_mode=False,
        cfg_modalities=["shape"],
        **kwargs,
    ):

        super().__init__(**kwargs)
     
        self.faster_dic = None
        self.current = None
        self.prev_v = None
        self.ss_params = None

    def forward(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        
        self.faster_dic, self.current = faster_init(
            self.inference_steps,
            faster_interval=self.ss_params['ss_faster_stride'],
            max_order=self.ss_params['ss_order'],
            first_enhance=self.ss_params['ss_warmup'],
            end_enhance=24,
        )
        result = self.generate(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        )

        return result
 

    def generate(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        for _, xt, v ,_ in self.generate_iter(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            pass
        return xt
    
    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq, d = self._prepare_t_and_d()
        

        for x_t, t ,v in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            d,
            *args_conditionals,
            **kwargs_conditionals,
        ):  
            yield t, x_t, v, ()


    def _generate_dynamics(
        self,
        x_t,
        t,
        d,
        *args_conditionals,
        **kwargs_conditionals,
    ):

        t = torch.tensor(
            [t * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d = torch.tensor(
            [d * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )

        faster_cal_type(self.faster_dic, self.current)
        self.current['stream'] = 'final'
        self.current['layer'] = 'final'
        self.current['module'] = 'final'
        faster_step_init(self.faster_dic, self.current)

        if self.current['type'] == 'full':
            v = self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
            self.prev_v = v
            faster_derivative_approximation(self.faster_dic, self.current, v)

        elif self.current['type'] == 'faster':
            v = faster_formula(self.faster_dic, self.current, self.prev_v, beta=self.ss_params['ss_momentum_beta'])
            self.prev_v = v
            
        self.current['step'] += 1
        return  v
    

    def normalize_map(self,m):
        m_min = m.min()
        m_max = m.max()
        return (m - m_min) / (m_max - m_min + 1e-6)
    

# ——————————————————————————————————————————————————————————————————
# ⭐⭐封装 easy
class ShortCut_easy(ShortCut):
    def __init__(
        self,
        no_shortcut=False,
        self_consistency_prob=0.25,
        shortcut_loss_weight=1.0,
        self_consistency_cfg_strength=3.0,
        ratio_cfg_samples_in_self_consistency_target=0.5,
        fm_in_shortcut_target_prob=0.0,
        fm_eps_max=0,
        batch_mode=False,
        cfg_modalities=["shape"],
        **kwargs,
    ):


        super().__init__(**kwargs)
        # 传入解码器,一会用
        solver_method = "euler_easy_ss" # 传入一个重新包装后的求解器
        solver_kwargs = {} 
        solver_kwargs.setdefault("thresh", 2.0)
        solver_kwargs.setdefault("ret_steps", 4)
        # 2.0/6->11
        # 3.0/4 ->8
        # 3.0/2 ->4
        self._solver_method, self._solver = self._get_solver(
            solver_method, solver_kwargs
        )

        # 生成参数
        self.decoder = None

    
    # 1.⭐⭐forward直接调用的东西
    def forward(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
    
        result = self.generate(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        )
        print(self._solver.k_map.shape)
        

        self.k_map = self._solver.k_map.squeeze(0)



        print(self.k_map.shape)
        
        # self.save_kmap_to_html(self.k_map)
        # self.plot_kmap_distribution(self.k_map)
        # self.save_kmap_to_html_mean(self.k_map)
        # import pdb; pdb.set_trace()
        
        return result
 
    # 2.generate
    def generate(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        for _, xt, v ,_ in self.generate_iter(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            pass
        return xt
    
    # 3.generate_iter
    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate samples using shortcut model"""
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq, d = self._prepare_t_and_d()
        
        self.debug_history = {
            't': [],
            'metrics': {} 
        }
        self.k_map = None
        # print(self.LEADER)
        for x_t, t ,v in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            d,
            *args_conditionals,
            **kwargs_conditionals,
        ):  
            # print("time", t)
            # if t >= 0.88:
            #     latent = x_t['shape']
            #     grid = self.decode(latent)
            #     num_points = (grid > 0).sum().item()
            #     dir_path = "/data/wmq/sam-3d-objects/voxels"
            #     os.makedirs(dir_path, exist_ok=True)
            #     file_name = os.path.join(dir_path,f"{2}-cache-{num_points}.html")
            #     save_voxel_to_html(grid ,file_name)



            current_t = t.item() if torch.is_tensor(t) else t
            self.debug_history['t'].append(current_t)

            # 显式指定我们要记录的 key
            target_key = 'shape'
            if isinstance(x_t, dict) and target_key in x_t:
                if target_key not in self.debug_history['metrics']:
                    self.debug_history['metrics'][target_key] = {'x_l1': [], 'v_l1': []}
                
                # 计算 x_t['shape'] 的 L1 Mean
                x_val = x_t[target_key]
                x_l1 = x_val.abs().mean().item()
                self.debug_history['metrics'][target_key]['x_l1'].append(x_l1)

                # 计算 v['shape'] 的 L1 Mean
                if isinstance(v, dict) and target_key in v:
                    v_val = v[target_key]
                    v_l1 = v_val.abs().mean().item()
                    self.debug_history['metrics'][target_key]['v_l1'].append(v_l1)
                else:
                    self.debug_history['metrics'][target_key]['v_l1'].append(0.0)


            yield t, x_t, v, ()

    # 4.generate_iter
    def _generate_dynamics(
        self,
        x_t,
        t,
        d,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate dynamics for ODE solver"""
        t = torch.tensor(
            [t * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d = torch.tensor(
            [d * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        # 4.self.reverse_fn,得到预测的速度
        v = self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
        return  v
    
    # 解码出形状
    def decode(self,shape_latent):
        ss = self.decoder(
            shape_latent.permute(0, 2, 1)
            .contiguous()
            .view(shape_latent.shape[0], 8, 16, 16, 16)
        )
        # print("ss.shape", ss.shape) # torch.Size([1, 1, 64, 64, 64])
        grid_t = (ss > 0)
        return grid_t


    # 绘制x_t和v的图
    def plot_debug_statistics(self, save_path=None, x_ylim=(0.5,1.0), v_ylim=(0,1.8)):
        """
        绘制 generate_iter 过程中记录的 x_t 和 v 的 L1 均值变化曲线。
        
        Args:
            save_path (str, optional): 图片保存路径.
            x_ylim (tuple/list, optional): x_l1 图的 Y 轴范围, 如 (0, 2.0).
            v_ylim (tuple/list, optional): v_l1 图的 Y 轴范围, 如 (0, 0.5).
        """
        import matplotlib.pyplot as plt
        import math

        if not hasattr(self, 'debug_history') or not self.debug_history['t']:
            print("Warning: No debug history found. Please run generate_iter first.")
            return

        # 获取数据字典
        metrics = self.debug_history['metrics']
        keys = list(metrics.keys())
        num_keys = len(keys)

        if num_keys == 0:
            return

        # 设置绘图布局：每个 key 占一行，包含两个子图 (State L1, Velocity L1)
        fig, axes = plt.subplots(num_keys, 2, figsize=(12, 4 * num_keys), sharex=True)
        
        # 如果只有一个 key，axes 是一维数组，需要处理一下
        if num_keys == 1:
            axes = axes.reshape(1, -1)

        for i, key in enumerate(keys):
            data = metrics[key]
            
            # 生成横坐标索引：1, 2, 3, 4, ...
            num_steps = len(data['x_l1'])
            step_indices = list(range(1, num_steps + 1))

            ax_x = axes[i, 0]
            ax_v = axes[i, 1]

            # ================= 绘制 State x_t L1 =================
            ax_x.plot(step_indices, data['x_l1'], label=f'{key} x_l1', color='tab:blue', marker='.')
            ax_x.set_title(f"State L1 Mean: {key}")
            ax_x.set_ylabel("Mean(|x|)")
            ax_x.grid(True, alpha=0.3)
            ax_x.legend()
            
            # 手动设置 x 的 Y 轴范围
            if x_ylim is not None:
                ax_x.set_ylim(x_ylim)

            # ================= 绘制 Velocity v L1 =================
            ax_v.plot(step_indices, data['v_l1'], label=f'{key} v_l1', color='tab:orange', marker='.')
            ax_v.set_title(f"Velocity L1 Mean: {key}")
            ax_v.set_ylabel("Mean(|v|)")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend()

            # 手动设置 v 的 Y 轴范围
            if v_ylim is not None:
                ax_v.set_ylim(v_ylim)

            # 只在最后一行显示 X 轴标签
            if i == num_keys - 1:
                ax_x.set_xlabel("Step Index")
                ax_v.set_xlabel("Step Index")

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Debug plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()


    def save_kmap_to_html(self, k_map, save_path="k_map_hard_focus.html"):
        import numpy as np
        import plotly.graph_objects as go
        import os
        import torch

        # --- 1. 数据预处理 ---
        if isinstance(k_map, torch.Tensor):
            k_data = k_map.detach().cpu().numpy()
        else:
            k_data = np.array(k_map)
        
        k_data = k_data.flatten()
        if k_data.size != 4096:
            print(f"Error: expected 4096 points, got {k_data.size}.")
            return

        volume = k_data.reshape(16, 16, 16)
        vals_abs = np.abs(volume).flatten()

        # --- 2. 计算自定义百分比排名 ---
        ranks = np.argsort(np.argsort(vals_abs))
        percentiles = ranks / (len(vals_abs) - 1)
        
        # 定义新的分箱逻辑：
        # 0: 0-40% (隐藏), 1: 40-60%, 2: 60-80%, 
        # 3: 80-85%, 4: 85-90%, 5: 90-95%, 6: 95-100% (高能区)
        bins = np.zeros_like(percentiles, dtype=int)
        bins[percentiles >= 0.40] = 1
        bins[percentiles >= 0.60] = 2
        bins[percentiles >= 0.80] = 3
        bins[percentiles >= 0.85] = 4
        bins[percentiles >= 0.90] = 5
        bins[percentiles >= 0.95] = 6
        
        num_bins = 7 # 总共 7 档

        # --- 3. 生成坐标并过滤 ---
        X, Y, Z = np.mgrid[0:16, 0:16, 0:16]
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        
        # 只保留 40% 以上的点 (即 bins >= 1)
        valid_mask = bins >= 4
        
        X_f, Y_f, Z_f = X[valid_mask], Y[valid_mask], Z[valid_mask]
        bins_f = bins[valid_mask]
        vals_f = vals_abs[valid_mask]
        perc_f = percentiles[valid_mask]

        # --- 4. 定义颜色映射 (增加高能区细节) ---
        colors_hex = [
            '#313695', # 0: 0-40% (Hidden)
            '#abd9e9', # 1: 40-60% (Light Blue)
            '#ffffbf', # 2: 60-80% (Pale Yellow)
            # --- 以下为 Top 20% 的细分 ---
            '#fdae61', # 3: 80-85% (Orange)
            '#f46d43', # 4: 85-90% (Light Red)
            '#d73027', # 5: 90-95% (Red)
            '#a50026'  # 6: 95-100% (Dark Red/Extreme)
        ]
        
        step_colorscale = []
        for i in range(num_bins):
            step_colorscale.append([i / num_bins, colors_hex[i]])
            step_colorscale.append([(i + 1) / num_bins, colors_hex[i]])

        # --- 5. 创建 Plotly 图形 ---
        fig = go.Figure(data=[go.Scatter3d(
            x=X_f, y=Y_f, z=Z_f,
            mode='markers',
            marker=dict(
                symbol='square',
                size=5,
                color=bins_f,
                colorscale=step_colorscale,
                cmin=0,
                cmax=num_bins,
                opacity=0.9,
                colorbar=dict(
                    title="Score Rank",
                    tickvals=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                    ticktext=["40-60%", "60-80%", "80-85%", "85-90%", "90-95%", "95-100%!"],
                    tickfont=dict(color="white"),
                )
            ),
            text=[f"Val: {v:.4f}<br>Top: {(1-p):.1%}" for v, p in zip(vals_f, perc_f)],
            hoverinfo='text'
        )])

        # --- 6. 布局优化 ---
        fig.update_layout(
            title="Focus K-Map: Precision Top 20%",
            scene=dict(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="black"
            ),
            paper_bgcolor="black",
            margin=dict(r=0, l=0, b=0, t=50)
        )

        fig.write_html(save_path)
        print(f"Rendering complete. High-energy regions were subdivided. Saved to: {save_path}")

    # 绘制kmap数值分布
    def plot_kmap_distribution(self, k_map,save_path="kmap_distribution.png"):
        """
        统计 4096 个 Token 的 k 值分布，绘制排序曲线和分布密度图。
        """
        # 1. 获取并处理数据
        if k_map is None:
            print("Warning: k_map is None, skip plotting.")
            return

        # 确保是 [4096] 的一维 numpy 数组
        k_map = k_map.detach().cpu().numpy().flatten()
        
        if len(k_map) != 4096:
            print(f"Warning: Expected 4096 tokens, got {len(k_map)}")

        # 2. 统计核心指标
        mean_val = np.mean(k_map)
        median_val = np.median(k_map)
        max_val = np.max(k_map)
        min_val = np.min(k_map)
        
        # 计算关键分位数 (用于之前的 20% / 80% 策略)
        p20 = np.percentile(k_map, 20)
        p80 = np.percentile(k_map, 80)

        # 3. 创建画布 (左图：排序曲线，右图：密度分布)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ==========================
        # 图 1: 排序后的 K 值曲线 (Sorted Line Chart)
        # ==========================
        # 作用：直观展示 Token 难度的增长趋势，方便找截断点
        sorted_k = np.sort(k_map)
        x_axis = np.arange(len(sorted_k))
        
        ax1 = axes[0]
        ax1.plot(x_axis, sorted_k, color='#1f77b4', linewidth=2, label='Sorted k values')
        
        # 标注关键线
        ax1.axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
        ax1.axvline(4096 * 0.2, color='green', linestyle=':', alpha=0.7, label=f'20% (Easy): k={p20:.2f}')
        ax1.axvline(4096 * 0.8, color='orange', linestyle=':', alpha=0.7, label=f'80% (Hard): k={p80:.2f}')
        
        # 填充区域 (模拟你的策略)
        ax1.fill_between(x_axis, 0, sorted_k, where=(x_axis < 4096*0.2), color='green', alpha=0.1)
        ax1.fill_between(x_axis, 0, sorted_k, where=(x_axis > 4096*0.8), color='orange', alpha=0.1)

        ax1.set_title(f"Sorted Token Difficulty (N={len(k_map)})")
        ax1.set_xlabel("Token Rank (Easy -> Hard)")
        ax1.set_ylabel("k Value (Diff Rate)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ==========================
        # 图 2: 分布密度图 (Distribution Density)
        # ==========================
        # 作用：展示 k 值主要集中在哪个区间
        ax2 = axes[1]
        
        # 尝试使用 Seaborn 画平滑曲线 (KDE)，如果没有则用 Matplotlib 直方图
        try:
            sns.kdeplot(k_map, ax=ax2, fill=True, color="purple", alpha=0.3, linewidth=2)
            ax2.set_ylabel("Density")
        except NameError:
            # Fallback to matplotlib histogram
            ax2.hist(k_map, bins=50, color='purple', alpha=0.5, density=True)
            ax2.set_ylabel("Frequency")

        # 标注均值
        ax2.axvline(mean_val, color='red', linestyle='--', label='Mean')
        ax2.axvline(median_val, color='blue', linestyle='-.', label='Median')
        
        ax2.set_title("Distribution of k Values")
        ax2.set_xlabel("k Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 4. 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"K-Map statistics plot saved to: {save_path}")
        print(f"   [Stats] Mean: {mean_val:.4f}, Median: {median_val:.4f}")
        print(f"   [Quantile] 20% quantile (easy threshold): {p20:.4f}")
        print(f"   [Quantile] 80% quantile (hard threshold): {p80:.4f}")
        plt.close()

    # 根据均值分点
    def save_kmap_to_html_mean(self, k_map, save_path="k_map_mean_based.html", multipliers=(1.25, 1.5, 2.0)):
        """
        基于均值倍率可视化 K-Map。
        Args:
            k_map: 输入数据 (4096 flattened or 16x16x16)
            save_path: 保存路径
            multipliers: 一个包含3个浮点数的元组，分别对应三档阈值的倍率。
                        默认 (1.0, 1.5, 2.0) 代表 >均值, >1.5倍均值, >2倍均值。
        """
        # --- 1. 数据预处理 ---
        if isinstance(k_map, torch.Tensor):
            k_data = k_map.detach().cpu().numpy()
        else:
            k_data = np.array(k_map)
        
        k_data = k_data.flatten()
        if k_data.size != 4096:
            print(f"Error: expected 4096 points, got {k_data.size}.")
            return

        # 恢复 3D 结构并取绝对值
        volume = k_data.reshape(16, 16, 16)
        vals_abs = np.abs(volume).flatten()
        
        # 计算统计指标
        mean_val = np.mean(vals_abs)
        max_val = np.max(vals_abs)
        
        # 防止全0数据的除法错误
        if mean_val == 0:
            print("Warning: data mean is 0; heatmap cannot be generated.")
            return

        # --- 2. 基于均值的分箱逻辑 ---
        # 解包倍率参数
        m1, m2, m3 = multipliers
        
        # 初始化 bins，默认为 0 (隐藏/背景)
        bins = np.zeros_like(vals_abs, dtype=int)
        
        # 逻辑：数值越大，Bin ID 越高，后者覆盖前者
        # Bin 1: > 1.0 * Mean
        bins[vals_abs >= mean_val * m1] = 1
        # Bin 2: > 1.5 * Mean
        bins[vals_abs >= mean_val * m2] = 2
        # Bin 3: > 2.0 * Mean
        bins[vals_abs >= mean_val * m3] = 3
        
        # 总共有 4 种状态：0(小于均值), 1(一档), 2(二档), 3(三档)
        num_bins = 4 

        # --- 3. 生成坐标并过滤 ---
        X, Y, Z = np.mgrid[0:16, 0:16, 0:16]
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        
        # 过滤：只显示大于均值点 (即 bins >= 1)
        # 如果你想连小于均值的也显示，可以改这里，但在3D散点图中通常只看高响应区
        valid_mask = bins >= 1
        
        X_f, Y_f, Z_f = X[valid_mask], Y[valid_mask], Z[valid_mask]
        bins_f = bins[valid_mask]
        vals_f = vals_abs[valid_mask]

        # --- 4. 定义颜色映射 (3档区分) ---
        # 0: Hidden (不显示), 1: Blue/Cyan, 2: Orange, 3: Red
        colors_hex = [
            '#ffffbf',
            '#fdae61',
            '#d73027',
            '#a50026',
        ]
        
        # 构建离散 Colorscale
        step_colorscale = []
        for i in range(num_bins):
            step_colorscale.append([i / num_bins, colors_hex[i]])
            step_colorscale.append([(i + 1) / num_bins, colors_hex[i]])

        # --- 5. 创建 Plotly 图形 ---
        fig = go.Figure(data=[go.Scatter3d(
            x=X_f, y=Y_f, z=Z_f,
            mode='markers',
            marker=dict(
                symbol='square',
                size=6, #稍微调大一点点以便观察
                color=bins_f,
                colorscale=step_colorscale,
                cmin=0,
                cmax=num_bins,
                opacity=0.9,
                colorbar=dict(
                    title="Mean Multiplier",
                    tickvals=[1.5, 2.5, 3.5], # 刻度位置在色块中间
                    ticktext=[
                        f">{m1}x Mean", 
                        f">{m2}x Mean", 
                        f">{m3}x Mean"
                    ],
                    tickfont=dict(color="white"),
                )
            ),
            # Hover 信息显示：具体数值 + 是均值的多少倍
            text=[f"Val: {v:.4f}<br>Ratio: {v/mean_val:.2f}x Mean" for v in vals_f],
            hoverinfo='text'
        )])

        # --- 6. 布局优化 ---
        fig.update_layout(
            title=f"Mean-Based K-Map (Mean={mean_val:.4f})",
            scene=dict(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="black"
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            margin=dict(r=0, l=0, b=0, t=50)
        )
        fig.write_html(save_path)
        print(f"Rendering complete. Mean-based multiplier bins {multipliers} saved to: {save_path}")
    


# ——————————————————————————————————————————————————————————————————
# ⭐⭐封装 taylorseer
from taylor_utils_ss import (
    derivative_approximation,
    taylor_cal_type,
    taylor_cache_init,
    taylor_formula,
    taylor_init,
)
class ShortCut_taylorseer(ShortCut):
    def __init__(
        self,
        no_shortcut=False,
        self_consistency_prob=0.25,
        shortcut_loss_weight=1.0,
        self_consistency_cfg_strength=3.0,
        ratio_cfg_samples_in_self_consistency_target=0.5,
        fm_in_shortcut_target_prob=0.0,
        fm_eps_max=0,
        batch_mode=False,
        cfg_modalities=["shape"],
        **kwargs,
    ):

        super().__init__(**kwargs)
        # 传入解码器,一会用
        # 生成参数
        self.decoder = None
        self.taylor_dic, self.current = taylor_init(self.inference_steps)

        self.k_map_shape = None
        self.prev_v_true = None # 前一次真实计算的结果
        self.prev_v = None# 前一次的结果

    
    # 1.⭐⭐forward直接调用的东西
    def forward(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        
        self.taylor_dic, self.current = taylor_init(self.inference_steps)
        result = self.generate(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        )

        # k_map_shape = self.k_map_shape.squeeze(0)
        # print(k_map_shape.shape)
        
        # self.plot_kmap_distribution(self.k_map_shape,"distr_important_tokens.png")
        # self.save_kmap_to_html_mean(self.k_map_shape,"mean_important_tokens.html")
        # import pdb; pdb.set_trace()

        return result
 
    # 2.generate
    def generate(self, x_shape, x_device, *args_conditionals, **kwargs_conditionals):
        for _, xt, v ,_ in self.generate_iter(
            x_shape,
            x_device,
            *args_conditionals,
            **kwargs_conditionals,
        ):
            pass
        print("Taylor computed steps", self.current['activated_steps'])
        return xt
    
    # 3.generate_iter
    def generate_iter(
        self,
        x_shape,
        x_device,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate samples using shortcut model"""
        x_0 = self._generate_noise(x_shape, x_device)
        t_seq, d = self._prepare_t_and_d()
        
        self.debug_history = {
            't': [],
            'metrics': {} 
        }

        for x_t, t ,v in self._solver.solve_iter(
            self._generate_dynamics,
            x_0,
            t_seq,
            d,
            *args_conditionals,
            **kwargs_conditionals,
        ):  

            current_t = t.item() if torch.is_tensor(t) else t
            self.debug_history['t'].append(current_t)


            yield t, x_t, v, ()

    # 4.generate_iter
    def _generate_dynamics(
        self,
        x_t,
        t,
        d,
        *args_conditionals,
        **kwargs_conditionals,
    ):
        """Generate dynamics for ODE solver"""
        t = torch.tensor(
            [t * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )
        d = torch.tensor(
            [d * self.time_scale], device=_get_device(x_t), dtype=torch.float32
        )


        taylor_cal_type(self.taylor_dic, self.current)
        self.current['stream'] = 'final'
        self.current['layer'] = 'final'
        self.current['module'] = 'final'
        taylor_cache_init(self.taylor_dic, self.current)

        v = {}
        # 4.self.reverse_fn,得到预测的速度
        if self.current['type'] == 'full':
            v = self.reverse_fn(x_t, t, *args_conditionals, d=d, **kwargs_conditionals)
           
            # ------------------------------------------------------------------------
            # 计算并更新k_map_shape的
            if self.prev_v_true is not None:
                prev_v_shape = self.prev_v_true['shape']

                if prev_v_shape is not None:
                    v_shape = v['shape']
                    v_shape = torch.norm(v_shape, p=2, dim=2, keepdim=True)
                    accel_shape = self.normalize_map(torch.norm(v_shape - prev_v_shape, p=2, dim=2, keepdim=True))
                    accel_weight = getattr(self, 'ACCELERATION_WEIGHT', 0.3)
                    current_k_map_shape = (accel_weight * accel_shape) + ((1.0 - accel_weight) * v_shape)

                    # EMA 更新 k_map
                    if self.k_map_shape is None:
                        self.k_map_shape = current_k_map_shape
                    else:
                        self.k_map_shape = 0.8 * self.k_map_shape + 0.2 * current_k_map_shape

            # 更新历史
            self.prev_v_true = v
            self.prev_v = v

            # ------------------------------------------------------------------------
            # 计算梯度
            derivative_approximation(self.taylor_dic, self.current, v)
            print("Do not skip")

        elif self.current['type'] == 'taylor':
            print("Skip")
            v = taylor_formula(self.taylor_dic, self.current, self.prev_v_true)
            self.prev_v = v
            

        self.current['step'] += 1
        return  v
    
    # 解码出形状
    def decode(self,shape_latent):
        ss = self.decoder(
            shape_latent.permute(0, 2, 1)
            .contiguous()
            .view(shape_latent.shape[0], 8, 16, 16, 16)
        )
        # print("ss.shape", ss.shape) # torch.Size([1, 1, 64, 64, 64])
        grid_t = (ss > 0)
        return grid_t
    
    # 绘制 easy 步数变化图
    def plot_debug_statistics(self, save_path=None, x_ylim=(0.5,1.0), v_ylim=(0,3.0)):

        """
        绘制 generate_iter 过程中记录的 x_t 和 v 的 L1 均值变化曲线。
        
        Args:
            save_path (str, optional): 图片保存路径.
            x_ylim (tuple/list, optional): x_l1 图的 Y 轴范围, 如 (0, 2.0).
            v_ylim (tuple/list, optional): v_l1 图的 Y 轴范围, 如 (0, 0.5).
        """
        import matplotlib.pyplot as plt
        import math

        if not hasattr(self, 'debug_history') or not self.debug_history['t']:
            print("Warning: No debug history found. Please run generate_iter first.")
            return

        # 获取数据字典
        metrics = self.debug_history['metrics']
        keys = list(metrics.keys())
        num_keys = len(keys)

        if num_keys == 0:
            return

        # 设置绘图布局：每个 key 占一行，包含两个子图 (State L1, Velocity L1)
        fig, axes = plt.subplots(num_keys, 2, figsize=(12, 4 * num_keys), sharex=True)
        
        # 如果只有一个 key，axes 是一维数组，需要处理一下
        if num_keys == 1:
            axes = axes.reshape(1, -1)

        for i, key in enumerate(keys):
            data = metrics[key]
            
            # 生成横坐标索引：1, 2, 3, 4, ...
            num_steps = len(data['x_l1'])
            step_indices = list(range(1, num_steps + 1))

            ax_x = axes[i, 0]
            ax_v = axes[i, 1]

            # ================= 绘制 State x_t L1 =================
            ax_x.plot(step_indices, data['x_l1'], label=f'{key} x_l1', color='tab:blue', marker='.')
            ax_x.set_title(f"State L1 Mean: {key}")
            ax_x.set_ylabel("Mean(|x|)")
            ax_x.grid(True, alpha=0.3)
            ax_x.legend()
            
            # 手动设置 x 的 Y 轴范围
            if x_ylim is not None:
                ax_x.set_ylim(x_ylim)

            # ================= 绘制 Velocity v L1 =================
            ax_v.plot(step_indices, data['v_l1'], label=f'{key} v_l1', color='tab:orange', marker='.')
            ax_v.set_title(f"Velocity L1 Mean: {key}")
            ax_v.set_ylabel("Mean(|v|)")
            ax_v.grid(True, alpha=0.3)
            ax_v.legend()

            # 手动设置 v 的 Y 轴范围
            if v_ylim is not None:
                ax_v.set_ylim(v_ylim)

            # 只在最后一行显示 X 轴标签
            if i == num_keys - 1:
                ax_x.set_xlabel("Step Index")
                ax_v.set_xlabel("Step Index")

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Debug plot saved to {save_path}")
        else:
            plt.show()
            
        plt.close()

    # 归一化
    def normalize_map(self,m):
        m_min = m.min()
        m_max = m.max()
        return (m - m_min) / (m_max - m_min + 1e-6)
    
    
    # 绘制kmap数值分布
    def plot_kmap_distribution(self, k_map,save_path="kmap_distribution.png"):
        """
        统计 4096 个 Token 的 k 值分布，绘制排序曲线和分布密度图。
        """
        # 1. 获取并处理数据
        if k_map is None:
            print("Warning: k_map is None, skip plotting.")
            return

        # 确保是 [4096] 的一维 numpy 数组
        k_map = k_map.detach().cpu().numpy().flatten()
        
        if len(k_map) != 4096:
            print(f"Warning: Expected 4096 tokens, got {len(k_map)}")

        # 2. 统计核心指标
        mean_val = np.mean(k_map)
        median_val = np.median(k_map)
        max_val = np.max(k_map)
        min_val = np.min(k_map)
        
        # 计算关键分位数 (用于之前的 20% / 80% 策略)
        p20 = np.percentile(k_map, 20)
        p80 = np.percentile(k_map, 80)

        # 3. 创建画布 (左图：排序曲线，右图：密度分布)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ==========================
        # 图 1: 排序后的 K 值曲线 (Sorted Line Chart)
        # ==========================
        # 作用：直观展示 Token 难度的增长趋势，方便找截断点
        sorted_k = np.sort(k_map)
        x_axis = np.arange(len(sorted_k))
        
        ax1 = axes[0]
        ax1.plot(x_axis, sorted_k, color='#1f77b4', linewidth=2, label='Sorted k values')
        
        # 标注关键线
        ax1.axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
        ax1.axvline(4096 * 0.2, color='green', linestyle=':', alpha=0.7, label=f'20% (Easy): k={p20:.2f}')
        ax1.axvline(4096 * 0.8, color='orange', linestyle=':', alpha=0.7, label=f'80% (Hard): k={p80:.2f}')
        
        # 填充区域 (模拟你的策略)
        ax1.fill_between(x_axis, 0, sorted_k, where=(x_axis < 4096*0.2), color='green', alpha=0.1)
        ax1.fill_between(x_axis, 0, sorted_k, where=(x_axis > 4096*0.8), color='orange', alpha=0.1)

        ax1.set_title(f"Sorted Token Difficulty (N={len(k_map)})")
        ax1.set_xlabel("Token Rank (Easy -> Hard)")
        ax1.set_ylabel("k Value (Diff Rate)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ==========================
        # 图 2: 分布密度图 (Distribution Density)
        # ==========================
        # 作用：展示 k 值主要集中在哪个区间
        ax2 = axes[1]
        
        # 尝试使用 Seaborn 画平滑曲线 (KDE)，如果没有则用 Matplotlib 直方图
        try:
            sns.kdeplot(k_map, ax=ax2, fill=True, color="purple", alpha=0.3, linewidth=2)
            ax2.set_ylabel("Density")
        except NameError:
            # Fallback to matplotlib histogram
            ax2.hist(k_map, bins=50, color='purple', alpha=0.5, density=True)
            ax2.set_ylabel("Frequency")

        # 标注均值
        ax2.axvline(mean_val, color='red', linestyle='--', label='Mean')
        ax2.axvline(median_val, color='blue', linestyle='-.', label='Median')
        
        ax2.set_title("Distribution of k Values")
        ax2.set_xlabel("k Value")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 4. 保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"K-Map statistics plot saved to: {save_path}")
        print(f"   [Stats] Mean: {mean_val:.4f}, Median: {median_val:.4f}")
        print(f"   [Quantile] 20% quantile (easy threshold): {p20:.4f}")
        print(f"   [Quantile] 80% quantile (hard threshold): {p80:.4f}")
        plt.close()

    # 根据均值分点
    def save_kmap_to_html_mean(self, k_map, save_path="k_map_mean_based.html", multipliers=(1.25, 1.5, 2.0)):
        """
        基于均值倍率可视化 K-Map。
        Args:
            k_map: 输入数据 (4096 flattened or 16x16x16)
            save_path: 保存路径
            multipliers: 一个包含3个浮点数的元组，分别对应三档阈值的倍率。
                        默认 (1.0, 1.5, 2.0) 代表 >均值, >1.5倍均值, >2倍均值。
        """
        # --- 1. 数据预处理 ---
        if isinstance(k_map, torch.Tensor):
            k_data = k_map.detach().cpu().numpy()
        else:
            k_data = np.array(k_map)
        
        k_data = k_data.flatten()
        if k_data.size != 4096:
            print(f"Error: expected 4096 points, got {k_data.size}.")
            return

        # 恢复 3D 结构并取绝对值
        volume = k_data.reshape(16, 16, 16)
        vals_abs = np.abs(volume).flatten()
        
        # 计算统计指标
        mean_val = np.mean(vals_abs)
        max_val = np.max(vals_abs)
        
        # 防止全0数据的除法错误
        if mean_val == 0:
            print("Warning: data mean is 0; heatmap cannot be generated.")
            return

        # --- 2. 基于均值的分箱逻辑 ---
        # 解包倍率参数
        m1, m2, m3 = multipliers
        
        # 初始化 bins，默认为 0 (隐藏/背景)
        bins = np.zeros_like(vals_abs, dtype=int)
        
        # 逻辑：数值越大，Bin ID 越高，后者覆盖前者
        # Bin 1: > 1.0 * Mean
        bins[vals_abs >= mean_val * m1] = 1
        # Bin 2: > 1.5 * Mean
        bins[vals_abs >= mean_val * m2] = 2
        # Bin 3: > 2.0 * Mean
        bins[vals_abs >= mean_val * m3] = 3
        
        # 总共有 4 种状态：0(小于均值), 1(一档), 2(二档), 3(三档)
        num_bins = 4 

        # --- 3. 生成坐标并过滤 ---
        X, Y, Z = np.mgrid[0:16, 0:16, 0:16]
        X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()
        
        # 过滤：只显示大于均值点 (即 bins >= 1)
        # 如果你想连小于均值的也显示，可以改这里，但在3D散点图中通常只看高响应区
        valid_mask = bins >= 1
        
        X_f, Y_f, Z_f = X[valid_mask], Y[valid_mask], Z[valid_mask]
        bins_f = bins[valid_mask]
        vals_f = vals_abs[valid_mask]

        # --- 4. 定义颜色映射 (3档区分) ---
        # 0: Hidden (不显示), 1: Blue/Cyan, 2: Orange, 3: Red
        colors_hex = [
            '#ffffbf',
            '#fdae61',
            '#d73027',
            '#a50026',
        ]
        
        # 构建离散 Colorscale
        step_colorscale = []
        for i in range(num_bins):
            step_colorscale.append([i / num_bins, colors_hex[i]])
            step_colorscale.append([(i + 1) / num_bins, colors_hex[i]])

        # --- 5. 创建 Plotly 图形 ---
        fig = go.Figure(data=[go.Scatter3d(
            x=X_f, y=Y_f, z=Z_f,
            mode='markers',
            marker=dict(
                symbol='square',
                size=6, #稍微调大一点点以便观察
                color=bins_f,
                colorscale=step_colorscale,
                cmin=0,
                cmax=num_bins,
                opacity=0.9,
                colorbar=dict(
                    title="Mean Multiplier",
                    tickvals=[1.5, 2.5, 3.5], # 刻度位置在色块中间
                    ticktext=[
                        f">{m1}x Mean", 
                        f">{m2}x Mean", 
                        f">{m3}x Mean"
                    ],
                    tickfont=dict(color="white"),
                )
            ),
            # Hover 信息显示：具体数值 + 是均值的多少倍
            text=[f"Val: {v:.4f}<br>Ratio: {v/mean_val:.2f}x Mean" for v in vals_f],
            hoverinfo='text'
        )])

        # --- 6. 布局优化 ---
        fig.update_layout(
            title=f"Mean-Based K-Map (Mean={mean_val:.4f})",
            scene=dict(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor="black"
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            margin=dict(r=0, l=0, b=0, t=50)
        )
        fig.write_html(save_path)
        print(f"Rendering complete. Mean-based multiplier bins {multipliers} saved to: {save_path}")

    
