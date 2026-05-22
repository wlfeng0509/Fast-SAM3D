from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from ...modules import sparse as sp

# # Utility function for subtraction.
# def sub(a, b):

#     assert a.feats.shape == b.feats.shape, "Feature dimensions must match"

#     new_feats = a.feats - b.feats  # Regular Tensor subtraction
#     new_coords = a.coords          # Keep coordinates unchanged

#     # Rewrap as SparseTensor
#     return sp.SparseTensor(feats=new_feats, coords=new_coords)


# Base sampler for SSG variants.
class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    # Data conversion.
    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    # Model forward pass.
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        return model(x_t, t, cond, **kwargs)

    # Single model forward pass plus data conversion.
    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    # Main sampling loop.
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        # Generate t_seq from 1 to 0.
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        
        print("t_pairs ,{t_pairs}" )

        # Iterate over each timestep.
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            
            
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)

# Sparse-structure sampler.
class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        # Apply CFG.
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)


# Taylor sampler.
from taylor_utils_ss import (
    derivative_approximation as taylor_derivative_approximation,
    taylor_cache_init,
    taylor_cal_type,
    taylor_formula,
    taylor_init,
)
class FlowEulerSampler_taylor(FlowEulerSampler):
    def __init__(
        self,
        sigma_min: float,
    ):
        super().__init__(sigma_min)
        self.cache_dic = None
        self.current = None
        self.prev_v = None

    # Override the base implementation.
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        sample = noise
        # Generate t_seq from 1 to 0.
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        
        print(f"t_pairs ,{len(t_pairs)}" )
        self.cache_dic, self.current = taylor_init(
            steps,
            taylor_interval=4,
            max_order=1,
            first_enhance=2,
            end_enhance=24,
        )
        # Iterate over each timestep.
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):

            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            # keys :   torch.Size([1, 8, 16, 16, 16]).torch.Size([1, 8, 16, 16, 16])
            # print(f"✅ keys :   {out['pred_x_prev'].shape}.{out['pred_x_0'].shape}")

            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        print("Activated steps:   ", self.current['activated_steps'])
        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):  
        
        # Determine the current sampling type.
        taylor_cal_type(self.cache_dic, self.current)
        self.current['stream'] = 'final'
        self.current['layer'] = 'final'
        self.current['module'] = 'final'
        taylor_cache_init(self.cache_dic, self.current)

        output = None
        # Reuse cached values according to the current type.
        print(self.current['type'])
        if self.current['type'] == 'full':
            pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
            pred_x_prev = x_t - (t - t_prev) * pred_v
            self.prev_v = pred_v
            taylor_derivative_approximation(self.cache_dic, self.current, pred_v)
            output = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

        elif self.current['type'] == 'taylor':
            pred_v = taylor_formula(self.cache_dic, self.current, self.prev_v, beta=0.5)
            pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
            pred_x_prev = x_t - (t - t_prev) * pred_v
            self.prev_v = pred_v
            output = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
        
        self.current['step'] += 1
        return output
    
    # Model forward pass.
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        
        output = model(x_t, t, cond, **kwargs)
        return output


    # Single model forward pass plus data conversion.
    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

class FlowEulerGuidanceIntervalSampler_taylor(GuidanceIntervalSamplerMixin, FlowEulerSampler_taylor):
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        
        # Apply CFG.
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)



# Faster sampler.
from token_slat.token_leader import TokenLeader
from token_slat.token_argparser import parse_token_args
from token_slat.selection import AdvancedStabilityTracker
from faster_utils_slat import faster_cal_type, faster_init
class FlowEulerSampler_faster(FlowEulerSampler):
    def __init__(
        self,
        sigma_min: float,
    ):
        super().__init__(sigma_min)

        self.LEADER = TokenLeader()
        self.stability_tracker = AdvancedStabilityTracker()
        self.args = parse_token_args()
        self.coords_scores = None

        # Set parameters.
        self.thresh = 5.0
        self.ret_steps = 2
        self.carving_ratio = 0.10
        self.dir_weight = 0.5
        self.cache_dic, self.current = faster_init(25)
        self.cache_dic['thresh'] = self.thresh 
        self.cache_dic['dir_weight'] =  self.dir_weight
        self.cache_dic['first_enhance'] = self.ret_steps
    
    # Inject coords_scores.
    def set_coords_scores(self,coords_scores):
        self.coords_scores = coords_scores

    # Initialize cache.
    def _init_token_state(self, x_t_shape, device, args, model):
        self.LEADER.set_parameters(args)
        if hasattr(model, 'dtype'):
             self.model_dtype = model.dtype
        elif hasattr(model, 'parameters'):
            try:
                self.model_dtype = next(model.parameters()).dtype
            except StopIteration:
                pass
  
        self.stability_tracker.reset(device = device, num_tokens = x_t_shape[0],latent_channels = x_t_shape[1])
        self.stability_tracker.coords_scores = self.coords_scores


    # Override the base implementation.
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        sample = noise
        # Generate t_seq from 1 to 0.
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        self.cache_dic, self.current = faster_init(steps)
        self.cache_dic['thresh'] = self.thresh
        self.cache_dic['dir_weight'] = self.dir_weight
        self.cache_dic['first_enhance'] = self.ret_steps
        
        # sample_feats:torch.Size([15201, 8]),noise_coords:torch.Size([15201, 4])
        print(f"sample_feats:{sample.feats.shape},noise_coords:{sample.coords.shape}")
  
        N,C = sample.feats.shape
        self.args.effective_steps = steps
        self.args.full_sampling_end_steps = int(np.ceil(steps * self.args.full_sampling_end_ratio))
        self.args.anchor_step = max(1, int(np.floor(steps * self.args.anchor_ratio)))
        self._init_token_state((N, C), sample.device, self.args, model)

        self.LEADER.total_tokens = N
        self.LEADER.schedule_is_set = True

        self.coords_raw = sample.coords
       
        # Iterate over each timestep.
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            cache = self.cache_dic['cache']
            self.current['is_token_active'] = False
            current_step = self.LEADER.current_step

            print(f"Timestep: {current_step},{self.LEADER.full_sampling_steps}")
            if self.current['use_token'] and cache['prev_v'] is not None and current_step >= self.LEADER.full_sampling_steps:
                self.current['num_to_skip'] = int(self.carving_ratio * N)
                if self.current['num_to_skip'] > 0 and self.current['num_to_skip'] < N:
                    self.current['is_token_active'] = True

            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            # keys :   torch.Size([1, 8, 16, 16, 16]).torch.Size([1, 8, 16, 16, 16])
            # print(f"✅ keys :   {out['pred_x_prev'].shape}.{out['pred_x_0'].shape}")

            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)

        print("Activated steps:   ", self.current['activated_steps'])
        ret.samples = sample
        return ret

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):  
        output = None
        should_calc = faster_cal_type(self.cache_dic, self.current, x_t.feats)
        velocity = None

        # print(f"kwargs.keys,{kwargs.keys()}")
        print(should_calc)
        if should_calc:
            print("Parameters", self.current['is_token_active'], self.current['use_token'])
            if self.current['is_token_active'] and self.current['use_token']:
                coords_scores = self.stability_tracker.coords_scores

                # print(coords_scores)
                # import pdb;pdb.set_trace()
                self.current['cached_indices'],  self.current['fast_update_indices'] = self.stability_tracker.update_and_select_combined(self.cache_dic['cache']['prev_v'], self.current['num_to_skip'],t=0, coords_scores = coords_scores,spatial_weight=0.3)
                
                # torch.Size([1520]) torch.Size([13681])
                # print(self.current['cached_indices'].shape,self.current['fast_update_indices'].shape)
                # import pdb;pdb.set_trace()
                # Select tokens.
                x_input_feats = x_t.feats[self.current['fast_update_indices'], :] if self.current['is_token_active'] else x_t.feats
                x_input_coords = x_t.coords[self.current['fast_update_indices'], :] if self.current['is_token_active'] else x_t.coords
                x_input = sp.SparseTensor(feats=x_input_feats, coords=x_input_coords)
                
                print(f"Token shapes before/after selection: {x_t.feats.shape},{x_input_feats.shape}")
                pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_input, t, cond, **kwargs)
                velocity_feats  = pred_v.feats
            else:
                pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
                velocity_feats  = pred_v.feats
            

            # Restore the original shape.
            if self.current['is_token_active'] and self.current['use_token']:
                final_v_tokens = self.cache_dic['cache']['prev_v'].clone() 
                final_v_tokens[self.current['fast_update_indices'], :] = velocity_feats.to(final_v_tokens.dtype)
                velocity_feats = final_v_tokens

            prev_x = self.cache_dic['cache']['prev_x']
            prev_prev_x = self.cache_dic['cache']['prev_prev_x']
            prev_v = self.cache_dic['cache']['prev_v']
            k = self.cache_dic['cache']['k']

            if prev_x is not None and prev_prev_x is not None:
                output_change = (velocity_feats - prev_v).abs().mean()
                prev_input_change = (prev_x - prev_prev_x).abs().mean() + 1e-8
                current_k = output_change / prev_input_change
                
                if k is None:
                    self.cache_dic['cache']['k'] = current_k
                else:
                    self.cache_dic['cache']['k'] = 0.7 * k + 0.3 * current_k 
     
            if prev_x is not None:
                self.cache_dic['cache']['prev_prev_x'] = prev_x
            self.cache_dic['cache']['prev_x'] = x_t.feats.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity_feats.detach().clone()
            self.cache_dic['cache']['easy'] = velocity_feats - x_t.feats
        
        else:
            # Reuse cache.
            velocity_feats = x_t.feats + self.cache_dic['cache']['easy']
            self.cache_dic['cache']['prev_x'] = x_t.feats.detach().clone()
            self.cache_dic['cache']['prev_v'] = velocity_feats.detach().clone()

        velocity = sp.SparseTensor(
            feats=velocity_feats,
            coords=self.coords_raw,
        )
        pred_v = velocity
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v

        output = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
        
        self.current['step'] += 1
        self.LEADER.increase_step()
        return output
    

    # Model forward pass.
    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))
        
        output = model(x_t, t, cond, **kwargs)
        return output


    # Single model forward pass plus data conversion.
    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

class FlowEulerGuidanceIntervalSampler_faster(GuidanceIntervalSamplerMixin, FlowEulerSampler_faster):
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        
        # Apply CFG.
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
