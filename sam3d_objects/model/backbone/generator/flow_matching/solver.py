# Copyright (c) Meta Platforms, Inc. and affiliates.
import optree
import torch
from functools import partial
import time
from sam3d_objects.data.utils import tree_tensor_map
import numpy as np
import torch
import torch.nn.functional as F


def linear_approximation_step(x_t, dt, velocity):
    # x_tp1 = x_t + velocity * dt
    x_tp1 = tree_tensor_map(lambda x, v: x + v * dt, x_t, velocity)
    return x_tp1


def gradient(output, x, create_graph: bool = False):
    tensors, pyspec = optree.tree_flatten(
        x, is_leaf=lambda x: isinstance(x, torch.Tensor)
    )
    grad_outputs = [torch.ones_like(output).detach() for _ in tensors]
    grads = torch.autograd.grad(
        output,
        tensors,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
    )
    return optree.tree_unflatten(pyspec, grads)


def simple_dynamics(x, t):
    return -x + torch.sin(t)


class ODESolver:
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        raise NotImplementedError

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            yield x_t, t0 ,v

    def solve(self, dynamics_fn, x_init, times, *args, **kwargs):
        for x_t, _, _, in self.solve_iter(dynamics_fn, x_init, times, *args, **kwargs):
            pass
        return x_t
    

# https://en.wikipedia.org/wiki/Euler_method
class Euler(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        # velocity
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        return x_tp1,velocity


# https://arxiv.org/abs/2505.05470
class SDE(ODESolver):
    def __init__(self, **kwargs):
        super().__init__()
        self.sde_strength = kwargs.get("sde_strength", 0.1)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        sigma = 1 - t
        var_t = sigma / (1 - torch.tensor(sigma).clamp(min=dt))
        std_dev_t = (
            torch.sqrt(variance) * self.sde_strength
        )  # self.sde_strength = alpha

        def compute_mean(x, v):
            drift_term = x * (std_dev_t**2 / (2 * sigma) * dt)
            velocity_term = v * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            return x + drift_term + velocity_term

        prev_sample_mean = tree_tensor_map(compute_mean, x_t, velocity)

        # Generate noise and compute final sample using tree_tensor_map
        def add_noise(mean_val):
            variance_noise = torch.randn_like(mean_val)
            return mean_val + std_dev_t * torch.sqrt(torch.tensor(dt)) * variance_noise

        prev_sample = tree_tensor_map(add_noise, prev_sample_mean)

        return prev_sample


# https://en.wikipedia.org/wiki/Midpoint_method
class Midpoint(ODESolver):
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        half_dt = 0.5 * dt

        x_mid = Euler.step(self, dynamics_fn, x_t, t, half_dt, *args, **kwargs)

        velocity_mid = dynamics_fn(x_mid, t + half_dt, *args, **kwargs)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_mid)
        return x_tp1,velocity_mid


# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
class RungeKutta4(ODESolver):

    def k1(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        return dynamics_fn(x_t, t, *args, **kwargs)

    def k2(self, dynamics_fn, x_t, t, dt, k1, *args, **kwargs):
        x_k1 = linear_approximation_step(x_t, dt * 0.5, k1)
        return dynamics_fn(x_k1, t + dt * 0.5, *args, **kwargs)

    def k3(self, dynamics_fn, x_t, t, dt, k2, *args, **kwargs):
        x_k2 = linear_approximation_step(x_t, dt * 0.5, k2)
        return dynamics_fn(x_k2, t + dt * 0.5, *args, **kwargs)

    def k4(self, dynamics_fn, x_t, t, dt, k3, *args, **kwargs):
        x_k3 = linear_approximation_step(x_t, dt, k3)
        return dynamics_fn(x_k3, t + dt, *args, **kwargs)

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        k1 = self.k1(dynamics_fn, x_t, t, dt, *args, **kwargs)
        k2 = self.k2(dynamics_fn, x_t, t, dt, k1, *args, **kwargs)
        k3 = self.k3(dynamics_fn, x_t, t, dt, k2, *args, **kwargs)
        k4 = self.k4(dynamics_fn, x_t, t, dt, k3, *args, **kwargs)

        def compute_velocity(k1, k2, k3, k4):
            return (k1 + 2 * k2 + 2 * k3 + k4) / 6

        velocity_k = tree_tensor_map(compute_velocity, k1, k2, k3, k4)
        x_tp1 = linear_approximation_step(x_t, dt, velocity_k)
        return x_tp1,velocity_k


from faster_utils_slat import faster_cal_type, faster_init
class Euler_faster_slat(ODESolver):
    def __init__(self, thresh=0.0, dir_weight=0.5, ret_steps=1, full_steps=25,carving_ratio = 0.0):
       
        super().__init__()
        self.thresh = thresh 
        self.dir_weight =  dir_weight
        self.ret_steps = ret_steps
        self.full_steps = full_steps
        self.carving_ratio = carving_ratio
       
        self.faster_dic = None
        self.current = None

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
     
        should_calc = True
        should_calc = faster_cal_type(self.faster_dic, self.current, x_t)
        if should_calc:
            if self.current['is_token_active'] and self.current['use_token']:
                coords_scores = self.stability_tracker.coords_scores
                self.current['cached_indices'],  self.current['fast_update_indices'] = self.stability_tracker.update_and_select_combined(self.faster_dic['cache']['prev_v'], self.current['num_to_skip'],t=0, coords_scores = coords_scores,spatial_weight=0.3)

                step_args_list= list(args)
                x_input = x_t[:, self.current['fast_update_indices'], :] if self.current['is_token_active'] else x_t

                if len(step_args_list) > 1:
                    full_coords = self.full_coords_backup
                    idx_np = self.current['fast_update_indices'].detach().cpu().numpy()
                    cropped_coords = full_coords[idx_np].astype(np.int32)
                    step_args_list[1] = cropped_coords
                            
                step_args = tuple(step_args_list)
                velocity = dynamics_fn(x_input, t, *step_args, **kwargs)


            else:

                velocity = dynamics_fn(x_t, t, *args, **kwargs)
            
            if self.current['is_token_active'] and self.current['use_token']:
                final_v_tokens = self.faster_dic['cache']['prev_v'].clone()
                final_v_tokens[:, self.current['fast_update_indices'], :] = velocity.to(final_v_tokens.dtype)
                velocity = final_v_tokens
           
            
            prev_x = self.faster_dic['cache']['prev_x']
            prev_prev_x = self.faster_dic['cache']['prev_prev_x']
            prev_v = self.faster_dic['cache']['prev_v']
            k = self.faster_dic['cache']['k']

            if prev_x is not None and prev_prev_x is not None:
                output_change = (velocity - prev_v).abs().mean()
                prev_input_change = (prev_x - prev_prev_x).abs().mean() + 1e-8
                current_k = output_change / prev_input_change
                
                if k is None:
                    self.faster_dic['cache']['k'] = current_k
                else:
                    self.faster_dic['cache']['k'] = 0.7 * k + 0.3 * current_k

     
            if prev_x is not None:
                self.faster_dic['cache']['prev_prev_x'] = prev_x
            self.faster_dic['cache']['prev_x'] = x_t.detach().clone()
            self.faster_dic['cache']['prev_v'] = velocity.detach().clone()
            self.faster_dic['cache']['easy'] = velocity - x_t
        else:

            velocity = x_t + self.faster_dic['cache']['easy']
    
            self.faster_dic['cache']['prev_x'] = x_t.detach().clone()
            self.faster_dic['cache']['prev_v'] = velocity.detach().clone()

        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        self.current['step'] += 1
        
        return x_tp1, velocity

    def solve_iter(self, dynamics_fn, x_init, times, LEADER, stability_tracker, *args, **kwargs):

        self.faster_dic, self.current = faster_init(self.full_steps)
        self.faster_dic['thresh'] = self.thresh
        self.faster_dic['dir_weight'] = self.dir_weight
        self.faster_dic['first_enhance'] = self.ret_steps

        self.LEADER = LEADER    
        self.stability_tracker = stability_tracker

        current_args_list = list(args) 
        if len(current_args_list) > 1:
            self.full_coords_backup = current_args_list[1] 
        
        B, N, C = x_init.shape
        LEADER.total_tokens = N
        LEADER.schedule_is_set = True
        self.last_coords =  current_args_list[1]

        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            
            cache = self.faster_dic['cache']
            self.current['is_token_active'] = False
            current_step = LEADER.current_step

            if self.current['use_token'] and cache['prev_v'] is not None and current_step >= LEADER.full_sampling_steps:
                self.current['num_to_skip'] = int(self.carving_ratio * N)
                if self.current['num_to_skip'] > 0 and self.current['num_to_skip'] < N:
                    self.current['is_token_active'] = True

            dt = t1 - t0
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)

            LEADER.increase_step()
            yield x_t, t0, v


# Easy-version solver designed for SLaT
class Euler_easy_slat(ODESolver):
    def __init__(self, thresh=0.10, ret_steps=3, full_steps=25):
        """
        Args:
            thresh (float): Accumulated error threshold. Larger is faster but lowers quality. Recommended range: 0.05 - 0.15.
            ret_steps (int): Warm-up steps. The first few steps are forced to compute without using Cache.
            full_steps (int): Total inference steps, used to determine whether execution is in the final stage, which is usually not skipped.
        """
        super().__init__()
        self.thresh = thresh   
        self.ret_steps = ret_steps 
        self.full_steps = full_steps 
        
        # Runtime state variables
        self.accumulated_error = 0.0  # Current accumulated prediction error; recompute when it exceeds the threshold
        self.k = None  # Sensitivity coefficient (K value), an approximate Lipschitz constant measuring output sensitivity to input
        self.prev_x = None          # previous-step input x_{t-1}
        self.prev_v = None          # previous-step output velocity_{t-1}
        self.prev_prev_x = None     # input from two steps ago x_{t-2} (used to compute historical rate of change)
        self.cache = None           # Core cache: stores the transform vector (velocity - x)
        
        # Statistics
        self.skipped_steps = 0      # record how many steps were skipped
        self.total_steps_run = 0    # record how many steps ran in total
        print(f"Real compute step indices: {self.calc_steps_list}")
        self.calc_steps_list = []

    # Reset function
    def reset_state(self):
        self.accumulated_error = 0.0
        self.k = None
        self.prev_x = None
        self.prev_v = None
        self.prev_prev_x = None
        self.cache = None
        self.skipped_steps = 0
        self.total_steps_run = 0
        self.calc_steps_list = []

    # 
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        self.total_steps_run += 1
        current_step_idx = self.total_steps_run - 1
        
        # 1. Forced-computation interval (warm-up and final steps)
        should_calc = True
        cutoff_steps = self.full_steps - 1 # the last two steps usually need fine adjustment
        
        if current_step_idx < self.ret_steps or current_step_idx >= cutoff_steps:
            should_calc = True
            self.accumulated_error = 0 # reset error
        else:
            # 2. Decision logic for trying to use Cache
            if self.prev_x is not None and self.prev_v is not None:
                # Compute input change (current x - previous x)
                input_change = (x_t - self.prev_x).abs().mean()
                
                # Predict error if K is available
                if self.k is not None:
                    # Normalization factor (uses the previous output norm to avoid scale issues)
                    output_norm = self.prev_v.abs().mean() + 1e-6
                    # Predicted error = sensitivity K * (input change / output norm)
                    pred_change = self.k * (input_change / output_norm)
                    self.accumulated_error += pred_change
                    
                    # Decision
                    if self.accumulated_error < self.thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_error = 0 # clear error before recomputation
                else:
                    should_calc = True # must compute when K is unavailable
                    
            else:
                should_calc = True

        # 3. Execute computation or reuse Cache
        if should_calc:
            # --- Actually run the model ---
            velocity = dynamics_fn(x_t, t, *args, **kwargs)
            self.calc_steps_list.append(current_step_idx)
            
            # --- Update Easy state (compute K and cache vector) ---
            if self.prev_v is not None and self.prev_x is not None:
                # Compute output change
                output_change = (velocity - self.prev_v).abs().mean()
                
                # Compute previous-step input change (used to compute K)
                if self.prev_prev_x is not None:
                    prev_input_change = (self.prev_x - self.prev_prev_x).abs().mean() + 1e-8
                    # Update K value: output change / input change
                    # Corresponds to Wan2.1: self.k = output_change / input_change
                    current_k = output_change / prev_input_change
                    
                    # Optionally smooth K updates (Exponential Moving Average)
                    if self.k is None:
                        self.k = current_k
                    else:
                        self.k = 0.5 * self.k + 0.5 * current_k 

            # Update history pointers
            self.prev_prev_x = self.prev_x
            self.prev_x = x_t.detach().clone() # Detach to avoid GPU memory leaks
            self.prev_v = velocity.detach().clone()
            
            # Update Cache: Wan2.1 core formula cache = output - input
            # The assumption here is v(x) \approx x + C
            self.cache = velocity - x_t
            
        else:
            # --- Skip computation (easy mode) ---
            self.skipped_steps += 1
            # Core reuse formula: current_output = current_input + cache
            # That is: v_t = x_t + (v_{t-1} - x_{t-1})

            # Compute error
            velocity = x_t + self.cache

            # Note: skipped steps do not update K or prev_prev_x because there is no real observation
            # But prev_x and prev_v must be updated so the next step can compute input_change
            # self.prev_prev_x = self.prev_x
            self.prev_x = x_t.detach().clone()
            self.prev_v = velocity.detach().clone()

        # 4. Euler advance (x_{t+1} = x_t + v * dt)
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        
        return x_tp1, velocity

    # solve_iter must be overridden to initialize state
    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        self.reset_state()
        print(f" Total steps   : {total_steps_run}")
        
        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            # Call the step above
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            yield x_t, t0, v
            
        real_runs = self.total_steps_run - self.skipped_steps
        print(f"Easy completed | total steps: {self.total_steps_run} | skipped steps: {self.skipped_steps} | real compute steps: {real_runs}")
        print(f"Real compute step indices: {self.calc_steps_list}")
        print(f"Speedup: {self.total_steps_run / real_runs:.2f}x")

# Easy-version solver designed for SS
# print("Euler_easy_ss", ret_steps, full_steps, thresh)

    def __init__(self, thresh=1.0, ret_steps=6, full_steps=25):
        """
        Args:
            thresh (float): Accumulated error threshold.
            ret_steps (int): Warm-up steps.
            full_steps (int): Total inference steps.
        """
        super().__init__()
        self.thresh = thresh    
        self.ret_steps = ret_steps  
        self.full_steps = full_steps 
        # print("Euler_easy_ss", ret_steps, full_steps, thresh)
        
        self.accumulated_error = 1.5   
        self.k = None   # Global scalar k (used to decide whether to skip)
        self.k_map = None # [Added] Spatial k_map (used to analyze Token difficulty)
        
        # State variables are now stored as dictionaries
        self.prev_x = None          
        self.prev_v = None          
        self.prev_prev_x = None  

        self.easy_cache = {}
        self.taylor_cache = None
        self.raw_cache = {}        
            
        self.skipped_step_indices = []

        # Taylor-related
        self.cache_dir = {}
        self.current = {
            'step': 0,
            'activated_steps': []
        }

        
    # Reset cache
    def reset_state(self):
        self.accumulated_error = 0.0
        self.k = None
        self.k_map = None # [Added] Reset k_map

        self.prev_x = None
        self.prev_v = None
        self.prev_prev_x = None

        self.easy_cache = {} 
        self.taylor_cache = None
        self.raw_cache = {} # Reuse this directly

        self.skipped_step_indices = []


    # Compute the mean absolute difference between two dictionaries (returns a scalar for input_change and output_change)
    def _compute_dict_diff_mean(self, d1, d2):
        """Compute the mean absolute difference between two dictionaries (returns a scalar for input_change and output_change)."""
        diffs = []
        # k = "shape"
        for k in d1.keys():
            diff = (d1[k] - d2[k]).abs().mean()
            diffs.append(diff)
        return torch.stack(diffs).mean()

    # Compute the mean value norm (returns a scalar for output_norm)
    def _compute_dict_norm_mean(self, d):
        """Compute the mean norm of a dictionary (returns a scalar for output_norm)."""
        norms = []
        for v in d.values():
            norms.append(v.abs().mean())
        return torch.stack(norms).mean()

    # Average the differences across modalities in the dictionary
    def _compute_dict_diff_map(self, d1, d2):
        """
        [Added] Compute the spatial distribution of differences between two dictionaries (returns a tensor map)
        Assumes the input shape is (B, C, H, W) or (B, C, N); averages over Dim=1 (Channel) while preserving spatial dimensions
        """
        diff_maps = []
        for k in d1.keys():
            # (d1[k] - d2[k]).abs() has shape (B, C, H, W)
            # .mean(dim=1, keepdim=True) has shape (B, 1, H, W), compressing the Channel dimension
            diff = (d1[k] - d2[k]).abs().mean(dim=1, keepdim=True)
            diff_maps.append(diff)
        
        # If there are multiple modalities, average their diff maps
        # Note: this assumes all modalities have the same spatial resolution. If not, use only the main modality (for example, 'shape')
        return torch.stack(diff_maps).mean(dim=0)

    # Copy dictionary
    def _dict_clone(self, d):
        """Clone + detach dictionary."""
        return {k: v.detach().clone() for k, v in d.items()}

    # Compute the difference between two dictionaries, still returning a dictionary
    def compute_dict_diff(self,d1, d2):
        # Use dictionary comprehensions for efficient computation
        return {k: d1[k] - d2[k] for k in d1.keys()}\
        
    # Compute the sum of two dictionaries, still returning a dictionary
    def compute_dict_add(self,d1, d2):
        # Use dictionary comprehensions for efficient computation
        return {k: d1[k] + d2[k] for k in d1.keys()}
    
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs): 
        current_step_idx = self.current.get('step', 0)
        # 1. Decision logic (keeps the original scalar k logic used to decide whether to skip the whole step)
        should_calc = True
        cutoff_steps = self.full_steps - 1 

        if current_step_idx < self.ret_steps or current_step_idx >= cutoff_steps:
            should_calc = True
            self.accumulated_error = 0 
        else:
            if self.prev_x is not None and self.prev_v is not None:
                input_change = self._compute_dict_diff_mean(x_t, self.prev_x)
                if self.k is not None:
                    output_norm = self._compute_dict_norm_mean(self.prev_v) + 1e-6
                    pred_change = self.k * (input_change / output_norm)
                    # print("pred_change",pred_change,"k",self.k,"(input_change / output_norm)",input_change / output_norm)
                    self.accumulated_error += pred_change

                    if self.accumulated_error >= self.thresh:
                        # print("error is large")
                        should_calc = True
                        self.accumulated_error = 0 
                    else:
                        should_calc = False

                else:
                    should_calc = True
                    self.accumulated_error = 0
            else:
                should_calc = True
                self.accumulated_error = 0

        print("should_calc",should_calc)
        # 2. Execute computation or reuse Cache
        if should_calc:
            velocity = dynamics_fn(x_t, t, *args, **kwargs)

            # --- Update scalar k (for global decisions) ---
            if self.prev_v is not None and self.prev_prev_x is not None:
                # output_change = velocity['shape'] - self.prev_v['shape']
                output_change = self._compute_dict_diff_mean(velocity, self.prev_v) + 1e-8
                # prev_input_change = self.prev_x['shape'], self.prev_prev_x['shape'] + 1e-8
                prev_input_change = self._compute_dict_diff_mean(self.prev_x, self.prev_prev_x) + 1e-8

                current_k = output_change / prev_input_change
                # print("current_k",current_k )

                self.k = current_k if self.k is None else 0.7 * self.k + 0.3 * current_k

            # ====================================================
            # Compute spatial k_map (velocity + acceleration score)
            if 'shape' in velocity:
                curr_v = velocity['shape'] # shape (B, N, C)
                
                # A. Compute velocity score (L2 Norm); dim=2 is usually the Channel dimension
                l2_scores = torch.norm(curr_v, p=2, dim=2, keepdim=True)

                # B. Compute acceleration score (difference from previous-step velocity)
                if self.prev_v is not None and 'shape' in self.prev_v:
                    prev_v = self.prev_v['shape']
                    accel_scores = torch.norm(curr_v - prev_v, p=2, dim=2, keepdim=True)
                else:
                    accel_scores = torch.zeros_like(l2_scores)

                # C. Spatial normalization (Min-Max Normalization)
                def normalize_map(m):
                    m_min = m.min()
                    m_max = m.max()
                    return (m - m_min) / (m_max - m_min + 1e-6)

                l2_norm_map = normalize_map(l2_scores)
                accel_norm_map = normalize_map(accel_scores)

                # D. Fuse scores
                accel_weight = getattr(self, 'ACCELERATION_WEIGHT', 0.7)
                # print("update accel_weight")
                current_k_map = (accel_weight * accel_norm_map) + ((1.0 - accel_weight) * l2_norm_map)

                # E. EMA update k_map
                if self.k_map is None:
                    self.k_map = current_k_map
                else:
                    # Update only within a specific step range, or update throughout the full process
                    self.k_map = 0.9 * self.k_map + 0.1 * current_k_map
            # ====================================================

            # Update history pointers
            self.prev_prev_x = self.prev_x
            self.prev_x = self._dict_clone(x_t)
            self.prev_v = self._dict_clone(velocity)

            # self.easy_cache['shape'] = velocity['shape']-x_t['shape']
            self.easy_cache =  self.compute_dict_diff(velocity , x_t)

            # self.raw_cache['6drotation_normalized'] = velocity['6drotation_normalized']
            # self.raw_cache['scale'] = velocity['scale']
            # self.raw_cache['shape'] = velocity['shape']
            # self.raw_cache['translation'] = velocity['translation']
            # self.raw_cache['translation_scale'] = velocity['translation_scale']

            # derivative_approximation_end(self.cache_dic, self.current, velocity)
            self.current['activated_steps'].append(self.current['step'])
            # self.raw_cache = self._dict_clone(velocity)
            
        else:
            # --- Skip computation and reuse Cache ---
            velocity = {}
            velocity =  self.compute_dict_add(x_t,self.easy_cache)

            # velocity['shape'] = x_t['shape'] + self.easy_cache['shape']
            # velocity['6drotation_normalized'] = self.raw_cache['6drotation_normalized']
            # velocity['scale'] = self.raw_cache['scale']
            # velocity['translation'] = self.raw_cache['translation']
            # velocity['translation_scale'] = self.raw_cache['translation_scale']
            
            
            # velocity = self._dict_clone(raw_cache)

            # velocity = dynamics_fn(x_t, t, *args, **kwargs)

            # velocity = taylor_formula_end(self.cache_dic, self.current)

            self.skipped_step_indices.append(current_step_idx)
            self.prev_x = self._dict_clone(x_t)
            self.prev_v = self._dict_clone(velocity)
            
            
        self.current['step'] += 1
        # 3. Euler advance 
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        return x_tp1, velocity


    # solve_iter must be overridden to initialize state and output statistics
    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        self.reset_state()
        print(f" Total steps   : {total_steps_run}")
        
        x_t = x_init
        
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            yield x_t, t0, v 
            
        # --- Statistics output section ---
        total_steps_run = self.current['step']+1
        computed_steps = total_steps_run - len(self.skipped_step_indices) 
        skip_ratio = (len(self.skipped_step_indices)  / total_steps_run * 100) if total_steps_run > 0 else 0
        speedup = (total_steps_run / computed_steps) if computed_steps > 0 else float('inf')
        
        print(f"\n{'='*30}")
        print(f" Easy SS summary")
        print(f"{'='*30}")
        print(f" Total steps   : {total_steps_run}")
        print(f" Computed steps: {computed_steps}")
        print(f" Skipped steps : {self.skipped_step_indices}")
        print(f"Speedup: {self.total_steps_run / real_runs:.2f}x")
        print(f"{'='*30}\n")
          
