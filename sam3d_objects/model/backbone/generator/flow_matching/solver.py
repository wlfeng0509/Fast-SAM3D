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
        # 速度
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


# ⭐ easy 版本的求解器 - 专为 slat 设计
class Euler_easy_slat(ODESolver):
    def __init__(self, thresh=0.10, ret_steps=3, full_steps=25):
        """
        Args:
            thresh (float): 累积误差阈值。越大越快，但画质越低。建议 0.05 - 0.15。
            ret_steps (int): 预热步数，开始的几步强制计算，不使用 Cache。
            full_steps (int): 总推理步数，用于判断是否处于结尾阶段（结尾通常不跳过）。
        """
        super().__init__()
        self.thresh = thresh   
        self.ret_steps = ret_steps 
        self.full_steps = full_steps 
        
        # 运行时状态变量
        self.accumulated_error = 0.0  # 当前累积的预测误差，超过阈值就会强制重算
        self.k = None  # 敏感度系数 (K值)，近似 Lipschitz 常数，衡量输出对输入的敏感度
        self.prev_x = None          # 上一步的输入 x_{t-1}
        self.prev_v = None          # 上一步的输出 velocity_{t-1}
        self.prev_prev_x = None     # 上上步的输入 x_{t-2} (用于计算历史变化率)
        self.cache = None           # 核心缓存：存储变换向量 (velocity - x)
        
        # 统计数据
        self.skipped_steps = 0      # 记录跳过了多少步
        self.total_steps_run = 0    # 记录总共跑了多少步
        print(f"Real compute step indices: {self.calc_steps_list}")
        self.calc_steps_list = []

    # 重置函数
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
        
        # 1. 强制计算区间 (Warm-up 和 结尾)
        should_calc = True
        cutoff_steps = self.full_steps - 1 # 最后两步通常需要精细调整
        
        if current_step_idx < self.ret_steps or current_step_idx >= cutoff_steps:
            should_calc = True
            self.accumulated_error = 0 # 重置误差
        else:
            # 2. 尝试使用 Cache 的决策逻辑
            if self.prev_x is not None and self.prev_v is not None:
                # 计算输入的变化量 (当前 x - 上一步 x)
                input_change = (x_t - self.prev_x).abs().mean()
                
                # 如果有 K 值，预测误差
                if self.k is not None:
                    # 归一化因子 (用上一步输出的模长，防止数值尺度问题)
                    output_norm = self.prev_v.abs().mean() + 1e-6
                    # 预测误差 = 敏感度 K * (输入变化 / 输出模长)
                    pred_change = self.k * (input_change / output_norm)
                    self.accumulated_error += pred_change
                    
                    # 判定
                    if self.accumulated_error < self.thresh:
                        should_calc = False
                    else:
                        should_calc = True
                        self.accumulated_error = 0 # 误差清零，准备重新计算
                else:
                    should_calc = True # 没有 K 值时必须计算
                    
            else:
                should_calc = True

        # 3. 执行计算 或 复用 Cache
        if should_calc:
            # --- 真实运行模型 ---
            velocity = dynamics_fn(x_t, t, *args, **kwargs)
            self.calc_steps_list.append(current_step_idx)
            
            # --- 更新 easy 状态 (计算 K 和缓存向量) ---
            if self.prev_v is not None and self.prev_x is not None:
                # 计算输出变化量
                output_change = (velocity - self.prev_v).abs().mean()
                
                # 计算上一步的输入变化量 (用于计算 K)
                if self.prev_prev_x is not None:
                    prev_input_change = (self.prev_x - self.prev_prev_x).abs().mean() + 1e-8
                    # 更新 K 值: 输出变化 / 输入变化
                    # 对应 Wan2.1: self.k = output_change / input_change
                    current_k = output_change / prev_input_change
                    
                    # 可以选择平滑更新 K (Exponential Moving Average)
                    if self.k is None:
                        self.k = current_k
                    else:
                        self.k = 0.5 * self.k + 0.5 * current_k 

            # 更新历史指针
            self.prev_prev_x = self.prev_x
            self.prev_x = x_t.detach().clone() # Detach 避免显存泄露
            self.prev_v = velocity.detach().clone()
            
            # 更新 Cache: Wan2.1 核心公式 cache = output - input
            # 这里的 assumption 是 v(x) \approx x + C
            self.cache = velocity - x_t
            
        else:
            # --- 跳过计算 (easy mode) ---
            self.skipped_steps += 1
            # 核心复用公式: current_output = current_input + cache
            # 即: v_t = x_t + (v_{t-1} - x_{t-1})

            # ⭐ 计算误差
            velocity = x_t + self.cache

            # 注意：跳过步不更新 K 值，也不更新 prev_prev_x，因为没有真实观测值
            # 但我们需要更新 prev_x 和 prev_v 以便下一步计算 input_change
            # self.prev_prev_x = self.prev_x
            self.prev_x = x_t.detach().clone()
            self.prev_v = velocity.detach().clone()

        # 4. 欧拉推进 (x_{t+1} = x_t + v * dt)
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        
        return x_tp1, velocity

    # 必须重写 solve_iter 来初始化状态
    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        self.reset_state()
        print(f" Total steps   : {total_steps_run}")
        
        x_t = x_init
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            # 调用上面的 step
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            yield x_t, t0, v
            
        real_runs = self.total_steps_run - self.skipped_steps
        print(f"Easy completed | total steps: {self.total_steps_run} | skipped steps: {self.skipped_steps} | real compute steps: {real_runs}")
        print(f"Real compute step indices: {self.calc_steps_list}")
        print(f"Speedup: {self.total_steps_run / real_runs:.2f}x")

# ⭐ easy 版本的求解器 - 专为 ss 设计
# print("Euler_easy_ss", ret_steps, full_steps, thresh)

    def __init__(self, thresh=1.0, ret_steps=6, full_steps=25):
        """
        Args:
            thresh (float): 累积误差阈值。
            ret_steps (int): 预热步数。
            full_steps (int): 总推理步数。
        """
        super().__init__()
        self.thresh = thresh    
        self.ret_steps = ret_steps  
        self.full_steps = full_steps 
        # print("Euler_easy_ss", ret_steps, full_steps, thresh)
        
        self.accumulated_error = 1.5   
        self.k = None   # 全局标量 k (用于决定跳过)
        self.k_map = None # [新增] 空间 k_map (用于分析 Token 难度)
        
        # 状态变量现在存储的是字典
        self.prev_x = None          
        self.prev_v = None          
        self.prev_prev_x = None  

        self.easy_cache = {}
        self.taylor_cache = None
        self.raw_cache = {}        
            
        self.skipped_step_indices = []

        # taylor相关的
        self.cache_dir = {}
        self.current = {
            'step': 0,
            'activated_steps': []
        }

        
    # 重置缓存
    def reset_state(self):
        self.accumulated_error = 0.0
        self.k = None
        self.k_map = None # [新增] 重置 k_map

        self.prev_x = None
        self.prev_v = None
        self.prev_prev_x = None

        self.easy_cache = {} 
        self.taylor_cache = None
        self.raw_cache = {} # 这个直接复用

        self.skipped_step_indices = []


    # 计算两个字典差值的绝对值均值 (返回标量, 用于 input_change, output_change)
    def _compute_dict_diff_mean(self, d1, d2):
        """计算两个字典差值的绝对值均值 (返回标量, 用于 input_change, output_change)"""
        diffs = []
        # k = "shape"
        for k in d1.keys():
            diff = (d1[k] - d2[k]).abs().mean()
            diffs.append(diff)
        return torch.stack(diffs).mean()

    # 计算值的模长均值 (返回标量, 用于 output_norm)
    def _compute_dict_norm_mean(self, d):
        """计算字典的模长均值 (返回标量, 用于 output_norm)"""
        norms = []
        for v in d.values():
            norms.append(v.abs().mean())
        return torch.stack(norms).mean()

    # 平均计算 字典内个模态的差值
    def _compute_dict_diff_map(self, d1, d2):
        """
        [新增] 计算两个字典差值的空间分布 (返回张量 map)
        假设输入形状为 (B, C, H, W) 或 (B, C, N)，在 Dim=1 (Channel) 上求平均，保留空间维度
        """
        diff_maps = []
        for k in d1.keys():
            # (d1[k] - d2[k]).abs() 形状为 (B, C, H, W)
            # .mean(dim=1, keepdim=True) 形状为 (B, 1, H, W)，即压缩 Channel 维度
            diff = (d1[k] - d2[k]).abs().mean(dim=1, keepdim=True)
            diff_maps.append(diff)
        
        # 如果有多个模态，我们将它们的 diff map 平均起来
        # 注意：这里假设所有模态的空间分辨率一致。如果不一致，建议只取主模态(如'shape')
        return torch.stack(diff_maps).mean(dim=0)

    # 复制字典
    def _dict_clone(self, d):
        """字典 Clone + Detach"""
        return {k: v.detach().clone() for k, v in d.items()}

    # 计算两个字典的差值，仍然返回字典
    def compute_dict_diff(self,d1, d2):
        # 使用字典推导式高效计算
        return {k: d1[k] - d2[k] for k in d1.keys()}\
        
    # 计算两个字典的和，仍然返回字典
    def compute_dict_add(self,d1, d2):
        # 使用字典推导式高效计算
        return {k: d1[k] + d2[k] for k in d1.keys()}
    
    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs): 
        current_step_idx = self.current.get('step', 0)
        # 1. 决策逻辑 (保持你原有的标量 k 逻辑，用于决定是否跳过整步)
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
        # 2. 执行计算 或 复用 Cache
        if should_calc:
            velocity = dynamics_fn(x_t, t, *args, **kwargs)

            # --- 更新标量 k (全局决策用) ---
            if self.prev_v is not None and self.prev_prev_x is not None:
                # output_change = velocity['shape'] - self.prev_v['shape']
                output_change = self._compute_dict_diff_mean(velocity, self.prev_v) + 1e-8
                # prev_input_change = self.prev_x['shape'], self.prev_prev_x['shape'] + 1e-8
                prev_input_change = self._compute_dict_diff_mean(self.prev_x, self.prev_prev_x) + 1e-8

                current_k = output_change / prev_input_change
                # print("current_k",current_k )

                self.k = current_k if self.k is None else 0.7 * self.k + 0.3 * current_k

            # ====================================================
            # ⭐ 计算空间 k_map (速度+加速度得分)
            if 'shape' in velocity:
                curr_v = velocity['shape'] # 形状 (B, N, C)
                
                # A. 计算速度得分 (L2 Norm)，dim=2 通常是 Channel 维度
                l2_scores = torch.norm(curr_v, p=2, dim=2, keepdim=True)

                # B. 计算加速度得分 (与前一步速度的差异)
                if self.prev_v is not None and 'shape' in self.prev_v:
                    prev_v = self.prev_v['shape']
                    accel_scores = torch.norm(curr_v - prev_v, p=2, dim=2, keepdim=True)
                else:
                    accel_scores = torch.zeros_like(l2_scores)

                # C. 空间归一化 (Min-Max Normalization)
                def normalize_map(m):
                    m_min = m.min()
                    m_max = m.max()
                    return (m - m_min) / (m_max - m_min + 1e-6)

                l2_norm_map = normalize_map(l2_scores)
                accel_norm_map = normalize_map(accel_scores)

                # D. 融合得分
                accel_weight = getattr(self, 'ACCELERATION_WEIGHT', 0.7)
                # print("update accel_weight")
                current_k_map = (accel_weight * accel_norm_map) + ((1.0 - accel_weight) * l2_norm_map)

                # E. EMA 更新 k_map
                if self.k_map is None:
                    self.k_map = current_k_map
                else:
                    # 只有在特定步数范围内更新，或者全过程更新
                    self.k_map = 0.9 * self.k_map + 0.1 * current_k_map
            # ====================================================

            # 更新历史指针
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
            # --- 跳过计算，复用 Cache ---
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
        # 3. 欧拉推进 
        x_tp1 = linear_approximation_step(x_t, dt, velocity)
        return x_tp1, velocity


    # 必须重写 solve_iter 来初始化状态并输出统计
    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        self.reset_state()
        print(f" Total steps   : {total_steps_run}")
        
        x_t = x_init
        
        for t0, t1 in zip(times[:-1], times[1:]):
            dt = t1 - t0
            x_t, v = self.step(dynamics_fn, x_t, t0, dt, *args, **kwargs)
            yield x_t, t0, v 
            
        # --- 统计输出部分 ---
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
          
