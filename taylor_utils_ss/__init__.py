
import torch
import math
from typing import Dict, Union, List


def taylor_init(num_steps, taylor_interval=3, max_order=1, first_enhance=2, end_enhance=24):
    """
    Initialize TaylorSeer state for the sparse-structure generator.
    """
    taylor_dic = {}
    taylor_state = {}
    taylor_state[-1] = {}

    taylor_dic['taylor_counter'] = 0
    taylor_state[-1]['final'] = {}
    taylor_state[-1]['final']['final'] = {}
    taylor_state[-1]['final']['final']['final'] = {}

    taylor_dic['cache'] = taylor_state
    taylor_dic['taylor_interval'] = taylor_interval
    taylor_dic['max_order'] = max_order
    taylor_dic['first_enhance'] = first_enhance
    taylor_dic['end_enhance'] = end_enhance
    taylor_dic['taylor_enabled'] = True

    current = {}
    current['activated_steps'] = []
    current['step'] = 0
    current['num_steps'] = num_steps

    return taylor_dic, current


def taylor_cal_type(taylor_dic, current):
    """
    Determine whether this sparse-structure TaylorSeer step runs the model or predicts by Taylor expansion.
    """
    first_steps = current['step'] < taylor_dic['first_enhance']
    end_steps = current['step'] >= taylor_dic['end_enhance']

    if first_steps or (taylor_dic['taylor_counter'] == taylor_dic['taylor_interval'] - 1) or end_steps:
        current['type'] = 'full'
        taylor_dic['taylor_counter'] = 0
        current['activated_steps'].append(current['step'])
    elif taylor_dic['taylor_enabled']:
        taylor_dic['taylor_counter'] += 1
        current['type'] = 'taylor'
    else:
        raise ValueError("Unsupported TaylorSeer calculation type")


# 计算近似导数
def derivative_approximation(cache_dic: Dict, current: Dict, feature: Union[Dict[str, torch.Tensor], torch.Tensor]):
    """Approximate Taylor derivatives from the latest cached feature."""
    if len(current['activated_steps']) < 2:
        difference_distance = 1.0
    else:
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    if isinstance(feature, torch.Tensor):
        feature = {'default': feature}

    prev_module_cache = cache_dic['cache'][-1][current['layer']][current['module']]
    updated_module_factors = {}

    for key, tensor_val in feature.items():
        current_key_factors = {0: tensor_val}
        prev_key_cache = prev_module_cache.get(key, None) if prev_module_cache else None

        if current['step'] > 0 and prev_key_cache is None:
            print(f"Warning: Step {current['step']} (Module: {current['module']}) | Key '{key}' is missing historical cache; high-order derivatives cannot be computed.")

        for i in range(cache_dic['max_order']):
            has_prev = (prev_key_cache is not None) and (i in prev_key_cache)
            is_within = current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)

            if has_prev and is_within:
                prev_val = prev_key_cache[i]
                current_val = current_key_factors[i]
                current_key_factors[i + 1] = (current_val - prev_val) / difference_distance
                print(f"Computed order {i+1} derivative for key '{key}'")
            else:
                print(f"Stop high-order computation: Step={current['step']}, Order={i+1}, Reason: PrevExist={has_prev}, IsWithin={is_within}")
                break

        updated_module_factors[key] = current_key_factors

    if current['layer'] not in cache_dic['cache'][-1]:
        cache_dic['cache'][-1][current['layer']] = {}

    cache_dic['cache'][-1][current['layer']][current['module']] = updated_module_factors
    print(f"Step {current['step']} cache write complete. Keys: {list(updated_module_factors.keys())}")
def taylor_formula(cache_dic: Dict, current: Dict, prev_v: Dict) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Predict the intermediate step result with a Taylor expansion."""
    x = current['step'] - current['activated_steps'][-1]
    print(f"Taylor prediction debug: Step={current['step']}, LastActive={current['activated_steps'][-1]}, Distance x={x}")

    module_cache = cache_dic['cache'][-1][current['layer']][current['module']]

    def compute_single_expansion(factors_dict, x_dist):
        result = 0
        for i in range(len(factors_dict)):
            term = (1 / math.factorial(i)) * factors_dict[i] * (x_dist ** i)
            result += term
        return result

    def compute_single_expansion_ema(factors_dict, x_dist, prev_value, factor=0.5):
        result = 0
        for i in range(len(factors_dict)):
            term = (1 / math.factorial(i)) * factors_dict[i] * (x_dist ** i)
            result += term
        return factor * prev_value + (1.0 - factor) * result

    first_val = next(iter(module_cache.values()))

    if isinstance(first_val, dict):
        output_dict = {}
        for key, factors in module_cache.items():
            full_keys = ['shape']
            ema_keys = ['6drotation_normalized', 'scale', 'translation', 'translation_scale']

            if key in full_keys:
                output_dict[key] = compute_single_expansion(factors, x)
                print("full")
            elif key in ema_keys:
                prev_value = prev_v[key]
                output_dict[key] = compute_single_expansion_ema(factors, x, prev_value, 0.9)
                print("ema")
            else:
                output_dict[key] = compute_single_expansion(factors, x)

        return output_dict

    return compute_single_expansion(module_cache, x)
def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and expand storage for different-order derivatives.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    # 存放初始状态的位置，-1是最后一个
    if current['step'] == (current['num_steps'] - 1):
        # print(f"clear cache,{current['step']},{current['num_steps']}")
        # cache_dic['cache'][-1][current['layer']][current['module']] = {}
        pass
