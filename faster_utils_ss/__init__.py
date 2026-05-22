import math
from typing import Dict, Union

import torch


def faster_init(num_steps, faster_interval=3, max_order=1, first_enhance=2, end_enhance=24):
    """
    Initialize Faster state for the sparse-structure generator.
    """
    faster_dic = {}
    faster_state = {}
    faster_state[-1] = {}

    faster_dic['faster_counter'] = 0
    faster_state[-1]['final'] = {}
    faster_state[-1]['final']['final'] = {}
    faster_state[-1]['final']['final']['final'] = {}

    faster_dic['cache'] = faster_state
    faster_dic['faster_interval'] = faster_interval
    faster_dic['max_order'] = max_order
    faster_dic['first_enhance'] = first_enhance
    faster_dic['end_enhance'] = end_enhance
    faster_dic['faster_enabled'] = True

    current = {}
    current['activated_steps'] = []
    current['step'] = 0
    current['num_steps'] = num_steps

    return faster_dic, current


def faster_cal_type(faster_dic, current):
    """
    Determine whether this sparse-structure Faster step runs the model or predicts by Faster expansion.
    """
    first_steps = current['step'] < faster_dic['first_enhance']
    end_steps = current['step'] >= faster_dic['end_enhance']

    if first_steps or (faster_dic['faster_counter'] == faster_dic['faster_interval'] - 1) or end_steps:
        current['type'] = 'full'
        faster_dic['faster_counter'] = 0
        current['activated_steps'].append(current['step'])
    elif faster_dic['faster_enabled']:
        faster_dic['faster_counter'] += 1
        current['type'] = 'faster'
    else:
        raise ValueError("Unsupported Faster calculation type")


def derivative_approximation(faster_dic: Dict, current: Dict, feature: Union[Dict[str, torch.Tensor], torch.Tensor]):
    if len(current['activated_steps']) < 2:
        difference_distance = 1.0
    else:
        difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

    if isinstance(feature, torch.Tensor):
        feature = {'default': feature}

    prev_module_cache = faster_dic['cache'][-1][current['layer']][current['module']]
    updated_module_factors = {}

    for key, tensor_val in feature.items():
        current_key_factors = {0: tensor_val}
        prev_key_cache = prev_module_cache.get(key, None) if prev_module_cache else None

        for i in range(faster_dic['max_order']):
            has_prev = (prev_key_cache is not None) and (i in prev_key_cache)
            is_within = current['step'] < (current['num_steps'] - faster_dic['first_enhance'] + 1)
            if has_prev and is_within:
                prev_val = prev_key_cache[i]
                current_val = current_key_factors[i]
                current_key_factors[i + 1] = (current_val - prev_val) / difference_distance
            else:
                break

        updated_module_factors[key] = current_key_factors

    if current['layer'] not in faster_dic['cache'][-1]:
        faster_dic['cache'][-1][current['layer']] = {}

    faster_dic['cache'][-1][current['layer']][current['module']] = updated_module_factors


def faster_formula(faster_dic: Dict, current: Dict, prev_v: Dict, beta=0.5) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    x = current['step'] - current['activated_steps'][-1]
    module_cache = faster_dic['cache'][-1][current['layer']][current['module']]

    def compute_single_expansion(factors_dict, x_dist):
        result = 0
        for i in range(len(factors_dict)):
            term = (1 / math.factorial(i)) * factors_dict[i] * (x_dist ** i)
            result += term
        return result

    def compute_single_expansion_ema(factors_dict, x_dist, prev_value, beta=0.5):
        result = 0
        for i in range(len(factors_dict)):
            term = (1 / math.factorial(i)) * factors_dict[i] * (x_dist ** i)
            result += term

        return beta * prev_value + (1.0 - beta) * result

    first_val = next(iter(module_cache.values()))

    if isinstance(first_val, dict):
        output_dict = {}
        for key, factors in module_cache.items():
            full_keys = ['shape']
            ema_keys = ['6drotation_normalized', 'scale', 'translation', 'translation_scale']

            if key in full_keys:
                output_dict[key] = compute_single_expansion(factors, x)
            if key in ema_keys:
                output_dict[key] = compute_single_expansion_ema(factors, x, prev_v[key], beta)

            output_dict[key] = compute_single_expansion(factors, x)

        return output_dict

    return compute_single_expansion(module_cache, x)


def faster_step_init(faster_dic: Dict, current: Dict):
    """
    Keep this hook for symmetry with Taylor/Faster call sites.
    """
    if current['step'] == (current['num_steps'] - 1):
        pass
