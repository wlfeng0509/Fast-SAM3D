from typing import Dict 
import torch
import math


def taylor_init(num_steps):
    """
    Initialize TaylorSeer state for the slat generator.
    """
    taylor_dic = {}
    taylor_state = {}
    taylor_state[-1] = {}

    taylor_state[-1]['double_stream'] = {}
    taylor_state[-1]['single_stream'] = {}
    taylor_dic['taylor_counter'] = 0

    taylor_state[-1]['final'] = {}
    taylor_state[-1]['final']['final'] = {}
    taylor_state[-1]['final']['final']['final'] = {}

    taylor_dic['cache'] = taylor_state

    mode = 'fast'
    if mode == 'fast':
        taylor_dic['taylor_interval'] = 6
        taylor_dic['max_order'] = 1
        taylor_dic['first_enhance'] = 2
    elif mode == 'mid':
        taylor_dic['taylor_interval'] = 4
        taylor_dic['max_order'] = 2
        taylor_dic['first_enhance'] = 3
    elif mode == 'detailed':
        taylor_dic['taylor_interval'] = 3
        taylor_dic['max_order'] = 2
        taylor_dic['first_enhance'] = 3

    taylor_dic['taylor_enabled'] = True

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = num_steps

    return taylor_dic, current


def taylor_cal_type(taylor_dic, current):
    """
    Determine whether this slat TaylorSeer step runs the model or predicts by Taylor expansion.
    """
    first_steps = current['step'] < taylor_dic['first_enhance']

    if first_steps or (taylor_dic['taylor_counter'] == taylor_dic['taylor_interval'] - 1):
        current['type'] = 'full'
        taylor_dic['taylor_counter'] = 0
        current['activated_steps'].append(current['step'])
    elif taylor_dic['taylor_enabled']:
        taylor_dic['taylor_counter'] += 1
        current['type'] = 'taylor'
    else:
        raise ValueError("Unsupported TaylorSeer calculation type")


def derivative_approximation(cache_dic: Dict, current: Dict, feature):
    """
    Compute derivative approximation.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    #difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    # current step
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['stream']][current['layer']][current['module']].get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    # print(updated_taylor_factors)
    cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors





def taylor_formula(cache_dic: Dict, current: Dict): 
    """
    Compute Taylor expansion error.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current['step'] - current['activated_steps'][-1]
    cache = cache_dic['cache'][-1][current['stream']][current['layer']][current['module']]
    output = 0

    for i in range(len(cache)):
        term =  (1 / math.factorial(i)) * cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i] * (x ** i)
        output += term
    
    # print(cache)
    return output






def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
    
    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current['step'] == 0) and (cache_dic['taylor_enabled']):
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}
