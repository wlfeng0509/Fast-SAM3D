import math
from typing import Dict

import torch


def taylor_init(num_steps, taylor_interval=3, max_order=1, first_enhance=2, end_enhance=24):
    """Initialize TaylorSeer state for TRELLIS sparse-structure sampling."""
    taylor_state = {-1: {"final": {"final": {"final": {}}}}}
    taylor_dic = {
        "cache": taylor_state,
        "taylor_counter": 0,
        "taylor_interval": taylor_interval,
        "max_order": max_order,
        "first_enhance": first_enhance,
        "end_enhance": end_enhance,
        "taylor_enabled": True,
    }
    current = {
        "activated_steps": [],
        "step": 0,
        "num_steps": num_steps,
    }
    return taylor_dic, current


def taylor_cal_type(taylor_dic, current):
    """Mark the current SS step as a full model step or a Taylor predicted step."""
    first_steps = current["step"] < taylor_dic["first_enhance"]
    end_steps = current["step"] >= taylor_dic["end_enhance"]
    interval_hit = taylor_dic["taylor_counter"] == taylor_dic["taylor_interval"] - 1

    if first_steps or interval_hit or end_steps:
        current["type"] = "full"
        taylor_dic["taylor_counter"] = 0
        current["activated_steps"].append(current["step"])
    elif taylor_dic["taylor_enabled"]:
        taylor_dic["taylor_counter"] += 1
        current["type"] = "taylor"
    else:
        raise ValueError("Unsupported TaylorSeer calculation type")


def derivative_approximation(taylor_dic: Dict, current: Dict, feature: torch.Tensor):
    if len(current["activated_steps"]) < 2:
        difference_distance = 1.0
    else:
        difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    prev_module_cache = taylor_dic["cache"][-1][current["layer"]][current["module"]]
    current_key_factors = {0: feature}
    prev_key_cache = prev_module_cache.get("default", None) if prev_module_cache else None

    for order in range(taylor_dic["max_order"]):
        has_prev = prev_key_cache is not None and order in prev_key_cache
        is_within = current["step"] < (current["num_steps"] - taylor_dic["first_enhance"] + 1)
        if not (has_prev and is_within):
            break
        current_key_factors[order + 1] = (
            current_key_factors[order] - prev_key_cache[order]
        ) / difference_distance

    taylor_dic["cache"][-1].setdefault(current["layer"], {})
    taylor_dic["cache"][-1][current["layer"]][current["module"]] = {
        "default": current_key_factors
    }


def taylor_formula(taylor_dic: Dict, current: Dict, prev_v: torch.Tensor = None, beta=0.5):
    x = current["step"] - current["activated_steps"][-1]
    factors = taylor_dic["cache"][-1][current["layer"]][current["module"]]["default"]

    result = 0
    for order, value in factors.items():
        result = result + (1 / math.factorial(order)) * value * (x ** order)

    if prev_v is not None:
        result = beta * prev_v + (1.0 - beta) * result
    return result


def taylor_cache_init(taylor_dic: Dict, current: Dict):
    """Kept as a named hook so sampler code mirrors FastSAM3D's layout."""
    if current["step"] == current["num_steps"] - 1:
        pass
