import torch.nn.functional as F


def faster_init(num_steps):
    """Initialize Faster/token-cache state for TRELLIS SLAT sampling."""
    faster_state = {
        "k": None,
        "prev_x": None,
        "prev_v": None,
        "prev_prev_x": None,
        "error": None,
        "feature": None,
        "easy": None,
    }
    faster_dic = {
        "thresh": 1.0,
        "dir_weight": 0.5,
        "faster_counter": 0,
        "cache": faster_state,
        "faster_enabled": True,
        "max_order": 2,
        "first_enhance": 1,
    }
    current = {
        "type": None,
        "activated_steps": [],
        "step": 0,
        "num_steps": num_steps,
        "use_token": True,
        "is_token_active": False,
        "num_to_skip": 0,
        "cache_indices": None,
        "fast_update_indices": None,
    }
    return faster_dic, current


def faster_cal_type(faster_dic, current, input):
    faster_state = faster_dic["cache"]
    is_first_steps = current["step"] < faster_dic["first_enhance"]
    should_calc = True

    if not is_first_steps:
        has_history = (
            faster_state["prev_x"] is not None
            and faster_state["prev_v"] is not None
            and faster_state["prev_prev_x"] is not None
        )
        if has_history:
            delta_x = input - faster_state["prev_x"]
            input_change_mag = delta_x.abs().mean()
            prev_delta_x = faster_state["prev_x"] - faster_state["prev_prev_x"]
            cos_sim = F.cosine_similarity(
                delta_x.reshape(1, -1),
                prev_delta_x.reshape(1, -1),
                dim=1,
            )
            direction_error = 1.0 - cos_sim.item()

            if faster_state["k"] is not None:
                output_norm = faster_state["prev_v"].abs().mean() + 1e-6
                mag_error = faster_state["k"] * (input_change_mag / output_norm)
                faster_state["error"] += mag_error
                should_calc = faster_state["error"] >= faster_dic["thresh"]
            else:
                should_calc = True
        else:
            should_calc = True

    if should_calc:
        faster_state["error"] = 0
        current["type"] = "full"
        faster_dic["faster_counter"] = 0
        current["activated_steps"].append(current["step"])
    else:
        current["type"] = "faster"
        faster_dic["faster_counter"] += 1

    return should_calc
