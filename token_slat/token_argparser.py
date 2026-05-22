
import argparse
import math


def parse_token_args():
    parser = argparse.ArgumentParser(
        description="Trellis with token pruning and sampler acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    mode_group = parser.add_argument_group('Mode Control & Steps')
    mode_group.add_argument("--use_token", action="store_true", help="Token pruning acceleration.")
    mode_group.add_argument("--euler_steps", type=int, default=25, help="--")
    
    token_strategy_group = parser.add_argument_group('Token Pruning Scheduling Strategy')
    token_strategy_group.add_argument("--anchor_ratio", type=float, default=None,
                                     help="--")
    token_strategy_group.add_argument("--assumed_slope", type=float, default=None, help="--")

    token_strategy_group.add_argument("--full_sampling_ratio", type=float, default=0.2,
                                     help="Phase 1 of token pruning.")
    token_strategy_group.add_argument("--full_sampling_end_ratio", type=float, default=0.75,
                                     help="Phase 2 of token pruning.")
    token_strategy_group.add_argument("--aggressive_cache_ratio", type=float, default=0.7,
                                     help="Phase 3 of token pruning.")

    token_strategy_group.add_argument("--final_phase_correction_freq", type=int, default=3,
                                     help="f_corr")

    io_group = parser.add_argument_group('Trellis I/O & Internal Options')
    io_group.add_argument("--seed", type=int, default=42, help="SEED。")
    io_group.add_argument("--resolution", type=int, default=16, help="Gs Stage。")
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()  

    active_sampler = "Euler"; args.effective_steps = args.euler_steps
    if args.assumed_slope is None: args.assumed_slope = -0.07
    if args.anchor_ratio is None: args.anchor_ratio = 0.2
    args.use_token = True
    if args.use_token:
        args.full_sampling_steps = math.floor(args.effective_steps * args.full_sampling_ratio)
        args.full_sampling_end_steps = math.ceil(args.effective_steps * args.full_sampling_end_ratio)
        calculated_anchor_step = math.floor(args.effective_steps * args.anchor_ratio)
        args.anchor_step = max(1, calculated_anchor_step)
    return args
