import argparse
import math


def parse_token_args():
    parser = argparse.ArgumentParser(
        description="TRELLIS token pruning and sampler acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode_group = parser.add_argument_group("Mode Control & Steps")
    mode_group.add_argument("--use_token", action="store_true", help="Token pruning acceleration.")
    mode_group.add_argument("--euler_steps", type=int, default=25, help="Euler sampling steps.")

    strategy_group = parser.add_argument_group("Token Pruning Scheduling Strategy")
    strategy_group.add_argument("--anchor_ratio", type=float, default=None)
    strategy_group.add_argument("--assumed_slope", type=float, default=None)
    strategy_group.add_argument("--full_sampling_ratio", type=float, default=0.2)
    strategy_group.add_argument("--full_sampling_end_ratio", type=float, default=0.75)
    strategy_group.add_argument("--aggressive_cache_ratio", type=float, default=0.7)
    strategy_group.add_argument("--final_phase_correction_freq", type=int, default=3)

    io_group = parser.add_argument_group("TRELLIS I/O & Internal Options")
    io_group.add_argument("--seed", type=int, default=42)
    io_group.add_argument("--resolution", type=int, default=16)

    args, _ = parser.parse_known_args()
    args.effective_steps = args.euler_steps
    args.assumed_slope = -0.07 if args.assumed_slope is None else args.assumed_slope
    args.anchor_ratio = 0.2 if args.anchor_ratio is None else args.anchor_ratio
    args.use_token = True

    if args.use_token:
        args.full_sampling_steps = math.floor(args.effective_steps * args.full_sampling_ratio)
        args.full_sampling_end_steps = math.ceil(args.effective_steps * args.full_sampling_end_ratio)
        args.anchor_step = max(1, math.floor(args.effective_steps * args.anchor_ratio))

    return args
