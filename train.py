"""
Headless training script — works locally and in Google Colab.

Usage
─────
  python train.py                          # defaults: AZ, 7×7, 200 iters
  python train.py --method alphazero --size 7 --iterations 500 --sims 100 --parallel 16
  python train.py --method ppo       --size 7 --iterations 2000
  python train.py --method reinforce --size 5 --iterations 5000
  python train.py --method qlearning --size 5 --episodes  100000
  python train.py --method dqn       --size 5 --episodes  10000
  python train.py --method dqn_minimax --size 5 --episodes 10000
  python train.py --method dqn_sb3   --size 7 --timesteps 300000 --opponent self
  python train.py --plot             # plot saved metrics after training

Colab quick-start
─────────────────
  !git clone <repo-url>
  %cd fhtw_hex_2026
  !pip install torch numpy
  !python train.py --method alphazero --size 5 --iterations 200 --sims 50 --parallel 16
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Train a Hex RL agent (headless)")
    p.add_argument("--method",     default="alphazero",
                   choices=["alphazero", "ppo", "reinforce", "dqn", "dqn_minimax", "dqn_sb3", "qlearning"],
                   help="RL method to train")
    p.add_argument("--size",       type=int,   default=7,   help="Board size (default 7)")

    # AlphaZero
    p.add_argument("--iterations",   type=int, default=200, help="[AZ/PPO] training iterations")
    p.add_argument("--sims",         type=int, default=50,  help="[AZ] MCTS sims per move")
    p.add_argument("--parallel",     type=int, default=8,
                   help="[AZ] parallel MCTS leaves per batch (must be <= sims/7 or tree search collapses)")
    p.add_argument("--resume-from",  type=int, default=0,
                   help="[AZ] resume from checkpoint at this iteration (e.g. 60 → loads alphazero_NxN_ckpt0060.pt)")
    p.add_argument("--channels",     type=int, default=128,
                   help="[AZ] residual feature channels (128 for ≤9×9, 256 for ≥11×11)")
    p.add_argument("--blocks",       type=int, default=5,
                   help="[AZ] residual blocks (5 for ≤9×9, 8 for ≥11×11)")
    p.add_argument("--workers",      type=int, default=1,
                   help="[AZ] parallel CPU self-play workers (default 1; try cpu_count-1)")

    # PPO / REINFORCE / DQN / Q-Learning
    p.add_argument("--episodes",   type=int,   default=None,
                   help="[PPO/REINFORCE/DQN/QL] override episode count")

    # DQN-SB3 (Stable-Baselines3)
    p.add_argument("--timesteps",  type=int,   default=300_000,
                   help="[dqn_sb3] total environment timesteps")
    p.add_argument("--opponent",   default="self", choices=["self", "random"],
                   help="[dqn_sb3] self-play snapshots (self) or uniform-random opponent")

    # Output
    p.add_argument("--plot",       action="store_true", help="Plot metrics after training")
    p.add_argument("--save-metrics", metavar="FILE",
                   help="Save metrics dict as JSON to FILE")
    return p.parse_args()


def train_alphazero(args):
    from submission.facade_alphazero import train, plot_metrics
    metrics = train(
        size        = args.size,
        iterations  = args.iterations,
        sims        = args.sims,
        parallel    = args.parallel,
        resume_from = args.resume_from,
        channels    = args.channels,
        blocks      = args.blocks,
        n_workers   = args.workers,
    )
    if args.plot:
        plot_metrics(metrics)
    return metrics


def train_ppo(args):
    from submission.facade_ppo import train, plot_metrics
    iters = args.episodes or args.iterations
    metrics = train(size=args.size, iterations=iters)
    if args.plot:
        plot_metrics(metrics)
    return metrics


def train_reinforce(args):
    from submission.facade_reinforce import train, plot_metrics
    eps = args.episodes or 15_000
    metrics = train(size=args.size, episodes=eps)
    if args.plot:
        plot_metrics(metrics)
    return metrics


def train_dqn(args):
    from submission.facade_dqn import train, plot_metrics
    eps = args.episodes or 10_000
    metrics = train(size=args.size, episodes=eps)
    if args.plot:
        plot_metrics(metrics)
    return metrics


def train_dqn_sb3(args):
    from submission.facade_dqn_sb3 import train, plot_metrics
    metrics = train(size=args.size, total_timesteps=args.timesteps, opponent=args.opponent)
    if args.plot:
        plot_metrics(metrics)
    return metrics


def train_dqn_minimax(args):
    from submission.facade_dqn_minimax import train, plot_metrics
    eps = args.episodes or 10_000
    metrics = train(size=args.size, episodes=eps)
    if args.plot:
        plot_metrics(metrics)
    return metrics


def train_qlearning(args):
    from submission.facade_qlearning import train
    eps = args.episodes or 100_000
    train(size=args.size, episodes=eps)
    return {}


TRAINERS = {
    "alphazero":   train_alphazero,
    "ppo":         train_ppo,
    "reinforce":   train_reinforce,
    "dqn":         train_dqn,
    "dqn_minimax": train_dqn_minimax,
    "dqn_sb3":     train_dqn_sb3,
    "qlearning":   train_qlearning,
}


def main():
    args = parse_args()

    print(f"Method : {args.method}")
    print(f"Board  : {args.size}×{args.size}")
    if args.method == "alphazero":
        print(f"Iters  : {args.iterations}  sims/move={args.sims}  parallel={args.parallel}")
    elif args.method == "dqn_sb3":
        print(f"Steps  : {args.timesteps:,}  opponent={args.opponent}")
    elif args.method in ("ppo",):
        print(f"Iters  : {args.episodes or args.iterations}")
    else:
        print(f"Episodes: {args.episodes}")
    print()

    metrics = TRAINERS[args.method](args)

    if args.save_metrics and metrics:
        path = args.save_metrics
        # tensors aren't JSON-serialisable; convert to plain Python
        safe = {}
        for k, v in metrics.items():
            if isinstance(v, list) and v and hasattr(v[0], "item"):
                safe[k] = [x.item() for x in v]
            else:
                safe[k] = v
        with open(path, "w") as f:
            json.dump(safe, f, indent=2)
        print(f"Metrics saved → {path}")


if __name__ == "__main__":
    main()
