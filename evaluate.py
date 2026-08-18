"""
CLI entrypoint to run Kaggriculture tournaments and evaluate bots locally.

Usage examples:
    python evaluate.py --agent1 starter --agent2 random --episodes 5
    python evaluate.py --agent1 starter --agent2 pass --episodes 3 --save-replays
"""

import argparse
import os
import sys

from src.evaluation.runner import run_tournament


def main():
    parser = argparse.ArgumentParser(description="Evaluate Kaggriculture agents locally.")
    parser.add_argument("--agent1", type=str, default="starter", help="First agent (name, script path, or baseline)")
    parser.add_argument("--agent2", type=str, default="random", help="Second agent (name, script path, or baseline)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to simulate")
    parser.add_argument("--steps", type=int, default=720, help="Number of turns per episode (default 720)")
    parser.add_argument("--no-swap", action="store_true", help="Disable player 0/1 side swapping")
    parser.add_argument("--save-replays", action="store_true", help="Save HTML/JSON match replays")
    parser.add_argument("--replays-dir", type=str, default="replays", help="Directory for match replays")
    parser.add_argument("--report-file", type=str, default=None, help="Path to save markdown tournament report")

    args = parser.parse_args()

    summary = run_tournament(
        agent0=args.agent1,
        agent1=args.agent2,
        episodes=args.episodes,
        episode_steps=args.steps,
        swap_positions=not args.no_swap,
        save_replays=args.save_replays,
        replays_dir=args.replays_dir,
    )

    if args.report_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_file)), exist_ok=True)
        with open(args.report_file, "w", encoding="utf-8") as f:
            f.write(summary.format_markdown_table())
        print(f"Tournament report saved to: {args.report_file}")


if __name__ == "__main__":
    main()
