"""
Generate Rich Multi-Industry Dataset CLI Entrypoint.
Simulates 50+ episodes across diverse crop, livestock, town arbitrage, and hybrid expert baselines.
Saves compressed dataset to data/multi_industry_dataset.npz and prints detailed statistical reports.
"""

import json
import os
import sys
import numpy as np

from src.training.dataset import generate_multi_industry_dataset


def main():
    num_episodes = 50
    save_path = "data/multi_industry_dataset.npz"
    base_seed = 3000

    print("=" * 80)
    print(" Kaggriculture Multi-Industry Simulation & Dataset Generator")
    print("=" * 80)

    stats = generate_multi_industry_dataset(
        num_episodes=num_episodes,
        save_path=save_path,
        base_seed=base_seed,
    )

    # Save summary report JSON
    json_path = "data/multi_industry_stats.json"
    clean_stats = {
        "total_samples": stats["total_samples"],
        "num_episodes": stats["num_episodes"],
        "grids_shape": stats["grids_shape"],
        "scalars_shape": stats["scalars_shape"],
        "actions_shape": stats["actions_shape"],
        "values_shape": stats["values_shape"],
        "duration_seconds": stats["duration_seconds"],
        "file_size_mb": stats["file_size_mb"],
        "action_distribution": {k: {"count": v[0], "percentage": round(v[1], 2)} for k, v in stats["action_distribution"].items()},
        "agent_mean_banks": {k: {"mean_bank": round(v[0], 1), "matches": v[1]} for k, v in stats["agent_mean_banks"].items()},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=2)

    print("\n" + "=" * 80)
    print(" SUMMARY STATISTICS REPORT")
    print("=" * 80)
    print(f"Total Trajectory Samples : {stats['total_samples']:,} state-action-value transitions")
    print(f"Total Match Episodes     : {stats['num_episodes']}")
    print(f"Spatial Grid Tensor Shape: {stats['grids_shape']}")
    print(f"Global Scalar Shape      : {stats['scalars_shape']}")
    print(f"NPZ Compressed File Size : {stats['file_size_mb']:.2f} MB")
    print(f"Stats JSON File Saved    : {json_path}")
    print("\n--- Action Class Distribution ---")
    print(f"{'Macro Action':<30} | {'Samples':<10} | {'Percentage':<10}")
    print("-" * 56)
    for act_name, (cnt, pct) in stats["action_distribution"].items():
        print(f"{act_name:<30} | {cnt:<10,} | {pct:>8.2f}%")

    print("\n--- Baseline Agent Performance (Average Final Bank Balance) ---")
    print(f"{'Agent Baseline':<25} | {'Episodes':<10} | {'Mean Final Bank':<18}")
    print("-" * 58)
    sorted_agents = sorted(stats["agent_mean_banks"].items(), key=lambda x: x[1][0], reverse=True)
    for agent_name, (mean_b, match_cnt) in sorted_agents:
        print(f"{agent_name:<25} | {match_cnt:<10} | ${mean_b:>14,.1f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
