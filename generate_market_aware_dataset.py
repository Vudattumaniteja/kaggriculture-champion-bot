"""
Generate Multi-Task Market-Aware Elite Dataset CLI Entrypoint.
Simulates 320 full 720-step matches across 14 CPU cores (>460,000 transitions).
Composition:
- 50% Hybrid 4-Quadrant Farming
- 30% Market-Aware Dynamic Pricing
- 20% Extreme Stress Testing
Saves to data/market_aware_elite_dataset.npz and generates markdown report in .scratch/market_aware_elite_dataset_report.md.
"""

import sys
import os

from src.training.generator_market_aware import generate_market_aware_elite_dataset


def main():
    num_matches = 320
    save_path = "data/market_aware_elite_dataset.npz"
    base_seed = 9999
    num_cores = 14

    print("=" * 80)
    print(" Kaggriculture Multi-Task Market-Aware Elite Dataset Pipeline")
    print(f" Target: {save_path} | Matches: {num_matches} | Cores: {num_cores}")
    print("=" * 80)

    stats = generate_market_aware_elite_dataset(
        num_matches=num_matches,
        save_path=save_path,
        base_seed=base_seed,
        num_cores=num_cores,
    )

    print("\nDataset generation completed successfully.")


if __name__ == "__main__":
    main()
