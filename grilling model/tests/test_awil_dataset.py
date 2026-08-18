import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
import torch

from src.awil_dataset import (
    apply_d4_augmentation,
    compute_awil_sample_weights,
    get_phase_bucket,
    parse_replay_transitions,
    AWILDataset,
)


class TestAWILDataset(unittest.TestCase):
    def setUp(self):
        # Synthetic transition
        self.spatial = np.zeros((24, 10, 10), dtype=np.float32)
        self.spatial[1, 2, 3] = 1.0  # Tile at row 2, col 3 has Carrot
        self.crop_target = np.zeros((5, 10, 10), dtype=np.float32)
        self.crop_target[1, 2, 3] = 1.0

        self.scalar = np.ones(72, dtype=np.float32) * 0.5
        self.market_fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)

    def test_d4_augmentation_rotations_and_reflections(self):
        # Test all 8 D4 transformations
        for transform_idx in range(8):
            aug_spatial, aug_crop_target, aug_scalar, aug_market = apply_d4_augmentation(
                spatial=self.spatial,
                crop_target=self.crop_target,
                scalar=self.scalar,
                market_fractions=self.market_fractions,
                transform_idx=transform_idx
            )
            self.assertEqual(aug_spatial.shape, (24, 10, 10))
            self.assertEqual(aug_crop_target.shape, (5, 10, 10))
            # Scalars and market fractions must remain invariant
            np.testing.assert_array_equal(aug_scalar, self.scalar)
            np.testing.assert_array_equal(aug_market, self.market_fractions)
            # Active element in crop target must match active element in spatial
            aug_spatial_carrot_pos = np.argwhere(aug_spatial[1] == 1.0)
            aug_target_carrot_pos = np.argwhere(aug_crop_target[1] == 1.0)
            np.testing.assert_array_equal(aug_spatial_carrot_pos, aug_target_carrot_pos)

    def test_phase_bucket_partitioning(self):
        self.assertEqual(get_phase_bucket(step=0), 0)      # Day 0 -> Bucket 0
        self.assertEqual(get_phase_bucket(step=140), 0)    # Day 5 -> Bucket 0
        self.assertEqual(get_phase_bucket(step=200), 1)    # Day 8 -> Bucket 1
        self.assertEqual(get_phase_bucket(step=350), 2)    # Day 14 -> Bucket 2
        self.assertEqual(get_phase_bucket(step=500), 3)    # Day 20 -> Bucket 3
        self.assertEqual(get_phase_bucket(step=700), 4)    # Day 29 -> Bucket 4

    def test_awil_sample_weight_bounds(self):
        # Generate 100 random returns and baseline values
        steps = np.random.randint(0, 720, size=100)
        returns = np.random.uniform(50000.0, 300000.0, size=100)
        values = np.random.uniform(40000.0, 280000.0, size=100)
        
        weights = compute_awil_sample_weights(steps=steps, returns=returns, baseline_values=values, tau=1.0)
        self.assertEqual(len(weights), 100)
        # All sample weights must be strictly bounded in [0.2, 5.0]
        self.assertTrue(np.all(weights >= 0.2), f"Min weight {weights.min()} < 0.2")
        self.assertTrue(np.all(weights <= 5.0), f"Max weight {weights.max()} > 5.0")

    def test_dataset_item_types(self):
        sample_transitions = [
            {
                "spatial": np.zeros((24, 10, 10), dtype=np.float32),
                "scalar": np.zeros(72, dtype=np.float32),
                "crop_heatmaps": np.zeros((5, 10, 10), dtype=np.float32),
                "workforce": 2,
                "land_expand": 0.0,
                "seed_replenish": np.zeros(5, dtype=np.float32),
                "market_fractions": np.zeros(9, dtype=np.float32),
                "terminal_return": 120000.0,
                "step": 50,
                "baseline_value": 110000.0
            }
        ]
        dataset = AWILDataset(transitions=sample_transitions, augment=True)
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        self.assertIn("x_spatial", item)
        self.assertIn("x_scalar", item)
        self.assertIn("sample_weight", item)
        self.assertGreaterEqual(item["sample_weight"].item(), 0.2)
        self.assertLessEqual(item["sample_weight"].item(), 5.0)


if __name__ == "__main__":
    unittest.main()
