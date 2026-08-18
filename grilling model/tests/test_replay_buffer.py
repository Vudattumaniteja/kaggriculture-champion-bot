import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
import torch

from src.replay_buffer import SumTree, PrioritizedReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.capacity = 1000
        self.buffer = PrioritizedReplayBuffer(capacity=self.capacity, alpha=0.6, beta_start=0.4, beta_end=1.0)

    def test_sumtree_insert_and_sample(self):
        tree = SumTree(capacity=100)
        tree.add(10.0, {"data": 1})
        tree.add(20.0, {"data": 2})
        tree.add(30.0, {"data": 3})
        self.assertAlmostEqual(tree.total_priority(), 60.0)

        idx, priority, data = tree.get(15.0)
        self.assertIsNotNone(data)

    def test_buffer_add_and_sample_batch(self):
        # Insert 100 transitions
        for i in range(100):
            self.buffer.add(
                transition={
                    "x_spatial": np.zeros((24, 10, 10), dtype=np.float32),
                    "x_scalar": np.zeros(72, dtype=np.float32),
                    "target_crop": np.zeros((5, 10, 10), dtype=np.float32),
                    "target_workforce": 1,
                    "target_land": 0.0,
                    "target_seed": np.zeros(5, dtype=np.float32),
                    "target_market": np.zeros(9, dtype=np.float32),
                    "target_value": 1000.0 * (i + 1),
                },
                td_error=10.0 * (i + 1)
            )

        self.assertEqual(len(self.buffer), 100)
        batch, tree_indices, is_weights = self.buffer.sample(batch_size=16)
        
        self.assertEqual(len(tree_indices), 16)
        self.assertEqual(len(is_weights), 16)
        self.assertIn("target_value", batch)
        self.assertEqual(len(batch["target_value"]), 16)
        # Weights should be normalized with max weight = 1.0
        self.assertAlmostEqual(float(np.max(is_weights)), 1.0, places=4)

    def test_update_priorities(self):
        self.buffer.add({"val": 1}, td_error=1.0)
        self.buffer.add({"val": 2}, td_error=1.0)
        
        _, indices, _ = self.buffer.sample(batch_size=2)
        new_td_errors = np.array([100.0, 50.0], dtype=np.float32)
        self.buffer.update_priorities(indices, new_td_errors)
        self.assertGreater(self.buffer.tree.total_priority(), 10.0)


if __name__ == "__main__":
    unittest.main()
