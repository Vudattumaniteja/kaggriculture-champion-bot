import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import time
import os
import tempfile
import numpy as np
import torch

from src.replay_buffer import (
    SumTree,
    PrioritizedReplayBuffer,
    CausalActionLogger,
    extract_verified_macro_actions,
)


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.capacity = 1000
        self.buffer = PrioritizedReplayBuffer(
            capacity=self.capacity,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_steps=1000,
        )

    def test_sumtree_insert_and_sample(self):
        tree = SumTree(capacity=100)
        tree.add(10.0, {"data": 1})
        tree.add(20.0, {"data": 2})
        tree.add(30.0, {"data": 3})
        self.assertAlmostEqual(tree.total_priority(), 60.0)

        idx, priority, data = tree.get(15.0)
        self.assertIsNotNone(data)

    def test_sumtree_proportional_distribution(self):
        tree = SumTree(capacity=10)
        tree.add(10.0, "low")
        tree.add(90.0, "high")
        self.assertAlmostEqual(tree.total_priority(), 100.0)

        counts = {"low": 0, "high": 0}
        total_p = tree.total_priority()
        np.random.seed(42)
        for _ in range(1000):
            val = np.random.uniform(0, total_p)
            _, _, item = tree.get(val)
            counts[item] += 1

        # "high" should be sampled roughly 9x more often than "low"
        self.assertGreater(counts["high"], 750)
        self.assertLess(counts["low"], 250)

    def test_sumtree_capacity_wrap_around(self):
        tree = SumTree(capacity=4)
        for i in range(10):
            tree.add(float(i + 1), f"item_{i}")

        self.assertEqual(tree.size, 4)
        # Last 4 added: item_6 (7.0), item_7 (8.0), item_8 (9.0), item_9 (10.0)
        # Total priority should be 7 + 8 + 9 + 10 = 34.0
        self.assertAlmostEqual(tree.total_priority(), 34.0)

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

    def test_importance_sampling_weights_and_beta_annealing(self):
        buf = PrioritizedReplayBuffer(
            capacity=100,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_steps=100,
        )
        for i in range(50):
            buf.add({"val": i}, td_error=float(i + 1))

        self.assertAlmostEqual(buf.beta, 0.4)
        _, _, weights1 = buf.sample(batch_size=10)
        self.assertGreater(buf.beta, 0.4)
        self.assertAlmostEqual(float(np.max(weights1)), 1.0, places=4)
        self.assertTrue(np.all(weights1 > 0.0))
        self.assertTrue(np.all(weights1 <= 1.0))

        # Sample multiple times to step beta towards 1.0
        for _ in range(120):
            buf.sample(batch_size=10)

        self.assertAlmostEqual(buf.beta, 1.0)

    def test_update_priorities(self):
        self.buffer.add({"val": 1}, td_error=1.0)
        self.buffer.add({"val": 2}, td_error=1.0)

        _, indices, _ = self.buffer.sample(batch_size=2)
        new_td_errors = np.array([100.0, 50.0], dtype=np.float32)
        self.buffer.update_priorities(indices, new_td_errors)
        self.assertGreater(self.buffer.tree.total_priority(), 10.0)

    def test_causal_action_logger_effective_actions(self):
        mock_obs = {
            "player": 0,
            "step": 100,
            "day": 4,
            "farms": [
                {
                    "money": 1500,
                    "unlocked_quadrants": ["NW"],
                    "hands": [[4, 4]],
                    "tiles": [[{"animal": "COW", "kind": "PASTURE"} if (r == 0 and c == 0) else None for c in range(10)] for r in range(10)],
                },
                {"money": 1000, "unlocked_quadrants": ["NW"], "hands": [], "tiles": [[None]*10]*10}
            ],
            "private": {
                "shed": {"WHEAT": 10, "CARROT": 20},
                "seeds": {"CARROT": 5},
                "inventories": [{}, {}]
            }
        }

        executed_market = [
            ["SELL", "WHEAT", 6],
            ["SELL", "CARROT", 20],
            ["BUY_LAND"],
            ["BUY_SEED", "CARROT", 3],
        ]
        farmer_act = ["PLANT", "CARROT"]
        hands_acts = [["WATER"]]

        verified = extract_verified_macro_actions(
            obs=mock_obs,
            farmer_action=farmer_act,
            hands_actions=hands_acts,
            market_orders=executed_market,
            raw_intents={
                "raw_market_fractions": np.ones(9, dtype=np.float32),
                "raw_land_expand": 1.0,
            }
        )

        self.assertIn("target_crop", verified)
        self.assertIn("target_workforce", verified)
        self.assertIn("target_land", verified)
        self.assertIn("target_seed", verified)
        self.assertIn("target_market", verified)

        # Verified market fractions: WHEAT sold 6 / 10 = 0.6, CARROT sold 20 / 20 = 1.0
        self.assertAlmostEqual(verified["target_market"][0], 0.6, places=4)
        self.assertAlmostEqual(verified["target_market"][1], 1.0, places=4)
        self.assertEqual(verified["target_land"], 1.0)
        self.assertEqual(verified["target_seed"][1], 3.0)
        self.assertEqual(verified["target_workforce"], 1)

        # Test logger full step
        transition = CausalActionLogger.log_step(
            obs=mock_obs,
            farmer_action=farmer_act,
            hands_actions=hands_acts,
            market_orders=executed_market,
            reward=50.0,
        )
        self.assertIn("x_spatial", transition)
        self.assertIn("x_scalar", transition)
        self.assertIn("target_market", transition)
        self.assertEqual(transition["x_spatial"].shape, (24, 10, 10))
        self.assertEqual(transition["x_scalar"].shape, (72,))

    def test_serialization_and_checkpoint_resuming(self):
        buf = PrioritizedReplayBuffer(capacity=50, alpha=0.6, beta_start=0.4, beta_end=1.0)
        for i in range(20):
            buf.add(
                transition={
                    "x_spatial": np.ones((24, 10, 10), dtype=np.float32) * i,
                    "x_scalar": np.ones(72, dtype=np.float32) * i,
                    "target_value": float(i * 100),
                },
                td_error=float(i + 1)
            )

        # Sample to advance step_count and beta
        buf.sample(batch_size=4)
        saved_step_count = buf.step_count
        saved_beta = buf.beta
        saved_total_p = buf.tree.total_priority()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "per_buffer.pt")
            buf.save_checkpoint(save_path)
            self.assertTrue(os.path.exists(save_path))

            # Resume in a fresh buffer
            resumed_buf = PrioritizedReplayBuffer(capacity=50)
            resumed_buf.load_checkpoint(save_path)

            self.assertEqual(len(resumed_buf), 20)
            self.assertEqual(resumed_buf.step_count, saved_step_count)
            self.assertAlmostEqual(resumed_buf.beta, saved_beta)
            self.assertAlmostEqual(resumed_buf.tree.total_priority(), saved_total_p)

            # Sample from resumed buffer
            batch, idxs, weights = resumed_buf.sample(batch_size=4)
            self.assertEqual(len(idxs), 4)
            self.assertEqual(len(weights), 4)
            self.assertIn("target_value", batch)

    def test_buffer_throughput_benchmark(self):
        """
        Verifies replay buffer sampling throughput > 5,000 transitions/sec on CPU.
        """
        buf = PrioritizedReplayBuffer(capacity=5000, alpha=0.6, beta_start=0.4, beta_end=1.0)
        dummy_trans = {
            "x_spatial": np.zeros((24, 10, 10), dtype=np.float32),
            "x_scalar": np.zeros(72, dtype=np.float32),
            "target_crop": np.zeros((5, 10, 10), dtype=np.float32),
            "target_workforce": 1,
            "target_land": 0.0,
            "target_seed": np.zeros(5, dtype=np.float32),
            "target_market": np.zeros(9, dtype=np.float32),
            "target_value": 50000.0,
        }

        # Populate buffer with 2000 transitions
        for i in range(2000):
            buf.add(dummy_trans, td_error=float((i % 100) + 1))

        # Benchmark sampling 20 batches of size 256 = 5120 transitions
        num_batches = 20
        batch_size = 256
        total_transitions = num_batches * batch_size

        t0 = time.perf_counter()
        for _ in range(num_batches):
            _ = buf.sample(batch_size=batch_size)
        elapsed = time.perf_counter() - t0

        throughput = total_transitions / max(elapsed, 1e-6)
        print(f"\nPER Sampling Throughput: {throughput:.1f} transitions/sec (elapsed: {elapsed*1000:.1f}ms for {total_transitions} transitions)")
        self.assertGreater(
            throughput,
            5000.0,
            f"Throughput {throughput:.1f} trans/sec did not meet > 5,000 trans/sec threshold",
        )


if __name__ == "__main__":
    unittest.main()
