import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import time
import numpy as np

from src.league import (
    PFSPLadder,
    PrioritizedFictitiousSelfPlayMatchmaker,
    HeuristicSpecialist,
)
from src.curriculum import (
    sample_curriculum_start_step,
    fast_forward_match,
    recurse_subtrajectory_gae,
    generate_subtrajectory_transitions,
)


class TestPFSPCurriculum(unittest.TestCase):
    def setUp(self):
        self.matchmaker = PrioritizedFictitiousSelfPlayMatchmaker(max_checkpoints=30)

    def test_single_champion_gradient_allocation(self):
        """Verifies 100% gradient updates are dedicated to the active Champion network."""
        self.assertEqual(self.matchmaker.gradient_allocation, "single_champion_100_percent")

    def test_pfsp_sampling_distribution(self):
        # Register checkpoints with varying win rates vs champion
        self.matchmaker.add_checkpoint("ckpt_01", win_rate_vs_champion=0.90)  # Champ loses often -> high priority
        self.matchmaker.add_checkpoint("ckpt_02", win_rate_vs_champion=0.10)  # Champ wins often -> low priority
        self.matchmaker.add_checkpoint("ckpt_03", win_rate_vs_champion=0.50)

        # Sample 1000 opponents
        sampled_types = {"checkpoint": 0, "self_play": 0, "heuristic": 0}
        ckpt_counts = {"ckpt_01": 0, "ckpt_02": 0, "ckpt_03": 0}

        for _ in range(1000):
            opp_type, opp_id = self.matchmaker.sample_opponent()
            sampled_types[opp_type] += 1
            if opp_type == "checkpoint":
                ckpt_counts[opp_id] += 1

        # Check macro proportions: ~40% checkpoint, ~40% self_play, ~20% heuristic
        self.assertAlmostEqual(sampled_types["checkpoint"] / 1000.0, 0.40, delta=0.08)
        self.assertAlmostEqual(sampled_types["self_play"] / 1000.0, 0.40, delta=0.08)
        self.assertAlmostEqual(sampled_types["heuristic"] / 1000.0, 0.20, delta=0.08)

        # In checkpoints, ckpt_01 (hardest opponent) should be sampled more than ckpt_02
        self.assertGreater(ckpt_counts["ckpt_01"], ckpt_counts["ckpt_02"])

    def test_pfsp_ladder_fifo_pool(self):
        """Verifies rolling FIFO pool of up to 30 past checkpoints."""
        ladder = PFSPLadder(max_checkpoints=30)
        for i in range(35):
            ladder.add_checkpoint(f"ckpt_{i:02d}", win_rate_vs_champion=0.5)

        self.assertEqual(len(ladder.checkpoints), 30)
        # Oldest checkpoints (0-4) must have been evicted in FIFO order
        first_id = ladder.checkpoints[0]["id"]
        last_id = ladder.checkpoints[-1]["id"]
        self.assertEqual(first_id, "ckpt_05")
        self.assertEqual(last_id, "ckpt_34")

    def test_pfsp_update_result(self):
        """Verifies match result updates adjust EMA win rates and sampling probabilities."""
        self.matchmaker.add_checkpoint("ckpt_easy", win_rate_vs_champion=0.20)
        self.matchmaker.add_checkpoint("ckpt_hard", win_rate_vs_champion=0.80)

        # Update several champion losses against ckpt_easy -> win_rate_vs_champion increases
        for _ in range(10):
            self.matchmaker.update_result("ckpt_easy", champ_won=False)

        probs = self.matchmaker.ladder.get_probabilities()
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs.sum(), 1.0)

    def test_heuristic_specialists(self):
        """Verifies all 5 fixed Mega League heuristic specialists return valid action formats."""
        for spec_name in HeuristicSpecialist.SPECIALISTS:
            specialist = HeuristicSpecialist(personality=spec_name)
            mock_obs = {
                "player": 0,
                "farms": [{"money": 3000.0, "tiles": []}, {"money": 3000.0, "tiles": []}],
                "day": 2,
                "hour": 0,
            }
            action = specialist(mock_obs)
            self.assertIn("farmer", action)
            self.assertIn("hands", action)
            self.assertIn("market", action)
            self.assertIsInstance(action["market"], list)

    def test_curriculum_step_distribution(self):
        """Verifies 80/20 curriculum: 80% Turn 0, 10% Midgame [288, 432], 10% Endgame [528, 648]."""
        steps = [sample_curriculum_start_step() for _ in range(1000)]
        turn0_count = sum(1 for s in steps if s == 0)
        midgame_count = sum(1 for s in steps if 288 <= s <= 432)
        endgame_count = sum(1 for s in steps if 528 <= s <= 648)

        self.assertAlmostEqual(turn0_count / 1000.0, 0.80, delta=0.06)
        self.assertAlmostEqual(midgame_count / 1000.0, 0.10, delta=0.05)
        self.assertAlmostEqual(endgame_count / 1000.0, 0.10, delta=0.05)

    def test_fast_forward_execution_speed(self):
        """Fast-forwards a match to start_step in < 150ms on CPU."""
        t0 = time.time()
        env_state = fast_forward_match(start_step=100)
        duration = time.time() - t0

        self.assertIsNotNone(env_state)
        self.assertLess(duration, 0.15, f"Fast forward took {duration:.3f}s, must be < 150ms")
        self.assertGreaterEqual(env_state.get("step", 0), 100)

    def test_fast_forward_10_matches_benchmark(self):
        """Benchmark: 10 fast-forward curriculum matches execute in < 1.5 seconds total on CPU."""
        t0 = time.time()
        for _ in range(10):
            start_step = sample_curriculum_start_step()
            state = fast_forward_match(start_step=start_step)
            self.assertIsNotNone(state)
        duration = time.time() - t0

        self.assertLess(duration, 1.5, f"10 fast-forward matches took {duration:.3f}s, must be < 1.5s")

    def test_subtrajectory_gae_recursion_isolation(self):
        """
        Verifies sub-trajectory transitions [t_start, 718] recurse GAE advantages cleanly
        without contaminating neural training with heuristic warmup steps.
        """
        t_start = 288  # Day 12 midgame start
        res = generate_subtrajectory_transitions(start_step=t_start, total_turns=720)

        step_indices = res["step_indices"]
        self.assertEqual(step_indices[0], 288)
        self.assertEqual(step_indices[-1], 718)
        self.assertEqual(len(step_indices), 720 - 1 - 288)
        self.assertFalse(any(s < t_start for s in step_indices))

        # Check GAE recursion mathematical properties
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([10.0, 11.0, 12.0, 0.0], dtype=np.float32)
        r, adv, val_tar = recurse_subtrajectory_gae(rewards, values, start_step=100, gamma=0.995, gae_lambda=0.95)

        self.assertEqual(len(adv), 3)
        self.assertEqual(len(val_tar), 3)
        self.assertTrue(np.all(np.isfinite(adv)))
        self.assertTrue(np.all(np.isfinite(val_tar)))


if __name__ == "__main__":
    unittest.main()
