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
)


class TestPFSPCurriculum(unittest.TestCase):
    def setUp(self):
        self.matchmaker = PrioritizedFictitiousSelfPlayMatchmaker(max_checkpoints=30)

    def test_pfsp_sampling_distribution(self):
        # Register a few checkpoints with varying win rates against champion
        self.matchmaker.add_checkpoint("ckpt_01", win_rate_vs_champion=0.90)  # Champ loses a lot -> high priority
        self.matchmaker.add_checkpoint("ckpt_02", win_rate_vs_champion=0.10)  # Champ wins a lot -> low priority
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

    def test_curriculum_step_distribution(self):
        steps = [sample_curriculum_start_step() for _ in range(1000)]
        turn0_count = sum(1 for s in steps if s == 0)
        midgame_count = sum(1 for s in steps if 288 <= s <= 432)
        endgame_count = sum(1 for s in steps if 528 <= s <= 648)

        self.assertAlmostEqual(turn0_count / 1000.0, 0.80, delta=0.06)
        self.assertAlmostEqual(midgame_count / 1000.0, 0.10, delta=0.05)
        self.assertAlmostEqual(endgame_count / 1000.0, 0.10, delta=0.05)

    def test_fast_forward_execution_speed(self):
        t0 = time.time()
        # Fast forward into midgame (step 100)
        env_state = fast_forward_match(start_step=100)
        duration = time.time() - t0
        
        self.assertIsNotNone(env_state)
        self.assertLess(duration, 0.35, f"Fast forward took {duration:.3f}s, must be < 350ms")
        self.assertGreaterEqual(env_state.get("step", 0), 100)


if __name__ == "__main__":
    unittest.main()
