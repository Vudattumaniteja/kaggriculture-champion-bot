import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
import torch

from src.pbrs_gae import (
    compute_state_potential,
    compute_pbrs_step_reward,
    TrajectoryGAEProcessor,
)


class TestPBRSGAE(unittest.TestCase):
    def setUp(self):
        self.processor = TrajectoryGAEProcessor(gamma=0.995, gae_lambda=0.95)

    def test_state_potential_deadweight_decay(self):
        # Day 0 (step 0): assets should have full potential value
        obs_early = {
            "step": 0, "day": 0,
            "farms": [{"money": 3000.0, "tiles": []}, {"money": 3000.0, "tiles": []}],
            "private": {"shed": {"WHEAT": 10, "MELON": 5}, "seeds": {"MELON": 4}}
        }
        pot_early = compute_state_potential(obs_early, player=0)
        
        # Day 29 Turn 718 (step 718): shed goods and seeds should decay to 0 potential
        obs_late = {
            "step": 718, "day": 29,
            "farms": [{"money": 3000.0, "tiles": []}, {"money": 3000.0, "tiles": []}],
            "private": {"shed": {"WHEAT": 10, "MELON": 5}, "seeds": {"MELON": 4}}
        }
        pot_late = compute_state_potential(obs_late, player=0)
        
        # At step 718, unliquidated shed goods receive 0 potential or deadweight penalty
        self.assertLess(pot_late, pot_early)

    def test_reward_telescoping_undiscounted(self):
        # Build synthetic 10-step trajectory
        cash_history = [3000.0, 3100.0, 3050.0, 3400.0, 3600.0, 4000.0, 4500.0, 5000.0, 6000.0, 7500.0]
        obs_sequence = []
        for t, cash in enumerate(cash_history):
            obs_sequence.append({
                "step": t, "day": t // 24,
                "farms": [{"money": cash, "tiles": []}, {"money": 3000.0, "tiles": []}],
                "private": {"shed": {}, "seeds": {}}
            })

        # Compute undiscounted shaped rewards (gamma=1.0)
        shaped_rewards = []
        for t in range(len(obs_sequence) - 1):
            s_t = obs_sequence[t]
            s_next = obs_sequence[t + 1]
            raw_r = 0.0
            r_prime = compute_pbrs_step_reward(s_t, s_next, raw_reward=raw_r, gamma=1.0, player=0)
            shaped_rewards.append(r_prime)

        # Sum of shaped rewards should telescopically match terminal cash delta: Cash_T - Cash_0
        total_shaped = sum(shaped_rewards)
        expected_delta = cash_history[-1] - cash_history[0]
        self.assertAlmostEqual(total_shaped, expected_delta, places=2)

    def test_gae_processor_advantage_bounds(self):
        T = 50
        rewards = np.random.normal(loc=10.0, scale=5.0, size=T).astype(np.float32)
        values = np.random.uniform(100.0, 500.0, size=T + 1).astype(np.float32)
        values[-1] = 0.0  # Terminal state value = 0

        shaped_r, adv, targets = self.processor.process_trajectory(
            rewards=rewards,
            values=values
        )

        self.assertEqual(len(adv), T)
        self.assertEqual(len(targets), T)
        self.assertFalse(np.isnan(adv).any())
        self.assertFalse(np.isnan(targets).any())


if __name__ == "__main__":
    unittest.main()
