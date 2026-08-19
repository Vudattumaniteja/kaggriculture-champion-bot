import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
import torch

from src.pbrs_gae import (
    MaturityAwarePBRS,
    compute_state_potential,
    compute_pbrs_step_reward,
    TrajectoryGAEProcessor,
)
from src.encoder import CROPS, PRODUCTS, ANIMALS, CROP_SPECS, BASE_PRICES
from src.network import NUM_BINS, V_MIN, V_MAX, two_hot_to_value


class TestPBRSGAE(unittest.TestCase):
    def setUp(self):
        self.pbrs = MaturityAwarePBRS(gamma=0.995, total_turns=720, deadweight_penalty_coeff=2.0)
        self.processor = TrajectoryGAEProcessor(
            gamma=0.995,
            gae_lambda=0.95,
            num_bins=NUM_BINS,
            v_min=V_MIN,
            v_max=V_MAX
        )

    def test_state_potential_deadweight_decay(self):
        # Day 0 (step 0): assets should have full potential value
        obs_early = {
            "step": 0, "day": 0,
            "farms": [{"money": 3000.0, "tiles": []}, {"money": 3000.0, "tiles": []}],
            "private": {"shed": {"WHEAT": 10, "MELON": 5}, "seeds": {"MELON": 4}}
        }
        pot_early = compute_state_potential(obs_early, player=0)
        
        # Day 29 Turn 718 (step 718): shed goods and seeds should decay to 0 potential with deadweight penalty
        obs_late = {
            "step": 718, "day": 29,
            "farms": [{"money": 3000.0, "tiles": []}, {"money": 3000.0, "tiles": []}],
            "private": {"shed": {"WHEAT": 10, "MELON": 5}, "seeds": {"MELON": 4}}
        }
        pot_late = compute_state_potential(obs_late, player=0)
        
        # At step 718, unliquidated shed goods receive deadweight penalty
        self.assertLess(pot_late, pot_early)
        # Verify deadweight penalty is 2.0x value of remaining shed + seeds
        wheat_val = 10 * BASE_PRICES["WHEAT"]
        melon_shed_val = 5 * BASE_PRICES["MELON"]
        melon_seed_val = 4 * CROP_SPECS["MELON"]["seed"]
        expected_deadweight = wheat_val + melon_shed_val + melon_seed_val
        expected_pot_late = 3000.0 - 2.0 * expected_deadweight
        self.assertAlmostEqual(pot_late, expected_pot_late, places=2)

    def test_maturity_decay_schedules(self):
        # 1. Crops: Melon takes 12 days = 288 turns
        melon_tile = [{"kind": "PLANT", "crop": "MELON", "growth": 0.0}]
        # rem_turns >= 288: full potential (weight = 1.0)
        obs_early_melon = {
            "step": 0, "farms": [{"money": 3000.0, "tiles": [melon_tile]}],
            "private": {"shed": {}, "seeds": {}}
        }
        pot_early_melon = self.pbrs.compute_potential(obs_early_melon, player=0)
        melon_base_val = BASE_PRICES["MELON"] * 2.0
        self.assertAlmostEqual(pot_early_melon, 3000.0 + melon_base_val, places=2)

        # rem_turns < 24 (step > 696): weight = 0.0
        obs_late_melon = {
            "step": 700, "farms": [{"money": 3000.0, "tiles": [melon_tile]}],
            "private": {"shed": {}, "seeds": {}}
        }
        pot_late_melon = self.pbrs.compute_potential(obs_late_melon, player=0)
        self.assertAlmostEqual(pot_late_melon, 3000.0, places=2)

        # 2. Animals: 10-day break-even = 240 turns
        cow_tile = [{"kind": "PASTURE", "animal": "COW"}]
        # rem_turns >= 240 (step <= 480): weight = 1.0
        obs_early_cow = {
            "step": 200, "farms": [{"money": 3000.0, "tiles": [cow_tile]}],
            "private": {"shed": {}, "seeds": {}}
        }
        pot_early_cow = self.pbrs.compute_potential(obs_early_cow, player=0)
        self.assertAlmostEqual(pot_early_cow, 3000.0 + 400.0, places=2)

        # rem_turns < 48 (step > 672): weight = 0.0
        obs_late_cow = {
            "step": 680, "farms": [{"money": 3000.0, "tiles": [cow_tile]}],
            "private": {"shed": {}, "seeds": {}}
        }
        pot_late_cow = self.pbrs.compute_potential(obs_late_cow, player=0)
        self.assertAlmostEqual(pot_late_cow, 3000.0, places=2)

        # 3. Seeds: Melon seed (12 days + 24 turns = 312 turns)
        # rem_turns >= 312: weight = 1.0
        obs_seed_ok = {
            "step": 100, "farms": [{"money": 3000.0, "tiles": []}],
            "private": {"shed": {}, "seeds": {"MELON": 2}}
        }
        pot_seed_ok = self.pbrs.compute_potential(obs_seed_ok, player=0)
        self.assertAlmostEqual(pot_seed_ok, 3000.0 + 2 * CROP_SPECS["MELON"]["seed"], places=2)

        # rem_turns < 312 (e.g. step = 500): weight = 0.0
        obs_seed_expired = {
            "step": 500, "farms": [{"money": 3000.0, "tiles": []}],
            "private": {"shed": {}, "seeds": {"MELON": 2}}
        }
        pot_seed_expired = self.pbrs.compute_potential(obs_seed_expired, player=0)
        self.assertAlmostEqual(pot_seed_expired, 3000.0, places=2)

    def test_purges_3k_bank_kink(self):
        # Potential must be smooth and linear across $3000 threshold (no piecewise kink / artificial bonus)
        obs_below = {"step": 100, "farms": [{"money": 2999.0, "tiles": []}], "private": {}}
        obs_exact = {"step": 100, "farms": [{"money": 3000.0, "tiles": []}], "private": {}}
        obs_above = {"step": 100, "farms": [{"money": 3001.0, "tiles": []}], "private": {}}

        pot_below = self.pbrs.compute_potential(obs_below, player=0)
        pot_exact = self.pbrs.compute_potential(obs_exact, player=0)
        pot_above = self.pbrs.compute_potential(obs_above, player=0)

        self.assertAlmostEqual(pot_exact - pot_below, 1.0, places=4)
        self.assertAlmostEqual(pot_above - pot_exact, 1.0, places=4)

    def test_50_synthetic_trajectories_exact_telescoping(self):
        np.random.seed(42)
        pbrs_undiscounted = MaturityAwarePBRS(gamma=1.0, total_turns=720)

        for traj_idx in range(50):
            # Decide trajectory length (full 720 or truncated between 10 and 200)
            if traj_idx % 2 == 0:
                T = 720
            else:
                T = np.random.randint(10, 201)

            # Generate realistic or random state sequence
            cash_0 = float(np.random.randint(1000, 5000))
            current_cash = cash_0
            current_opp_cash = float(np.random.randint(1000, 5000))

            obs_sequence = []
            for t in range(T + 1):
                # Simulate cash changes, crops, seeds, shed goods
                if t < T:
                    cash_delta = float(np.random.normal(loc=15.0, scale=30.0))
                    current_cash = max(100.0, current_cash + cash_delta)
                    opp_delta = float(np.random.normal(loc=12.0, scale=25.0))
                    current_opp_cash = max(100.0, current_opp_cash + opp_delta)

                # Add some intermediate assets if not at terminal step of full game
                tiles = []
                shed = {}
                seeds = {}

                if t < T and t < 650:
                    if np.random.rand() > 0.5:
                        tiles.append([{"kind": "PLANT", "crop": "CARROT", "growth": float(t % 72) / 72.0}])
                    if np.random.rand() > 0.5:
                        shed["WHEAT"] = int(np.random.randint(0, 10))
                    if np.random.rand() > 0.5:
                        seeds["CARROT"] = int(np.random.randint(0, 5))
                elif t == T and T == 720:
                    # Clean terminal state at T=720: assets fully liquidated
                    tiles = []
                    shed = {}
                    seeds = {}

                obs = {
                    "step": t,
                    "day": t // 24,
                    "farms": [
                        {"money": current_cash, "tiles": tiles},
                        {"money": current_opp_cash, "tiles": []}
                    ],
                    "private": {"shed": shed, "seeds": seeds}
                }
                obs_sequence.append(obs)

                # Set initial state assets for t=0 clean baseline check
                if t == 0:
                    obs["farms"][0]["tiles"] = []
                    obs["farms"][1]["tiles"] = []
                    obs["private"]["shed"] = {}
                    obs["private"]["seeds"] = {}

            # Test Single-Player Telescoping: sum(r'_t) == Phi(s_T) - Phi(s_0)
            shaped_rewards = pbrs_undiscounted.shape_trajectory(obs_sequence, player=0)
            total_shaped_reward = float(np.sum(shaped_rewards))

            phi_0 = pbrs_undiscounted.compute_potential(obs_sequence[0], player=0)
            phi_T = pbrs_undiscounted.compute_potential(obs_sequence[-1], player=0)
            expected_telescoping = phi_T - phi_0

            self.assertAlmostEqual(
                total_shaped_reward,
                expected_telescoping,
                places=2,
                msg=f"Trajectory {traj_idx} failed single-player telescoping check!"
            )

            if T == 720 and len(obs_sequence[-1]["private"]["shed"]) == 0:
                expected_cash_delta = obs_sequence[-1]["farms"][0]["money"] - obs_sequence[0]["farms"][0]["money"]
                self.assertAlmostEqual(
                    total_shaped_reward,
                    expected_cash_delta,
                    places=2,
                    msg=f"Trajectory {traj_idx} failed terminal cash equality check!"
                )

            # Test Relative Potential Telescoping: sum(r'_t) == Phi_rel(s_T) - Phi_rel(s_0)
            pbrs_rel = MaturityAwarePBRS(gamma=1.0, total_turns=720, relative=True)
            shaped_rel = pbrs_rel.shape_trajectory(obs_sequence, player=0)
            total_shaped_rel = float(np.sum(shaped_rel))

            phi_rel_0 = pbrs_rel.compute_potential(obs_sequence[0], player=0)
            phi_rel_T = pbrs_rel.compute_potential(obs_sequence[-1], player=0)
            expected_rel_telescoping = phi_rel_T - phi_rel_0

            self.assertAlmostEqual(
                total_shaped_rel,
                expected_rel_telescoping,
                places=2,
                msg=f"Trajectory {traj_idx} failed relative telescoping check!"
            )

    def test_gae_processor_trajectory_and_two_hot_targets(self):
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

        # Compute two-hot symlog target distributions
        two_hot = self.processor.compute_two_hot_targets(targets)
        self.assertEqual(two_hot.shape, (T, NUM_BINS))
        
        # Verify distributions sum to 1.0
        prob_sums = two_hot.sum(dim=-1).numpy()
        np.testing.assert_allclose(prob_sums, np.ones(T), rtol=1e-5, atol=1e-5)

        # Verify decoding two-hot distributions matches original scalar targets within bin resolution
        recovered_values = two_hot_to_value(two_hot).numpy()
        np.testing.assert_allclose(recovered_values, targets, rtol=1e-2, atol=2.0)

    def test_process_match_end_to_end(self):
        T = 24
        cash_history = [3000.0 + i * 50.0 for i in range(T + 1)]
        obs_sequence = []
        for t, cash in enumerate(cash_history):
            obs_sequence.append({
                "step": t, "day": t // 24,
                "farms": [{"money": cash, "tiles": []}, {"money": 3000.0, "tiles": []}],
                "private": {"shed": {}, "seeds": {}}
            })

        value_estimates = np.array(cash_history, dtype=np.float32)
        match_data = self.processor.process_match(
            observations=obs_sequence,
            value_estimates=value_estimates,
            player=0
        )

        self.assertIn("shaped_rewards", match_data)
        self.assertIn("advantages", match_data)
        self.assertIn("value_targets", match_data)
        self.assertIn("two_hot_targets", match_data)
        self.assertIn("potentials", match_data)

        self.assertEqual(len(match_data["shaped_rewards"]), T)
        self.assertEqual(len(match_data["advantages"]), T)
        self.assertEqual(len(match_data["value_targets"]), T)
        self.assertEqual(match_data["two_hot_targets"].shape, (T, NUM_BINS))


if __name__ == "__main__":
    unittest.main()
