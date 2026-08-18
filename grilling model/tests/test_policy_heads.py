import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import torch
import torch.nn.functional as F
import numpy as np

from src.network import (
    ChampionBackbone,
    ChampionCritic,
    ChampionPolicyNetwork,
    ChampionFullNetwork,
    build_action_masks,
    apply_action_masks,
    compute_masked_probabilities,
    sample_masked_categorical,
    CROPS,
    PRODUCTS,
    SEED_COSTS,
    QUADRANT_COSTS,
    SHED_TILES,
    MAX_SEED_BUFFER,
)


class TestPolicyHeads(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.spatial = torch.randn(self.batch_size, 24, 10, 10)
        self.scalar = torch.randn(self.batch_size, 72)
        self.full_net = ChampionFullNetwork()
        self.policy_net = ChampionPolicyNetwork()

    def test_forward_output_shapes(self):
        outputs = self.full_net(self.spatial, self.scalar)
        
        # Verify all 6 factorized policy heads
        self.assertIn("crop_heatmaps", outputs)
        self.assertEqual(outputs["crop_heatmaps"].shape, (self.batch_size, 5, 10, 10))
        
        self.assertIn("livestock_quotas", outputs)
        self.assertEqual(outputs["livestock_quotas"].shape, (self.batch_size, 3))
        
        self.assertIn("workforce_logits", outputs)
        self.assertEqual(outputs["workforce_logits"].shape, (self.batch_size, 13))
        
        self.assertIn("land_expand_logit", outputs)
        self.assertEqual(outputs["land_expand_logit"].shape, (self.batch_size, 1))
        
        self.assertIn("seed_replenish_logits", outputs)
        self.assertEqual(outputs["seed_replenish_logits"].shape, (self.batch_size, 5))
        
        self.assertIn("market_fractions", outputs)
        self.assertEqual(outputs["market_fractions"].shape, (self.batch_size, 9))
        
        # Critic & Representations
        self.assertIn("value_logits", outputs)
        self.assertEqual(outputs["value_logits"].shape, (self.batch_size, 1001))
        
        self.assertIn("win_logit", outputs)
        self.assertEqual(outputs["win_logit"].shape, (self.batch_size, 1))
        
        self.assertIn("z_spatial", outputs)
        self.assertEqual(outputs["z_spatial"].shape, (self.batch_size, 64, 10, 10))
        
        self.assertIn("z_global", outputs)
        self.assertEqual(outputs["z_global"].shape, (self.batch_size, 256))

    def test_spatial_crop_masking_unowned_quadrants_and_shed(self):
        # 1. Starting with only NW quadrant unlocked
        masks_nw = build_action_masks(
            money=3000.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={}
        )
        crop_mask = masks_nw["crop_spatial_mask"]
        self.assertEqual(crop_mask.shape, (1, 5, 10, 10))

        # Check NW quadrant valid tiles
        for r in range(5):
            for c in range(5):
                if (r, c) in SHED_TILES:
                    self.assertEqual(crop_mask[0, :, r, c].sum().item(), 0.0, f"Shed tile ({r}, {c}) should be masked")
                else:
                    self.assertEqual(crop_mask[0, :, r, c].mean().item(), 1.0, f"NW tile ({r}, {c}) should be valid")

        # Check NE, SW, SE quadrants are masked
        for r in range(10):
            for c in range(10):
                if r >= 5 or c >= 5:
                    self.assertEqual(crop_mask[0, :, r, c].sum().item(), 0.0, f"Locked tile ({r}, {c}) should be masked")

        # Apply mask to crop logits
        raw_crop_logits = torch.randn(1, 5, 10, 10)
        masked_crop_logits = apply_action_masks(raw_crop_logits, crop_mask)

        # Spatial softmax over grid for each crop
        flat_masked = masked_crop_logits.view(1, 5, 100)
        spatial_probs = F.softmax(flat_masked, dim=-1).view(1, 5, 10, 10)

        # Locked quadrants and shed tiles must have 0.0 probability
        for r in range(10):
            for c in range(10):
                if (r >= 5 or c >= 5) or ((r, c) in SHED_TILES):
                    for crop_idx in range(5):
                        self.assertEqual(spatial_probs[0, crop_idx, r, c].item(), 0.0)

    def test_workforce_wage_affordability_masking(self):
        # Case A: Cash = 45, current workers = 1. Max affordable = 2 workers ($40)
        masks_a = build_action_masks(
            money=45.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={}
        )
        wf_logits = torch.randn(1, 13)
        masked_a = apply_action_masks(wf_logits, masks_a["workforce_mask"])
        probs_a = F.softmax(masked_a, dim=-1)

        # k=1 and k=2 are valid
        self.assertGreater(probs_a[0, 1].item(), 0.0)
        self.assertGreater(probs_a[0, 2].item(), 0.0)
        # k >= 3 must have exactly 0 probability
        for k in range(3, 13):
            self.assertEqual(probs_a[0, k].item(), 0.0)

        # Case B: Cash = 10 (less than 1 worker wage $20), current workers = 1
        masks_b = build_action_masks(
            money=10.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={}
        )
        masked_b = apply_action_masks(wf_logits, masks_b["workforce_mask"])
        probs_b = F.softmax(masked_b, dim=-1)
        # k=1 is retained as fallback
        self.assertAlmostEqual(probs_b[0, 1].item(), 1.0, places=5)
        for k in range(13):
            if k != 1:
                self.assertEqual(probs_b[0, k].item(), 0.0)

        # Case C: Cash = 0, current workers = 0
        masks_c = build_action_masks(
            money=0.0,
            unlocked_quads=["NW"],
            num_workers=0,
            shed={}
        )
        masked_c = apply_action_masks(wf_logits, masks_c["workforce_mask"])
        probs_c = F.softmax(masked_c, dim=-1)
        self.assertAlmostEqual(probs_c[0, 0].item(), 1.0, places=5)

    def test_quadrant_unlock_cost_masking(self):
        # 1. Next is NE ($1000)
        masks_ne_poor = build_action_masks(money=999.0, unlocked_quads=["NW"], num_workers=1, shed={})
        self.assertEqual(masks_ne_poor["land_expand_mask"], 0.0)
        self.assertEqual(masks_ne_poor["land_mask_tensor"].item(), 0.0)

        masks_ne_rich = build_action_masks(money=1000.0, unlocked_quads=["NW"], num_workers=1, shed={})
        self.assertEqual(masks_ne_rich["land_expand_mask"], 1.0)
        self.assertEqual(masks_ne_rich["land_mask_tensor"].item(), 1.0)

        # 2. Next is SW ($2000)
        masks_sw_poor = build_action_masks(money=1999.0, unlocked_quads=["NW", "NE"], num_workers=1, shed={})
        self.assertEqual(masks_sw_poor["land_expand_mask"], 0.0)

        masks_sw_rich = build_action_masks(money=2000.0, unlocked_quads=["NW", "NE"], num_workers=1, shed={})
        self.assertEqual(masks_sw_rich["land_expand_mask"], 1.0)

        # 3. Next is SE ($4000)
        masks_se_poor = build_action_masks(money=3999.0, unlocked_quads=["NW", "NE", "SW"], num_workers=1, shed={})
        self.assertEqual(masks_se_poor["land_expand_mask"], 0.0)

        masks_se_rich = build_action_masks(money=4000.0, unlocked_quads=["NW", "NE", "SW"], num_workers=1, shed={})
        self.assertEqual(masks_se_rich["land_expand_mask"], 1.0)

        # 4. All unlocked -> Cannot buy land
        masks_all = build_action_masks(money=100000.0, unlocked_quads=["NW", "NE", "SW", "SE"], num_workers=1, shed={})
        self.assertEqual(masks_all["land_expand_mask"], 0.0)

    def test_seed_replenishment_inventory_and_cash_bounds(self):
        # Costs: WHEAT: $10, CARROT: $20, TOMATO: $50, STRAWBERRY: $100, MELON: $80
        # Case A: Cash = $25 -> Can buy WHEAT ($10) or CARROT ($20), cannot buy TOMATO ($50), MELON ($80), STRAWBERRY ($100)
        masks_a = build_action_masks(
            money=25.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={},
            seeds={"WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0}
        )
        seed_mask_a = masks_a["seed_replenish_mask"]
        self.assertEqual(seed_mask_a[0, 0].item(), 1.0)  # WHEAT
        self.assertEqual(seed_mask_a[0, 1].item(), 1.0)  # CARROT
        self.assertEqual(seed_mask_a[0, 2].item(), 0.0)  # TOMATO
        self.assertEqual(seed_mask_a[0, 3].item(), 0.0)  # STRAWBERRY
        self.assertEqual(seed_mask_a[0, 4].item(), 0.0)  # MELON

        seed_logits = torch.randn(1, 5)
        masked_seed_logits = apply_action_masks(seed_logits, seed_mask_a)
        probs_seed = F.softmax(masked_seed_logits, dim=-1)
        self.assertGreater(probs_seed[0, 0].item(), 0.0)
        self.assertGreater(probs_seed[0, 1].item(), 0.0)
        self.assertEqual(probs_seed[0, 2].item(), 0.0)
        self.assertEqual(probs_seed[0, 3].item(), 0.0)
        self.assertEqual(probs_seed[0, 4].item(), 0.0)

        # Case B: Inventory buffer reached (e.g. 20 seeds)
        masks_b = build_action_masks(
            money=5000.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={},
            seeds={"WHEAT": 20, "CARROT": 5, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0},
            max_seed_buffer=20
        )
        seed_mask_b = masks_b["seed_replenish_mask"]
        self.assertEqual(seed_mask_b[0, 0].item(), 0.0, "WHEAT reached max buffer 20, should be masked")
        self.assertEqual(seed_mask_b[0, 1].item(), 1.0, "CARROT under buffer, should be legal")

        # Case C: Shed full (100 items) -> Cannot buy any seeds
        full_shed = {prod: 12 for prod in PRODUCTS}  # 9 * 12 = 108 items
        masks_c = build_action_masks(
            money=5000.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed=full_shed,
            seeds={}
        )
        self.assertEqual(masks_c["seed_replenish_mask"].sum().item(), 0.0)

    def test_continuous_market_liquidation_fractions(self):
        outputs = self.full_net(self.spatial, self.scalar)
        market_fractions = outputs["market_fractions"]
        self.assertEqual(market_fractions.shape, (self.batch_size, 9))
        
        # Verify continuous range in [0, 1]
        self.assertTrue((market_fractions >= 0.0).all())
        self.assertTrue((market_fractions <= 1.0).all())

        # Test market mask with empty shed vs stocked shed
        shed = {"WHEAT": 10, "MILK": 5}  # Only WHEAT (idx 0) and MILK (idx 6)
        masks = build_action_masks(money=1000.0, unlocked_quads=["NW"], num_workers=1, shed=shed)
        m_mask = masks["market_mask"]
        self.assertEqual(m_mask[0, 0].item(), 1.0)
        self.assertEqual(m_mask[0, 6].item(), 1.0)
        self.assertEqual(m_mask[0, 1].item(), 0.0)  # CARROT empty

        masked_fractions = market_fractions[0:1] * m_mask
        self.assertGreater(masked_fractions[0, 0].item(), 0.0)
        self.assertGreater(masked_fractions[0, 6].item(), 0.0)
        self.assertEqual(masked_fractions[0, 1].item(), 0.0)

    def test_50_constrained_action_masking_scenarios(self):
        """
        Action mask tester verifying that illegal actions receive strictly zero probability mass
        across 50 diverse constrained financial and game state scenarios.
        """
        scenarios = []
        cash_levels = [0.0, 5.0, 15.0, 20.0, 40.0, 50.0, 80.0, 100.0, 500.0, 999.0, 1000.0, 1999.0, 2000.0, 3999.0, 4000.0, 10000.0]
        quadrant_configs = [
            ["NW"],
            ["NW", "NE"],
            ["NW", "NE", "SW"],
            ["NW", "NE", "SW", "SE"],
        ]
        worker_counts = [0, 1, 2, 4, 8, 12]

        # Generate 50+ deterministic permutations
        scenario_idx = 0
        for money in cash_levels:
            for q_cfg in quadrant_configs:
                w_count = worker_counts[scenario_idx % len(worker_counts)]
                seed_inv = {crop: (scenario_idx * 3) % 25 for crop in CROPS}
                shed_inv = {prod: (scenario_idx * 5) % 30 for prod in PRODUCTS}
                if scenario_idx % 7 == 0:
                    shed_inv = {prod: 15 for prod in PRODUCTS}  # Overflow shed (>100)

                scenarios.append({
                    "money": money,
                    "quads": q_cfg,
                    "workers": w_count,
                    "shed": shed_inv,
                    "seeds": seed_inv,
                })
                scenario_idx += 1
                if len(scenarios) >= 50:
                    break
            if len(scenarios) >= 50:
                break

        self.assertGreaterEqual(len(scenarios), 50, "Must test at least 50 constrained scenarios")

        for idx, sc in enumerate(scenarios):
            money = sc["money"]
            quads = sc["quads"]
            workers = sc["workers"]
            shed = sc["shed"]
            seeds = sc["seeds"]

            masks = build_action_masks(
                money=money,
                unlocked_quads=quads,
                num_workers=workers,
                shed=shed,
                seeds=seeds,
                max_seed_buffer=20
            )

            # 1. Workforce Mask & Softmax
            raw_wf_logits = torch.randn(1, 13)
            masked_wf_logits = apply_action_masks(raw_wf_logits, masks["workforce_mask"])
            wf_probs = F.softmax(masked_wf_logits, dim=-1)

            # Assert strictly zero prob on masked positions
            wf_mask_tensor = masks["workforce_mask"]
            for k in range(13):
                if wf_mask_tensor[0, k].item() == 0.0:
                    self.assertEqual(
                        wf_probs[0, k].item(),
                        0.0,
                        f"Scenario {idx} (cash={money}, workers={workers}): workforce k={k} must have 0.0 prob"
                    )
                else:
                    self.assertGreater(
                        wf_probs[0, k].item(),
                        0.0,
                        f"Scenario {idx}: legal workforce k={k} must have >0 prob"
                    )
            self.assertAlmostEqual(wf_probs.sum().item(), 1.0, places=5)

            # 2. Crop Spatial Mask & Softmax
            crop_mask = masks["crop_spatial_mask"]
            raw_crop_logits = torch.randn(1, 5, 10, 10)
            masked_crop_logits = apply_action_masks(raw_crop_logits, crop_mask)
            flat_crop_probs = F.softmax(masked_crop_logits.view(1, 5, 100), dim=-1).view(1, 5, 10, 10)

            for r in range(10):
                for c in range(10):
                    is_valid = crop_mask[0, 0, r, c].item() > 0.5
                    for crop_i in range(5):
                        if not is_valid:
                            self.assertEqual(
                                flat_crop_probs[0, crop_i, r, c].item(),
                                0.0,
                                f"Scenario {idx}: unowned/shed tile ({r}, {c}) must receive 0 prob"
                            )
                        else:
                            self.assertGreater(
                                flat_crop_probs[0, crop_i, r, c].item(),
                                0.0,
                                f"Scenario {idx}: owned tile ({r}, {c}) must receive >0 prob"
                            )

            # 3. Autonomous Seed Replenishment Mask & Softmax
            raw_seed_logits = torch.randn(1, 5)
            seed_mask = masks["seed_replenish_mask"]
            masked_seed_logits = apply_action_masks(raw_seed_logits, seed_mask)
            seed_probs = F.softmax(masked_seed_logits, dim=-1)

            if seed_mask.sum().item() > 0:
                for crop_i in range(5):
                    if seed_mask[0, crop_i].item() == 0.0:
                        self.assertEqual(
                            seed_probs[0, crop_i].item(),
                            0.0,
                            f"Scenario {idx}: unaffordable/buffered seed {CROPS[crop_i]} must have 0 prob"
                        )
                    else:
                        self.assertGreater(seed_probs[0, crop_i].item(), 0.0)
                self.assertAlmostEqual(seed_probs.sum().item(), 1.0, places=5)

            # 4. Land Expansion Mask
            land_mask = masks["land_expand_mask"]
            self.assertIn(land_mask, [0.0, 1.0])

            # No NaNs anywhere
            self.assertFalse(torch.isnan(wf_probs).any())
            self.assertFalse(torch.isnan(flat_crop_probs).any())
            self.assertFalse(torch.isnan(seed_probs).any())

    def test_forward_with_explicit_masks(self):
        masks = build_action_masks(
            money=25.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={"WHEAT": 5},
            seeds={"WHEAT": 2}
        )

        outputs = self.full_net(
            self.spatial,
            self.scalar,
            crop_mask=masks["crop_spatial_mask"],
            workforce_mask=masks["workforce_mask"],
            land_mask=masks["land_mask_tensor"],
            seed_mask=masks["seed_replenish_mask"],
            market_mask=masks["market_mask"],
        )

        # Workforce > 2 must be clamped to -1e9
        wf_logits = outputs["workforce_logits"]
        for k in range(3, 13):
            self.assertLess(wf_logits[:, k].max().item(), -1e8)

        # Market fractions for unsheltered goods should be zeroed
        market_fractions = outputs["market_fractions"]
        self.assertEqual(market_fractions[:, 1].sum().item(), 0.0)  # CARROT
        self.assertGreater(market_fractions[:, 0].sum().item(), 0.0)  # WHEAT

    def test_gradient_backward_propagation_all_heads(self):
        spatial = torch.randn(2, 24, 10, 10, requires_grad=True)
        scalar = torch.randn(2, 72, requires_grad=True)

        outputs = self.full_net(spatial, scalar)
        
        # Loss combining all heads
        l_crop = outputs["crop_heatmaps"].sum()
        l_live = outputs["livestock_quotas"].sum()
        l_wf = outputs["workforce_logits"].sum()
        l_land = outputs["land_expand_logit"].sum()
        l_seed = outputs["seed_replenish_logits"].sum()
        l_mkt = outputs["market_fractions"].sum()
        l_val = outputs["value_logits"].sum()
        l_win = outputs["win_logit"].sum()

        total_loss = l_crop + l_live + l_wf + l_land + l_seed + l_mkt + l_val + l_win
        total_loss.backward()

        self.assertIsNotNone(spatial.grad)
        self.assertIsNotNone(scalar.grad)
        self.assertFalse(torch.isnan(spatial.grad).any())
        self.assertFalse(torch.isnan(scalar.grad).any())


if __name__ == "__main__":
    unittest.main()
