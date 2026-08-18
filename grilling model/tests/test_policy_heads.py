import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import torch
import numpy as np

from src.network import (
    ChampionBackbone,
    ChampionPolicyNetwork,
    ChampionFullNetwork,
    build_action_masks,
    apply_action_masks,
)


class TestPolicyHeads(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.spatial = torch.randn(self.batch_size, 24, 10, 10)
        self.scalar = torch.randn(self.batch_size, 72)
        self.full_net = ChampionFullNetwork()

    def test_forward_output_shapes(self):
        outputs = self.full_net(self.spatial, self.scalar)
        
        # Verify all heads
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
        
        self.assertIn("value_logits", outputs)
        self.assertEqual(outputs["value_logits"].shape, (self.batch_size, 1001))
        
        self.assertIn("win_logit", outputs)
        self.assertEqual(outputs["win_logit"].shape, (self.batch_size, 1))

    def test_pre_softmax_action_masking(self):
        # Create synthetic mask for workforce: cash allows max 2 workers ($40)
        money = 45.0
        masks = build_action_masks(
            money=money,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={"WHEAT": 5}
        )
        
        workforce_logits = torch.randn(1, 13)
        masked_logits = apply_action_masks(workforce_logits, masks["workforce_mask"])
        
        # Workers > 2 ($60+) must have -1e9 logit
        for k in range(3, 13):
            self.assertLess(masked_logits[0, k].item(), -1e8)
            
        # Softmax on masked logits should give ~0 prob to masked indices
        probs = torch.softmax(masked_logits, dim=-1)
        for k in range(3, 13):
            self.assertAlmostEqual(probs[0, k].item(), 0.0, places=5)

    def test_land_expansion_mask(self):
        # NW only unlocked, NE costs 1000. With 500 money, buying land is illegal
        masks = build_action_masks(
            money=500.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={}
        )
        self.assertEqual(masks["land_expand_mask"], 0.0)

        # With 1500 money, buying land is legal
        masks_rich = build_action_masks(
            money=1500.0,
            unlocked_quads=["NW"],
            num_workers=1,
            shed={}
        )
        self.assertEqual(masks_rich["land_expand_mask"], 1.0)


if __name__ == "__main__":
    unittest.main()
