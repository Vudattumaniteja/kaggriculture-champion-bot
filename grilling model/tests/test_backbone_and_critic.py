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
    ChampionCritic,
    symlog,
    symexp,
    value_to_two_hot,
    two_hot_to_value,
    two_hot_symlog_loss,
    NUM_BINS,
    V_MIN,
    V_MAX,
)


class TestBackboneAndCritic(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.spatial = torch.randn(self.batch_size, 24, 10, 10)
        self.scalar = torch.randn(self.batch_size, 72)
        self.backbone = ChampionBackbone()
        self.critic = ChampionCritic()

    def test_backbone_forward_shapes(self):
        z_spatial, z_global = self.backbone(self.spatial, self.scalar)
        self.assertEqual(z_spatial.shape, (self.batch_size, 64, 10, 10))
        self.assertEqual(z_global.shape, (self.batch_size, 256))
        self.assertFalse(torch.isnan(z_spatial).any())
        self.assertFalse(torch.isnan(z_global).any())

    def test_critic_forward_shapes(self):
        _, z_global = self.backbone(self.spatial, self.scalar)
        value_logits, win_logit = self.critic(z_global)
        self.assertEqual(value_logits.shape, (self.batch_size, NUM_BINS))
        self.assertEqual(win_logit.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(value_logits).any())
        self.assertFalse(torch.isnan(win_logit).any())

    def test_symlog_and_symexp_bijection(self):
        test_vals = torch.tensor([-100000.0, -50.0, 0.0, 3000.0, 250000.0, 3260000.0])
        s_vals = symlog(test_vals)
        recovered = symexp(s_vals)
        torch.testing.assert_close(recovered, test_vals, rtol=1e-4, atol=1e-4)

    def test_two_hot_encoding_and_decoding(self):
        target_cash = torch.tensor([-50000.0, 0.0, 3000.0, 150000.0])
        two_hot = value_to_two_hot(target_cash, num_bins=NUM_BINS, v_min=V_MIN, v_max=V_MAX)
        self.assertEqual(two_hot.shape, (4, NUM_BINS))
        # Each row should sum to 1.0
        row_sums = two_hot.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

        # Decode back
        decoded_cash = two_hot_to_value(two_hot, num_bins=NUM_BINS, v_min=V_MIN, v_max=V_MAX)
        torch.testing.assert_close(decoded_cash, target_cash, rtol=1e-2, atol=10.0)

    def test_critic_loss_backward(self):
        _, z_global = self.backbone(self.spatial, self.scalar)
        value_logits, win_logit = self.critic(z_global)
        
        target_cash = torch.tensor([1000.0, 5000.0, 20000.0, 80000.0])
        target_win = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
        
        loss = two_hot_symlog_loss(value_logits, target_cash, num_bins=NUM_BINS, v_min=V_MIN, v_max=V_MAX)
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(win_logit, target_win)
        total_loss = loss + bce_loss
        
        total_loss.backward()
        self.assertFalse(torch.isnan(total_loss))
        self.assertGreater(total_loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
