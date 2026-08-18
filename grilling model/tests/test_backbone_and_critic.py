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
    SEBlock,
    FiLMSEBlock,
    ChampionBackbone,
    ChampionCritic,
    symlog,
    symexp,
    get_symlog_bin_centers,
    value_to_two_hot,
    two_hot_to_value,
    two_hot_symlog_loss,
    NUM_BINS,
    V_MIN,
    V_MAX,
)


class TestBackboneAndCritic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.spatial = torch.randn(self.batch_size, 24, 10, 10)
        self.scalar = torch.randn(self.batch_size, 72)
        self.backbone = ChampionBackbone()
        self.critic = ChampionCritic()

    def test_se_block_forward(self):
        se = SEBlock(channels=64, reduction=4)
        x = torch.randn(4, 64, 10, 10)
        out = se(x)
        self.assertEqual(out.shape, (4, 64, 10, 10))
        self.assertFalse(torch.isnan(out).any())

    def test_film_se_block_forward(self):
        film_se = FiLMSEBlock(channels=64, econ_dim=128)
        x = torch.randn(4, 64, 10, 10)
        z_econ = torch.randn(4, 128)
        out = film_se(x, z_econ)
        self.assertEqual(out.shape, (4, 64, 10, 10))
        self.assertFalse(torch.isnan(out).any())

    def test_backbone_forward_shapes(self):
        z_spatial, z_global = self.backbone(self.spatial, self.scalar)
        self.assertEqual(z_spatial.shape, (self.batch_size, 64, 10, 10))
        self.assertEqual(z_global.shape, (self.batch_size, 256))
        self.assertFalse(torch.isnan(z_spatial).any())
        self.assertFalse(torch.isnan(z_global).any())

    def test_film_modulation_conditioning(self):
        """Verify that different economic scalar inputs modulate identical spatial inputs differently."""
        x_spatial = torch.randn(2, 24, 10, 10)
        # Duplicate the same spatial input for both items
        x_spatial[1] = x_spatial[0].clone()
        
        # Provide two distinct economic scalar features
        x_scalar = torch.randn(2, 72)
        x_scalar[1] = x_scalar[0] + 5.0
        
        z_spatial, z_global = self.backbone(x_spatial, x_scalar)
        
        # Despite identical spatial input, FiLM conditioning must result in distinct feature maps
        spatial_diff = (z_spatial[0] - z_spatial[1]).abs().sum().item()
        global_diff = (z_global[0] - z_global[1]).abs().sum().item()
        
        self.assertGreater(spatial_diff, 1e-3, "FiLM modulation should alter spatial representations")
        self.assertGreater(global_diff, 1e-3, "FiLM modulation should alter global representations")

    def test_critic_forward_shapes(self):
        _, z_global = self.backbone(self.spatial, self.scalar)
        value_logits, win_logit = self.critic(z_global)
        self.assertEqual(value_logits.shape, (self.batch_size, NUM_BINS))
        self.assertEqual(win_logit.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(value_logits).any())
        self.assertFalse(torch.isnan(win_logit).any())

    def test_critic_prediction_helpers(self):
        _, z_global = self.backbone(self.spatial, self.scalar)
        pred_value = self.critic.predict_value(z_global)
        pred_win_prob = self.critic.predict_win_prob(z_global)
        
        self.assertEqual(pred_value.shape, (self.batch_size,))
        self.assertEqual(pred_win_prob.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(pred_value).any())
        self.assertFalse(torch.isnan(pred_win_prob).any())
        self.assertTrue((pred_win_prob >= 0.0).all() and (pred_win_prob <= 1.0).all())

    def test_symlog_and_symexp_bijection(self):
        test_vals = torch.tensor([-3260000.0, -100000.0, -50.0, 0.0, 3000.0, 250000.0, 3260000.0])
        s_vals = symlog(test_vals)
        recovered = symexp(s_vals)
        torch.testing.assert_close(recovered, test_vals, rtol=1e-4, atol=1e-4)

    def test_symlog_bounds_and_range(self):
        """Check that V_MIN/V_MAX = +/-15.0 corresponds to +/-$3.26M."""
        max_val = symexp(torch.tensor(15.0)).item()
        min_val = symexp(torch.tensor(-15.0)).item()
        self.assertAlmostEqual(max_val, float(np.expm1(15.0)), delta=1.0)
        self.assertAlmostEqual(min_val, float(-np.expm1(15.0)), delta=1.0)
        self.assertGreater(max_val, 3.26e6)
        self.assertLess(min_val, -3.26e6)

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

    def test_two_hot_edge_cases_and_clamping(self):
        # Extreme cash values beyond +/- $3.26M
        target_cash = torch.tensor([-1e8, 1e8, 0.0])
        two_hot = value_to_two_hot(target_cash, num_bins=NUM_BINS, v_min=V_MIN, v_max=V_MAX)
        self.assertEqual(two_hot.shape, (3, NUM_BINS))
        self.assertFalse(torch.isnan(two_hot).any())
        
        # Row sums must be 1.0
        row_sums = two_hot.sum(dim=-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))

        # First row should have mass at bin 0
        self.assertAlmostEqual(two_hot[0, 0].item(), 1.0, places=4)
        # Second row should have mass at bin NUM_BINS - 1
        self.assertAlmostEqual(two_hot[1, NUM_BINS - 1].item(), 1.0, places=4)

    def test_forward_and_backward_batch_32_synthetic_inputs(self):
        """Batch of 32 synthetic inputs forward & backward gradient propagation passes without NaN loss."""
        b_size = 32
        synthetic_spatial = torch.randn(b_size, 24, 10, 10, requires_grad=True)
        synthetic_scalar = torch.randn(b_size, 72, requires_grad=True)
        
        backbone = ChampionBackbone()
        critic = ChampionCritic()
        
        z_spatial, z_global = backbone(synthetic_spatial, synthetic_scalar)
        value_logits, win_logit = critic(z_global)
        
        # Synthetic target dollar cash spanning diverse positive, zero, and negative values
        target_cash = torch.linspace(-500000.0, 2500000.0, steps=b_size)
        target_win = torch.bernoulli(torch.full((b_size, 1), 0.5))
        
        val_loss = two_hot_symlog_loss(value_logits, target_cash, num_bins=NUM_BINS, v_min=V_MIN, v_max=V_MAX)
        win_loss = F.binary_cross_entropy_with_logits(win_logit, target_win)
        total_loss = val_loss + 0.5 * win_loss
        
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))
        self.assertGreater(total_loss.item(), 0.0)
        
        total_loss.backward()
        
        # Check that gradients exist and are finite for all model parameters
        for name, param in backbone.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient in backbone parameter: {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in backbone parameter: {name}")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient in backbone parameter: {name}")
            
        for name, param in critic.named_parameters():
            self.assertIsNotNone(param.grad, f"Missing gradient in critic parameter: {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in critic parameter: {name}")
            self.assertFalse(torch.isinf(param.grad).any(), f"Inf gradient in critic parameter: {name}")


if __name__ == "__main__":
    unittest.main()
