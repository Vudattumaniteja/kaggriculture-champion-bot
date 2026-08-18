import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.world_model import (
    TwoScaleHierarchicalWorldModel,
    MicroDynamics,
    MacroDaySkipDynamics,
    AuxiliaryForesightDecoders,
)
from src.trainer import StagedAWILTrainer
from src.network import ChampionFullNetwork
from src.awil_dataset import AWILDataset


class TestWorldModelTraining(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.latent_dim = 128
        self.action_dim = 32
        self.world_model = TwoScaleHierarchicalWorldModel(latent_dim=self.latent_dim, action_dim=self.action_dim)

    def test_micro_and_macro_dynamics_forward_shapes(self):
        z_t = torch.randn(self.batch_size, self.latent_dim)
        a_strat = torch.randn(self.batch_size, self.action_dim)
        
        # Test 1-step micro transition
        z_next_micro = self.world_model.step_micro(z_t, a_strat)
        self.assertEqual(z_next_micro.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(z_next_micro).any())

        # Test 24-step macro day-skip transition
        z_next_day = self.world_model.step_day(z_t, a_strat)
        self.assertEqual(z_next_day.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(z_next_day).any())

    def test_auxiliary_decoders(self):
        z_t = torch.randn(self.batch_size, self.latent_dim)
        yield_pred, price_pred, shop_pred = self.world_model.decode_auxiliary(z_t)
        
        self.assertEqual(yield_pred.shape, (self.batch_size, 2, 10, 10))
        self.assertEqual(price_pred.shape, (self.batch_size, 9))
        self.assertEqual(shop_pred.shape, (self.batch_size, 8))
        self.assertFalse(torch.isnan(yield_pred).any())
        self.assertFalse(torch.isnan(price_pred).any())
        self.assertFalse(torch.isnan(shop_pred).any())

    def test_staged_trainer_step(self):
        # Create synthetic dataset and model
        network = ChampionFullNetwork()
        sample_transitions = [
            {
                "spatial": torch.randn(24, 10, 10).numpy(),
                "scalar": torch.randn(72).numpy(),
                "crop_heatmaps": torch.zeros(5, 10, 10).numpy(),
                "workforce": 1,
                "land_expand": 0.0,
                "seed_replenish": torch.zeros(5).numpy(),
                "market_fractions": torch.zeros(9).numpy(),
                "terminal_return": 100000.0,
                "step": 24,
                "baseline_value": 90000.0
            }
            for _ in range(8)
        ]
        dataset = AWILDataset(transitions=sample_transitions, augment=False)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        trainer = StagedAWILTrainer(network=network, world_model=self.world_model, lr=1e-3)
        metrics = trainer.train_epoch(dataloader)
        
        self.assertIn("total_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertFalse(torch.isnan(torch.tensor(metrics["total_loss"])))
        self.assertGreater(metrics["total_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
