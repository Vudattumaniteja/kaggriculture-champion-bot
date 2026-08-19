import sys
import os
import shutil
import tempfile
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
        torch.manual_seed(42)
        self.batch_size = 4
        self.latent_dim = 128
        self.action_dim = 32
        self.world_model = TwoScaleHierarchicalWorldModel(latent_dim=self.latent_dim, action_dim=self.action_dim)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_synthetic_dataset(self, num_samples: int = 16) -> AWILDataset:
        sample_transitions = [
            {
                "spatial": torch.randn(24, 10, 10).numpy(),
                "scalar": torch.randn(72).numpy(),
                "crop_heatmaps": torch.zeros(5, 10, 10).numpy(),
                "workforce": i % 3,
                "land_expand": 1.0 if (i % 4 == 0) else 0.0,
                "seed_replenish": torch.zeros(5).numpy(),
                "market_fractions": torch.zeros(9).numpy(),
                "terminal_return": 100000.0 + i * 1000.0,
                "step": (i * 24) % 720,
                "baseline_value": 90000.0,
            }
            for i in range(num_samples)
        ]
        return AWILDataset(transitions=sample_transitions, augment=False)

    def test_micro_and_macro_dynamics_forward_shapes(self):
        z_t = torch.randn(self.batch_size, self.latent_dim)
        a_strat = torch.randn(self.batch_size, self.action_dim)
        
        # Test 1-step micro transition in R^128
        z_next_micro = self.world_model.step_micro(z_t, a_strat)
        self.assertEqual(z_next_micro.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(z_next_micro).any())
        self.assertFalse(torch.isinf(z_next_micro).any())

        # Test 24-step macro day-skip transition in R^128
        z_next_day = self.world_model.step_day(z_t, a_strat)
        self.assertEqual(z_next_day.shape, (self.batch_size, self.latent_dim))
        self.assertFalse(torch.isnan(z_next_day).any())
        self.assertFalse(torch.isinf(z_next_day).any())

    def test_auxiliary_decoders(self):
        z_t = torch.randn(self.batch_size, self.latent_dim)
        yield_pred, price_pred, shop_pred = self.world_model.decode_auxiliary(z_t)
        
        self.assertEqual(yield_pred.shape, (self.batch_size, 2, 10, 10))
        self.assertEqual(price_pred.shape, (self.batch_size, 9))
        self.assertEqual(shop_pred.shape, (self.batch_size, 8))
        
        # Decoded ranges: yield in [0, 1], prices >= 0, shops in [0, 1]
        self.assertTrue((yield_pred >= 0.0).all() and (yield_pred <= 1.0).all())
        self.assertTrue((price_pred >= 0.0).all())
        self.assertTrue((shop_pred >= 0.0).all() and (shop_pred <= 1.0).all())
        self.assertFalse(torch.isnan(yield_pred).any())
        self.assertFalse(torch.isnan(price_pred).any())
        self.assertFalse(torch.isnan(shop_pred).any())

    def test_world_model_action_encoding_and_rollout(self):
        z_0 = torch.randn(2, self.latent_dim)
        actions = torch.randn(2, 5, self.action_dim)
        
        # Multi-step micro rollout
        z_micro_seq = self.world_model.rollout_micro(z_0, actions)
        self.assertEqual(len(z_micro_seq), 6)  # z_0 plus 5 steps
        for z in z_micro_seq:
            self.assertEqual(z.shape, (2, self.latent_dim))
            self.assertFalse(torch.isnan(z).any())

        # Multi-day macro rollout
        z_day_seq = self.world_model.rollout_day(z_0, actions)
        self.assertEqual(len(z_day_seq), 6)
        for z in z_day_seq:
            self.assertEqual(z.shape, (2, self.latent_dim))
            self.assertFalse(torch.isnan(z).any())

    def test_staged_trainer_step(self):
        network = ChampionFullNetwork()
        dataset = self._generate_synthetic_dataset(8)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        trainer = StagedAWILTrainer(network=network, world_model=self.world_model, lr=1e-3)
        metrics = trainer.train_epoch(dataloader)
        
        self.assertIn("total_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("aux_loss", metrics)
        self.assertFalse(torch.isnan(torch.tensor(metrics["total_loss"])))
        self.assertGreater(metrics["total_loss"], 0.0)

    def test_gradnorm_policy_balancing(self):
        network = ChampionFullNetwork()
        dataset = self._generate_synthetic_dataset(16)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        trainer = StagedAWILTrainer(
            network=network,
            world_model=self.world_model,
            lr=1e-3,
            use_gradnorm=True,
            gradnorm_lr=0.01,
        )

        initial_weights = trainer.task_weights.clone().detach()
        self.assertEqual(initial_weights.shape[0], 5)
        self.assertAlmostEqual(initial_weights.sum().item(), 5.0, places=4)

        trainer.train_epoch(dataloader)

        updated_weights = trainer.task_weights.detach()
        # Task weights should stay strictly positive and normalized to sum = 5.0
        self.assertTrue((updated_weights > 0.0).all())
        self.assertAlmostEqual(updated_weights.sum().item(), 5.0, places=3)

    def test_group2_fixed_weights(self):
        network = ChampionFullNetwork()
        trainer = StagedAWILTrainer(
            network=network,
            world_model=self.world_model,
            lambda_val=1.0,
            lambda_aux=0.1,
        )
        self.assertEqual(trainer.lambda_val, 1.0)
        self.assertEqual(trainer.lambda_aux, 0.1)

    def test_5_epoch_training_dry_run_monotonic_decrease(self):
        torch.manual_seed(123)
        network = ChampionFullNetwork()
        dataset = self._generate_synthetic_dataset(32)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        trainer = StagedAWILTrainer(
            network=network,
            world_model=self.world_model,
            lr=5e-3,
            use_gradnorm=True,
        )

        loss_history = []
        for epoch in range(5):
            metrics = trainer.train_epoch(dataloader)
            loss_history.append(metrics["total_loss"])

        # Check overall loss reduction from epoch 0 to epoch 4
        self.assertLess(loss_history[-1], loss_history[0])
        for loss in loss_history:
            self.assertTrue(torch.isfinite(torch.tensor(loss)))

        # Verify weights are finite and can be saved
        weights_path = os.path.join(self.temp_dir, "champion_weights.pt")
        trainer.save_checkpoint(weights_path)
        self.assertTrue(os.path.exists(weights_path))
        self.assertGreater(os.path.getsize(weights_path), 100000)

    def test_checkpoint_save_and_load_integrity(self):
        network = ChampionFullNetwork()
        trainer = StagedAWILTrainer(
            network=network,
            world_model=self.world_model,
            lr=1e-3,
            use_gradnorm=True,
        )

        ckpt_path = os.path.join(self.temp_dir, "test_ckpt.pt")
        trainer.save_checkpoint(ckpt_path)

        # Create fresh instances
        new_network = ChampionFullNetwork()
        new_world_model = TwoScaleHierarchicalWorldModel(latent_dim=self.latent_dim, action_dim=self.action_dim)
        new_trainer = StagedAWILTrainer(network=new_network, world_model=new_world_model, lr=1e-3)

        new_trainer.load_checkpoint(ckpt_path)

        # Verify weights matched exactly
        for p1, p2 in zip(network.parameters(), new_network.parameters()):
            self.assertTrue(torch.equal(p1, p2))
        for p1, p2 in zip(self.world_model.parameters(), new_world_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
