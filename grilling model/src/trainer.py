"""
Two-Group Staged AWIL and Overnight Reinforcement Learning Trainer.
"""

from typing import Any, Dict, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.network import ChampionFullNetwork, two_hot_symlog_loss
from src.world_model import TwoScaleHierarchicalWorldModel


class StagedAWILTrainer:
    """
    Two-Group Staged Trainer for AWIL Imitation and World Model Pre-training.
    - Group 1: Strategic Policy Heads (balanced dynamically / weighted by sample_weight)
    - Group 2: Critic Value Head (locked lambda_val = 1.0) and Auxiliary Foresight Decoders (locked lambda_aux = 0.1)
    """
    def __init__(
        self,
        network: ChampionFullNetwork,
        world_model: Optional[TwoScaleHierarchicalWorldModel] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        lambda_val: float = 1.0,
        lambda_aux: float = 0.1,
    ):
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.world_model = world_model.to(self.device) if world_model is not None else None
        self.lambda_val = lambda_val
        self.lambda_aux = lambda_aux

        params = list(self.network.parameters())
        if self.world_model is not None:
            params += list(self.world_model.parameters())

        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.network.train()
        if self.world_model is not None:
            self.world_model.train()

        total_loss_acc = 0.0
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        batches = 0

        for batch in dataloader:
            x_spatial = batch["x_spatial"].to(self.device)
            x_scalar = batch["x_scalar"].to(self.device)
            target_crop = batch["target_crop_heatmaps"].to(self.device)
            target_workforce = batch["target_workforce"].to(self.device)
            target_land = batch["target_land_expand"].to(self.device)
            target_seed = batch["target_seed_replenish"].to(self.device)
            target_market = batch["target_market_fractions"].to(self.device)
            target_value = batch["target_value"].to(self.device)
            weights = batch["sample_weight"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.network(x_spatial, x_scalar)

            # Group 1: Strategic Policy Losses
            crop_loss = F.mse_loss(outputs["crop_heatmaps"], target_crop, reduction="none").mean(dim=[1, 2, 3])
            wf_loss = F.cross_entropy(outputs["workforce_logits"], target_workforce, reduction="none")
            land_loss = F.binary_cross_entropy_with_logits(outputs["land_expand_logit"].squeeze(-1), target_land, reduction="none")
            seed_loss = F.mse_loss(outputs["seed_replenish_logits"], target_seed, reduction="none").mean(dim=-1)
            market_loss = F.mse_loss(outputs["market_fractions"], target_market, reduction="none").mean(dim=-1)

            policy_per_sample = crop_loss + wf_loss + land_loss + seed_loss + market_loss
            policy_loss = (policy_per_sample * weights).mean()

            # Group 2: Critic Value Loss
            val_loss = two_hot_symlog_loss(outputs["value_logits"], target_value)

            # World Model Auxiliary Losses
            aux_loss = torch.tensor(0.0, device=self.device)
            if self.world_model is not None:
                z_econ = outputs["z_global"][:, :self.world_model.latent_dim]
                yield_pred, price_pred, shop_pred = self.world_model.decode_auxiliary(z_econ)
                aux_loss = yield_pred.mean() * 0.01 + price_pred.mean() * 0.01 + shop_pred.mean() * 0.01

            total_loss = policy_loss + self.lambda_val * val_loss + self.lambda_aux * aux_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss_acc += total_loss.item()
            policy_loss_acc += policy_loss.item()
            value_loss_acc += val_loss.item()
            batches += 1

        return {
            "total_loss": total_loss_acc / max(batches, 1),
            "policy_loss": policy_loss_acc / max(batches, 1),
            "value_loss": value_loss_acc / max(batches, 1),
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "world_model_state_dict": self.world_model.state_dict() if self.world_model is not None else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        if self.world_model is not None and checkpoint.get("world_model_state_dict") is not None:
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
