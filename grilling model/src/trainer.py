"""
Two-Group Staged AWIL and Overnight Reinforcement Learning Trainer.
"""

from typing import Any, Dict, List, Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.network import (
    ChampionFullNetwork,
    two_hot_symlog_loss,
    get_cosine_entropy_coeff,
)
from src.world_model import TwoScaleHierarchicalWorldModel


class StagedAWILTrainer:
    """
    Two-Group Staged Trainer for AWIL Imitation and World Model Pre-training.
    - Group 1: Strategic Policy Heads (balanced dynamically via GradNorm and sample weights)
    - Group 2: Critic Value Head (locked lambda_val = 1.0) and Auxiliary Foresight Decoders (locked lambda_aux = 0.1)
    - Policy Entropy Regularization: Cosine annealed from 0.05 -> 0.002
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
        use_gradnorm: bool = True,
        gradnorm_alpha: float = 0.12,
        gradnorm_lr: float = 1e-3,
        total_steps: int = 100000,
        entropy_start: float = 0.05,
        entropy_end: float = 0.002,
    ):
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.world_model = world_model.to(self.device) if world_model is not None else None
        self.lambda_val = lambda_val
        self.lambda_aux = lambda_aux
        self.use_gradnorm = use_gradnorm
        self.gradnorm_alpha = gradnorm_alpha
        self.gradnorm_lr = gradnorm_lr
        self.total_steps = total_steps
        self.entropy_start = entropy_start
        self.entropy_end = entropy_end
        self.global_step = 0

        # Group 1 Task weights for 5 policy heads: Crop, Workforce, Land, Seed, Market
        self.task_weights = nn.Parameter(torch.ones(5, device=self.device, dtype=torch.float32))
        self.weights_optimizer = torch.optim.Adam([self.task_weights], lr=self.gradnorm_lr)
        self.initial_task_losses: Optional[List[float]] = None

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
        aux_loss_acc = 0.0
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

            # Group 1: Strategic Policy Losses (5 Heads)
            crop_loss_sample = F.mse_loss(outputs["crop_heatmaps"], target_crop, reduction="none").mean(dim=[1, 2, 3])
            wf_loss_sample = F.cross_entropy(outputs["workforce_logits"], target_workforce, reduction="none")
            land_loss_sample = F.binary_cross_entropy_with_logits(outputs["land_expand_logit"].squeeze(-1), target_land, reduction="none")
            seed_loss_sample = F.mse_loss(outputs["seed_replenish_logits"], target_seed, reduction="none").mean(dim=-1)
            market_loss_sample = F.mse_loss(outputs["market_fractions"], target_market, reduction="none").mean(dim=-1)

            L_crop = (crop_loss_sample * weights).mean()
            L_wf = (wf_loss_sample * weights).mean()
            L_land = (land_loss_sample * weights).mean()
            L_seed = (seed_loss_sample * weights).mean()
            L_market = (market_loss_sample * weights).mean()
            task_losses = [L_crop, L_wf, L_land, L_seed, L_market]

            # Group 2: Critic Value Loss (locked lambda_val)
            val_loss = two_hot_symlog_loss(outputs["value_logits"], target_value)

            # Group 2: Auxiliary Foresight Decoders Loss (locked lambda_aux)
            aux_loss = torch.tensor(0.0, device=self.device)
            if self.world_model is not None:
                z_econ = outputs["z_global"][:, :self.world_model.latent_dim]
                yield_pred, price_pred, shop_pred = self.world_model.decode_auxiliary(z_econ)

                target_y = batch.get("target_yield")
                if target_y is None:
                    target_y = x_spatial[:, 5:7, :, :]
                else:
                    target_y = target_y.to(self.device)

                target_p = batch.get("target_price")
                if target_p is None:
                    target_p = x_scalar[:, 11:20]
                else:
                    target_p = target_p.to(self.device)

                target_s = batch.get("target_shop")
                if target_s is None:
                    target_s = x_scalar[:, 38:46]
                else:
                    target_s = target_s.to(self.device)

                loss_y = F.mse_loss(yield_pred, target_y)
                loss_p = F.mse_loss(price_pred, target_p)
                loss_s = F.mse_loss(shop_pred, target_s)
                aux_loss = loss_y + loss_p + loss_s

            # GradNorm Policy Balancing for Group 1
            if self.use_gradnorm:
                if self.initial_task_losses is None:
                    self.initial_task_losses = [max(tl.detach().item(), 1e-4) for tl in task_losses]

                shared_w = self.network.backbone.in_conv[0].weight
                G_list = []
                for i in range(5):
                    grads = torch.autograd.grad(
                        self.task_weights[i] * task_losses[i],
                        shared_w,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    g_i = grads[0]
                    if g_i is not None:
                        G_list.append(torch.norm(g_i, 2))
                    else:
                        G_list.append(torch.zeros(1, device=self.device))

                G_stack = torch.stack(G_list)
                G_avg = G_stack.mean().detach()

                loss_ratios = torch.tensor([task_losses[i].item() / max(self.initial_task_losses[i], 1e-6) for i in range(5)], device=self.device)
                inverse_train_rates = loss_ratios / max(loss_ratios.mean().item(), 1e-6)
                target_G = G_avg * (inverse_train_rates ** self.gradnorm_alpha)

                gradnorm_loss = sum(F.l1_loss(G_list[i], target_G[i].detach()) for i in range(5))

                self.weights_optimizer.zero_grad()
                gradnorm_loss.backward(retain_graph=True)
                self.weights_optimizer.step()

                with torch.no_grad():
                    self.task_weights.data = torch.clamp(self.task_weights.data, min=0.01)
                    self.task_weights.data = self.task_weights.data * (5.0 / self.task_weights.data.sum())

            # Policy Entropy Regularization (Cosine Annealed: 0.05 -> 0.002)
            wf_probs = F.softmax(outputs["workforce_logits"], dim=-1)
            wf_log_probs = F.log_softmax(outputs["workforce_logits"], dim=-1)
            policy_entropy = -(wf_probs * wf_log_probs).sum(dim=-1).mean()
            
            ent_coeff = get_cosine_entropy_coeff(
                step=self.global_step,
                total_steps=self.total_steps,
                start_coeff=self.entropy_start,
                end_coeff=self.entropy_end,
            )
            entropy_loss = -ent_coeff * policy_entropy

            # Aggregate losses
            policy_loss = sum(self.task_weights[i].detach() * task_losses[i] for i in range(5))
            total_loss = policy_loss + self.lambda_val * val_loss + self.lambda_aux * aux_loss + entropy_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
            if self.world_model is not None:
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=5.0)
            self.optimizer.step()
            self.global_step += 1

            total_loss_acc += total_loss.item()
            policy_loss_acc += policy_loss.item()
            value_loss_acc += val_loss.item()
            aux_loss_acc += aux_loss.item()
            batches += 1

        return {
            "total_loss": total_loss_acc / max(batches, 1),
            "policy_loss": policy_loss_acc / max(batches, 1),
            "value_loss": value_loss_acc / max(batches, 1),
            "aux_loss": aux_loss_acc / max(batches, 1),
            "entropy": policy_entropy.item() if batches > 0 else 0.0,
            "entropy_coeff": ent_coeff if batches > 0 else self.entropy_start,
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "world_model_state_dict": self.world_model.state_dict() if self.world_model is not None else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "task_weights": self.task_weights.detach().cpu(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        if self.world_model is not None and checkpoint.get("world_model_state_dict") is not None:
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "task_weights" in checkpoint and checkpoint["task_weights"] is not None:
            self.task_weights.data.copy_(checkpoint["task_weights"].to(self.device))

