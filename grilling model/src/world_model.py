from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroDynamics(nn.Module):
    """
    Models 1-step within-day latent state transitions: z_{t+1} = g_micro(z_t, a_strat).
    """
    def __init__(self, latent_dim: int = 128, action_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z_t: torch.Tensor, a_strat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([z_t, a_strat], dim=-1)
        delta = self.net(fused)
        return z_t + delta


class MacroDaySkipDynamics(nn.Module):
    """
    Models 24-step day boundaries in a single latent step: z_{d+1, 0} = g_day(z_{d, 23}, a_day).
    """
    def __init__(self, latent_dim: int = 128, action_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, z_d_23: torch.Tensor, a_day: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([z_d_23, a_day], dim=-1)
        delta = self.net(fused)
        return z_d_23 + delta


class AuxiliaryForesightDecoders(nn.Module):
    """
    Physical grounding decoders predicting next-morning crop yields, commodity spot prices, and shop demands.
    """
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        # 1. Next-morning Crop Yields: (2, 10, 10) (yield units + maturation ratio)
        self.yield_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * 10 * 10),
            nn.Sigmoid()
        )

        # 2. Next-day Commodity Spot Prices: 9 dims
        self.price_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9),
            nn.ReLU()
        )

        # 3. Next-day Town Shop Demands: 8 dims
        self.shop_decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = z.shape[0]
        yields = self.yield_decoder(z).view(b, 2, 10, 10)
        prices = self.price_decoder(z)
        shops = self.shop_decoder(z)
        return yields, prices, shops


class TwoScaleHierarchicalWorldModel(nn.Module):
    """
    Unified Hierarchical World Model managing micro, macro dynamics, and auxiliary physical decoders.
    """
    def __init__(self, latent_dim: int = 128, action_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.micro = MicroDynamics(latent_dim, action_dim)
        self.macro = MacroDaySkipDynamics(latent_dim, action_dim)
        self.auxiliary = AuxiliaryForesightDecoders(latent_dim)

    def step_micro(self, z_t: torch.Tensor, a_strat: torch.Tensor) -> torch.Tensor:
        a_encoded = self.encode_action(a_strat)
        return self.micro(z_t, a_encoded)

    def step_day(self, z_d_23: torch.Tensor, a_day: torch.Tensor) -> torch.Tensor:
        a_encoded = self.encode_action(a_day)
        return self.macro(z_d_23, a_encoded)

    def decode_auxiliary(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.auxiliary(z)

    def encode_action(self, action: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Converts either a raw action tensor or a policy action dictionary into (B, action_dim) tensor.
        """
        if isinstance(action, torch.Tensor):
            if action.dim() == 1:
                action = action.unsqueeze(0)
            if action.shape[-1] == self.action_dim:
                return action
            elif action.shape[-1] < self.action_dim:
                pad = torch.zeros(*action.shape[:-1], self.action_dim - action.shape[-1], device=action.device, dtype=action.dtype)
                return torch.cat([action, pad], dim=-1)
            else:
                return action[..., :self.action_dim]

        elif isinstance(action, dict):
            # Extract components from policy dictionary
            parts = []
            if "crop_heatmaps" in action:
                crop = action["crop_heatmaps"]
                if crop.dim() == 4:
                    crop_summary = crop.mean(dim=(-1, -2))  # (B, 5)
                else:
                    crop_summary = crop.view(crop.shape[0], -1)[:, :5]
                parts.append(crop_summary)

            if "workforce_logits" in action:
                wf = action["workforce_logits"]
                wf_prob = F.softmax(wf, dim=-1)
                parts.append(wf_prob[:, :4])  # top workforce features (B, 4)

            if "land_expand_logit" in action:
                land = torch.sigmoid(action["land_expand_logit"].view(-1, 1))
                parts.append(land)

            if "seed_replenish_logits" in action:
                seed = action["seed_replenish_logits"]
                parts.append(seed[:, :5])

            if "market_fractions" in action:
                mkt = action["market_fractions"]
                parts.append(mkt[:, :9])

            if parts:
                fused = torch.cat(parts, dim=-1)
                return self.encode_action(fused)
            else:
                return torch.zeros(1, self.action_dim)

        raise TypeError(f"Unsupported action type: {type(action)}")

    def rollout_micro(self, z_0: torch.Tensor, action_sequence: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs multi-step within-day micro rollout.
        action_sequence: Shape (B, T, action_dim)
        Returns: List of (T+1) latent tensors [z_0, z_1, ..., z_T]
        """
        latents = [z_0]
        cur_z = z_0
        T = action_sequence.shape[1]
        for t in range(T):
            cur_a = action_sequence[:, t, :]
            cur_z = self.step_micro(cur_z, cur_a)
            latents.append(cur_z)
        return latents

    def rollout_day(self, z_0: torch.Tensor, action_sequence: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs multi-day macro day-skip rollout.
        action_sequence: Shape (B, D, action_dim)
        Returns: List of (D+1) latent tensors [z_0, z_1, ..., z_D]
        """
        latents = [z_0]
        cur_z = z_0
        D = action_sequence.shape[1]
        for d in range(D):
            cur_a = action_sequence[:, d, :]
            cur_z = self.step_day(cur_z, cur_a)
            latents.append(cur_z)
        return latents
