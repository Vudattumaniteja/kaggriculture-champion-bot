"""
Two-Scale Hierarchical World Model with Micro/Macro Dynamics and Auxiliary Decoders.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroDynamics(nn.Module):
    """
    Models 1-step within-day latent state transitions: z_{t+1} = g_micro(z_t, a_strat).
    """
    def __init__(self, latent_dim: int = 128, action_dim: int = 32):
        super().__init__()
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
        return self.micro(z_t, a_strat)

    def step_day(self, z_d_23: torch.Tensor, a_day: torch.Tensor) -> torch.Tensor:
        return self.macro(z_d_23, a_day)

    def decode_auxiliary(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.auxiliary(z)
