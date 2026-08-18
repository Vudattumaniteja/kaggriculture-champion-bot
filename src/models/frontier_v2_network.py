"""
Frontier V2 Grandmaster Policy-Value, Dynamics & Auxiliary Neural Network Architecture.

Components:
1. Spatial SE-ResNet Trunk (11x10x10 farm grid with Squeeze-and-Excitation Conv layers).
2. Economic MLP Trunk (32-dim global features -> 64-dim embedding).
3. 128-dim Unified Latent State z_t.
4. Policy Head pi_theta(a|s) over 10 macro-actions.
5. 601-Bin Two-Hot Symlog Categorical Value Head in [-15, +15].
   Symlog: h(z) = sign(z) * ln(|z| + 1).
6. SSL World Dynamics Head s_hat_{t+4} for latent imagination.
7. KataGo Auxiliary Heads:
   - Spatial Tile Yield Head Y_hat(r, c) (expected crop/animal yield per grid cell).
   - Town Shop Price Forecaster P_hat(t+24) (expected prices 24 turns ahead).
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import NUM_MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS

# Value discretization parameters
NUM_BINS = 601
V_MIN = -15.0
V_MAX = 15.0
BIN_WIDTH = (V_MAX - V_MIN) / (NUM_BINS - 1)  # 30.0 / 600 = 0.05
BIN_CENTERS = torch.linspace(V_MIN, V_MAX, NUM_BINS)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Two-hot symlog transformation h(x) = sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse symlog transformation h^{-1}(y) = sign(y) * (exp(|y|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def scalar_to_two_hot(scalar: torch.Tensor, v_min: float = V_MIN, v_max: float = V_MAX, num_bins: int = NUM_BINS) -> torch.Tensor:
    """
    Converts a continuous scalar target (e.g. raw reward or cash) into a 601-bin two-hot symlog distribution.
    Args:
        scalar: (B, 1) or (B,) tensor
    Returns:
        probs: (B, NUM_BINS) two-hot target distribution
    """
    if scalar.dim() == 1:
        scalar = scalar.unsqueeze(-1)
    
    # 1. Symlog transform and clamp
    transformed = torch.clamp(symlog(scalar), v_min, v_max)  # (B, 1)
    
    # 2. Compute bin coordinates
    bin_width = (v_max - v_min) / (num_bins - 1)
    coords = (transformed - v_min) / bin_width  # in [0, num_bins - 1]
    
    low_indices = torch.clamp(torch.floor(coords).long(), 0, num_bins - 2)
    high_indices = low_indices + 1
    
    high_weights = coords - low_indices.float()
    low_weights = 1.0 - high_weights
    
    two_hot = torch.zeros(scalar.shape[0], num_bins, device=scalar.device, dtype=torch.float32)
    two_hot.scatter_add_(1, low_indices, low_weights)
    two_hot.scatter_add_(1, high_indices, high_weights)
    
    return two_hot


def categorical_to_scalar(logits_or_probs: torch.Tensor, is_logits: bool = True, v_min: float = V_MIN, v_max: float = V_MAX, num_bins: int = NUM_BINS) -> torch.Tensor:
    """
    Decodes 601-bin categorical logits/probs back to continuous scalar value via expected symlog and symexp.
    Args:
        logits_or_probs: (B, NUM_BINS)
        is_logits: bool
    Returns:
        expected_scalar: (B, 1)
    """
    if is_logits:
        probs = F.softmax(logits_or_probs, dim=-1)
    else:
        probs = logits_or_probs
    
    bin_centers = torch.linspace(v_min, v_max, num_bins, device=probs.device, dtype=probs.dtype)
    expected_symlog = torch.sum(probs * bin_centers, dim=-1, keepdim=True)  # (B, 1)
    return symexp(expected_symlog)


class SqueezeExcitation(nn.Module):
    """Channel-wise Squeeze-and-Excitation (SE) block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


class SEResBlock(nn.Module):
    """Residual Block with Squeeze-and-Excitation."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels, reduction=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual, inplace=True)
        return out


class FrontierV2Network(nn.Module):
    """
    Frontier V2 Grandmaster Neural Network.
    Integrates Spatial SE-ResNet, Economic MLP, 601-Bin Two-Hot Symlog Value Head,
    SSL World Dynamics Head, and KataGo Auxiliary Heads.
    """
    def __init__(
        self,
        in_channels: int = SPATIAL_CHANNELS,  # 11
        scalar_dim: int = SCALAR_DIM,         # 32
        num_actions: int = NUM_MACRO_ACTIONS, # 10
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.scalar_dim = scalar_dim
        self.hidden_dim = hidden_dim

        # 1. Spatial SE-ResNet Trunk (11x10x10 farm grid)
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([SEResBlock(64) for _ in range(num_res_blocks)])
        self.spatial_fc = nn.Sequential(
            nn.Linear(64 * 10 * 10, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 2. Economic Scalar MLP Trunk (32 global features)
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # 3. Unified Latent Fusion Trunk -> z_t (128-dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Head 1: Policy Head pi_theta(a|s) -> logits over 10 macro actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions),
        )

        # Head 2: 601-Bin Two-Hot Symlog Categorical Value Head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_BINS),
        )

        # Head 3: SSL World Dynamics Head -> predicts s_hat_{t+4} (32 scalars)
        self.ssl_dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim + num_actions, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, scalar_dim),
        )

        # Head 4: KataGo Auxiliary Tile Yield Head Y_hat(r, c) (10x10 expected yields)
        self.tile_yield_head = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # Yields are non-negative
        )

        # Head 5: KataGo Auxiliary Town Shop Price Forecaster P_hat(t+24) (9 product prices)
        self.price_forecaster_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9),
        )

    def extract_latent_from_spatial(self, x_spatial: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Fast latent fusion using precomputed spatial embedding."""
        x_scalar = self.scalar_fc(scalars)
        return self.fusion(torch.cat([x_spatial, x_scalar], dim=1))

    def extract_features(self, grid: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts spatial feature maps and the 128-dim unified latent representation z_t.
        Returns:
            latent_z: (B, 128)
            x_spatial: (B, 128)
            spatial_map: (B, 64, 10, 10)
        """
        x_grid = self.spatial_stem(grid)
        for res_block in self.res_blocks:
            x_grid = res_block(x_grid)
        
        x_spatial = self.spatial_fc(x_grid.view(x_grid.size(0), -1))
        x_scalar = self.scalar_fc(scalars)
        
        latent_z = self.fusion(torch.cat([x_spatial, x_scalar], dim=1))
        return latent_z, x_spatial, x_grid

    def forward(
        self,
        grid: torch.Tensor,
        scalars: torch.Tensor,
        action_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Full multi-task forward pass.
        """
        latent_z, _, spatial_map = self.extract_features(grid, scalars)

        policy_logits = self.policy_head(latent_z)
        value_logits = self.value_head(latent_z)

        pred_future_scalars = None
        if action_indices is not None:
            action_one_hot = F.one_hot(action_indices, num_classes=self.num_actions).float()
            dynamics_input = torch.cat([latent_z, action_one_hot], dim=1)
            pred_future_scalars = self.ssl_dynamics_head(dynamics_input)

        pred_tile_yield = self.tile_yield_head(spatial_map)
        pred_prices = self.price_forecaster_head(latent_z)

        return policy_logits, value_logits, pred_future_scalars, pred_tile_yield, pred_prices

    @torch.no_grad()
    def predict(
        self, grid: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast CPU inference mode returning action probabilities and expected scalar value."""
        self.eval()
        latent_z, _, _ = self.extract_features(grid, scalars)
        policy_logits = self.policy_head(latent_z)
        value_logits = self.value_head(latent_z)

        probs = F.softmax(policy_logits, dim=-1)
        expected_val = categorical_to_scalar(value_logits, is_logits=True)
        return probs, expected_val
