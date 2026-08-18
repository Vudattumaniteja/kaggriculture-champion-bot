"""
FiLM-Modulated SE-ResNet Neural Backbone, Decoupled Multi-Head Policy & 1001-Bin Symlog Critic.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BINS = 1001
V_MIN = -15.0
V_MAX = 15.0


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(y: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential inverse of symlog: sign(y) * (exp(|y|) - 1)."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


def get_symlog_bin_centers(num_bins: int = NUM_BINS, v_min: float = V_MIN, v_max: float = V_MAX, device=None) -> torch.Tensor:
    """Returns the tensor of bin center coordinates in symlog space."""
    return torch.linspace(v_min, v_max, steps=num_bins, device=device)


def value_to_two_hot(values: torch.Tensor, num_bins: int = NUM_BINS, v_min: float = V_MIN, v_max: float = V_MAX) -> torch.Tensor:
    """
    Transforms scalar dollar cash targets into two-hot categorical distributions over symlog bins.
    """
    device = values.device
    s_vals = symlog(values).clamp(v_min, v_max)
    bin_width = (v_max - v_min) / (num_bins - 1)
    
    # Normalized position [0, num_bins - 1]
    norm_pos = (s_vals - v_min) / bin_width
    low_idx = torch.floor(norm_pos).long().clamp(0, num_bins - 2)
    high_idx = (low_idx + 1).clamp(0, num_bins - 1)
    
    weight_high = norm_pos - low_idx.float()
    weight_low = 1.0 - weight_high
    
    two_hot = torch.zeros(values.shape[0], num_bins, device=device)
    two_hot.scatter_add_(1, low_idx.unsqueeze(1), weight_low.unsqueeze(1))
    two_hot.scatter_add_(1, high_idx.unsqueeze(1), weight_high.unsqueeze(1))
    
    return two_hot


def two_hot_to_value(probs: torch.Tensor, num_bins: int = NUM_BINS, v_min: float = V_MIN, v_max: float = V_MAX) -> torch.Tensor:
    """
    Converts two-hot probability distribution over bins back to expected scalar dollar cash.
    """
    bin_centers = get_symlog_bin_centers(num_bins, v_min, v_max, device=probs.device)
    expected_symlog = (probs * bin_centers).sum(dim=-1)
    return symexp(expected_symlog)


def two_hot_symlog_loss(logits: torch.Tensor, target_values: torch.Tensor, num_bins: int = NUM_BINS, v_min: float = V_MIN, v_max: float = V_MAX) -> torch.Tensor:
    """
    Computes categorical cross-entropy loss between critic logits and two-hot symlog targets.
    """
    targets = value_to_two_hot(target_values, num_bins=num_bins, v_min=v_min, v_max=v_max)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(targets * log_probs).sum(dim=-1).mean()
    return loss


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention module."""
    def __init__(self, channels: int = 64, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class FiLMSEBlock(nn.Module):
    """Residual convolutional block with FiLM affine modulation and SE attention."""
    def __init__(self, channels: int = 64, econ_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        
        # FiLM projection for scale (gamma) and shift (beta)
        self.film_proj = nn.Linear(econ_dim, 2 * channels)

    def forward(self, x: torch.Tensor, z_econ: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        
        # Apply FiLM: out = (1 + gamma) * out + beta
        gamma_beta = self.film_proj(z_econ)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        out = (1.0 + gamma) * out + beta
        
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out += residual
        return F.relu(out, inplace=True)


class ChampionBackbone(nn.Module):
    """
    FiLM-Modulated Convolutional Backbone Trunk.
    Takes 24-channel spatial tensor and 72-dim economic scalar vector.
    Outputs:
    - z_spatial: Shape (B, 64, 10, 10)
    - z_global: Shape (B, 256)
    """
    def __init__(self, spatial_in: int = 24, scalar_in: int = 72, channels: int = 64, econ_dim: int = 128, global_dim: int = 256):
        super().__init__()
        # Economic scalar encoder MLP
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_in, econ_dim),
            nn.LayerNorm(econ_dim),
            nn.ReLU(inplace=True),
            nn.Linear(econ_dim, econ_dim),
            nn.LayerNorm(econ_dim),
            nn.ReLU(inplace=True),
        )

        # Spatial entry conv
        self.in_conv = nn.Sequential(
            nn.Conv2d(spatial_in, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 3 FiLM-modulated SE-ResNet blocks
        self.res1 = FiLMSEBlock(channels, econ_dim)
        self.res2 = FiLMSEBlock(channels, econ_dim)
        self.res3 = FiLMSEBlock(channels, econ_dim)

        # Global fusion head
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(channels + econ_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_spatial: torch.Tensor, x_scalar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_econ = self.scalar_encoder(x_scalar)
        
        feat = self.in_conv(x_spatial)
        feat = self.res1(feat, z_econ)
        feat = self.res2(feat, z_econ)
        z_spatial = self.res3(feat, z_econ)
        
        pooled_spatial = self.spatial_pool(z_spatial).view(x_spatial.shape[0], -1)
        fused = torch.cat([pooled_spatial, z_econ], dim=-1)
        z_global = self.global_proj(fused)
        
        return z_spatial, z_global


class ChampionCritic(nn.Module):
    """
    1001-Bin Categorical Symlog Value Critic & Auxiliary Win-Probability Head.
    """
    def __init__(self, global_dim: int = 256, num_bins: int = NUM_BINS):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_bins)
        )
        self.win_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, z_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        value_logits = self.value_head(z_global)
        win_logit = self.win_head(z_global)
        return value_logits, win_logit


class ChampionPolicyNetwork(nn.Module):
    """
    Decoupled Multi-Head Policy Actor.
    Emits factorized operational action targets.
    """
    def __init__(self, spatial_channels: int = 64, global_dim: int = 256):
        super().__init__()
        # 1. Spatial Crop Allocation Heatmap: 5 crops x 10 x 10
        self.crop_head = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 5, kernel_size=1)
        )

        # 2. Livestock Target Herd Sizes (Goose, Sheep, Cow): 3 dims
        self.livestock_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

        # 3. Workforce Recruitment (0..12 farmhands): 13 discrete logits
        self.workforce_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 13)
        )

        # 4. Land Acquisition (Quadrant expansion): 1 logit
        self.land_head = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        # 5. Autonomous Seed Replenishment (5 crops): 5 logits
        self.seed_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5)
        )

        # 6. Continuous Market Liquidation Fractions (9 commodities): 9 continuous [0, 1]
        self.market_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9),
            nn.Sigmoid()
        )

    def forward(self, z_spatial: torch.Tensor, z_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "crop_heatmaps": self.crop_head(z_spatial),
            "livestock_quotas": self.livestock_head(z_global),
            "workforce_logits": self.workforce_head(z_global),
            "land_expand_logit": self.land_head(z_global),
            "seed_replenish_logits": self.seed_head(z_global),
            "market_fractions": self.market_head(z_global),
        }


class ChampionFullNetwork(nn.Module):
    """
    Unified Champion Network combining Backbone, Critic, and Decoupled Policy Actor.
    """
    def __init__(self):
        super().__init__()
        self.backbone = ChampionBackbone()
        self.critic = ChampionCritic()
        self.policy = ChampionPolicyNetwork()

    def forward(self, x_spatial: torch.Tensor, x_scalar: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_spatial, z_global = self.backbone(x_spatial, x_scalar)
        val_logits, win_logit = self.critic(z_global)
        policy_outputs = self.policy(z_spatial, z_global)
        
        return {
            **policy_outputs,
            "value_logits": val_logits,
            "win_logit": win_logit,
            "z_spatial": z_spatial,
            "z_global": z_global,
        }


def apply_action_masks(logits: torch.Tensor, masks: torch.Tensor, mask_value: float = -1e9) -> torch.Tensor:
    """
    Applies analytical boolean/binary action masks to logits prior to softmax.
    Valid positions (mask == 1) remain untouched; invalid positions (mask == 0) are set to mask_value.
    """
    masks = masks.to(logits.device)
    return torch.where(masks > 0.5, logits, torch.full_like(logits, mask_value))


def build_action_masks(money: float, unlocked_quads: List[str], num_workers: int, shed: Dict[str, int]) -> Dict[str, Any]:
    """
    Computes analytical pre-softmax action masks for workforce hiring and land acquisition.
    """
    unlocked_set = set(unlocked_quads)

    # 1. Workforce Hiring Mask: hiring k total workers requires (k * 20) daily wage reserve
    workforce_mask = torch.zeros(13, dtype=torch.float32)
    for k in range(13):
        # Current workers cannot be fired; additional hires cost wages
        if k >= num_workers and (k * 20.0 <= money or k == num_workers):
            workforce_mask[k] = 1.0
        elif k == num_workers:
            workforce_mask[k] = 1.0

    # 2. Land Expansion Mask
    # Next quadrant cost: NE=$1000, SW=$2000, SE=$4000
    next_quad_cost = None
    if "NE" not in unlocked_set:
        next_quad_cost = 1000.0
    elif "SW" not in unlocked_set:
        next_quad_cost = 2000.0
    elif "SE" not in unlocked_set:
        next_quad_cost = 4000.0

    land_expand_mask = 1.0 if (next_quad_cost is not None and money >= next_quad_cost) else 0.0

    return {
        "workforce_mask": workforce_mask.unsqueeze(0),
        "land_expand_mask": land_expand_mask,
    }
