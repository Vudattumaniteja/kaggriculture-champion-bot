"""
FiLM-Modulated SE-ResNet Neural Backbone, Decoupled Multi-Head Policy & 1001-Bin Symlog Critic.
Includes Pre-Softmax Analytical Action Masking for all Factorized Decision Spaces.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BINS = 1001
V_MIN = -15.0
V_MAX = 15.0

CROPS: List[str] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS: List[str] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS: List[str] = ["GOOSE", "COW", "SHEEP"]

SEED_COSTS: Dict[str, float] = {
    "WHEAT": 10.0,
    "CARROT": 20.0,
    "TOMATO": 50.0,
    "STRAWBERRY": 100.0,
    "MELON": 80.0,
}

QUADRANT_COSTS: Dict[str, float] = {
    "NE": 1000.0,
    "SW": 2000.0,
    "SE": 4000.0,
}

SHED_TILES: List[Tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]
MAX_SEED_BUFFER: int = 20


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
    flat_values = values.reshape(-1)
    s_vals = symlog(flat_values).clamp(v_min, v_max)
    bin_width = (v_max - v_min) / (num_bins - 1)
    
    norm_pos = (s_vals - v_min) / bin_width
    low_idx = torch.floor(norm_pos).long().clamp(0, num_bins - 2)
    high_idx = (low_idx + 1).clamp(0, num_bins - 1)
    
    weight_high = norm_pos - low_idx.float()
    weight_low = 1.0 - weight_high
    
    two_hot = torch.zeros(flat_values.shape[0], num_bins, device=device, dtype=torch.float32)
    two_hot.scatter_add_(1, low_idx.unsqueeze(1), weight_low.unsqueeze(1))
    two_hot.scatter_add_(1, high_idx.unsqueeze(1), weight_high.unsqueeze(1))
    
    if values.dim() > 1 and values.shape[-1] != 1:
        return two_hot.view(*values.shape, num_bins)
    return two_hot.view(values.shape[0], num_bins)


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

    def predict_value(self, z_global: torch.Tensor) -> torch.Tensor:
        """Predicts expected scalar dollar cash value."""
        value_logits, _ = self.forward(z_global)
        probs = F.softmax(value_logits, dim=-1)
        return two_hot_to_value(probs)

    def predict_win_prob(self, z_global: torch.Tensor) -> torch.Tensor:
        """Predicts win probability in [0, 1]."""
        _, win_logit = self.forward(z_global)
        return torch.sigmoid(win_logit)


class ChampionPolicyNetwork(nn.Module):
    """
    Decoupled Multi-Head Policy Actor.
    Emits factorized operational action targets:
    1. Spatial crop heatmaps (5 x 10 x 10) conditioned on owned quadrants
    2. Livestock target quotas (3 dims: GOOSE, COW, SHEEP)
    3. Workforce recruitment logits (13 dims: 0..12 farmhands)
    4. Land acquisition logit (1 dim: unlock quadrant)
    5. Autonomous seed replenishment logits (5 dims: WHEAT..MELON)
    6. Continuous market liquidation fractions (9 dims in [0, 1])
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

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_global: torch.Tensor,
        crop_mask: Optional[torch.Tensor] = None,
        workforce_mask: Optional[torch.Tensor] = None,
        land_mask: Optional[torch.Tensor] = None,
        seed_mask: Optional[torch.Tensor] = None,
        market_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        crop_logits = self.crop_head(z_spatial)
        livestock_quotas = self.livestock_head(z_global)
        workforce_logits = self.workforce_head(z_global)
        land_logit = self.land_head(z_global)
        seed_logits = self.seed_head(z_global)
        market_fractions = self.market_head(z_global)

        if crop_mask is not None:
            crop_logits = apply_action_masks(crop_logits, crop_mask)
        if workforce_mask is not None:
            workforce_logits = apply_action_masks(workforce_logits, workforce_mask)
        if land_mask is not None:
            land_logit = apply_action_masks(land_logit, land_mask)
        if seed_mask is not None:
            seed_logits = apply_action_masks(seed_logits, seed_mask)
        if market_mask is not None:
            market_fractions = market_fractions * market_mask.to(market_fractions.device)

        return {
            "crop_heatmaps": crop_logits,
            "livestock_quotas": livestock_quotas,
            "workforce_logits": workforce_logits,
            "land_expand_logit": land_logit,
            "seed_replenish_logits": seed_logits,
            "market_fractions": market_fractions,
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

    def forward(
        self,
        x_spatial: torch.Tensor,
        x_scalar: torch.Tensor,
        crop_mask: Optional[torch.Tensor] = None,
        workforce_mask: Optional[torch.Tensor] = None,
        land_mask: Optional[torch.Tensor] = None,
        seed_mask: Optional[torch.Tensor] = None,
        market_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_spatial, z_global = self.backbone(x_spatial, x_scalar)
        val_logits, win_logit = self.critic(z_global)
        policy_outputs = self.policy(
            z_spatial,
            z_global,
            crop_mask=crop_mask,
            workforce_mask=workforce_mask,
            land_mask=land_mask,
            seed_mask=seed_mask,
            market_mask=market_mask,
        )
        
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
    Valid positions (mask > 0.5) remain untouched; invalid positions (mask <= 0.5) are set to mask_value.
    """
    if isinstance(masks, (int, float, bool)):
        masks = torch.tensor(masks, device=logits.device, dtype=torch.float32)
    else:
        masks = masks.to(logits.device)
    return torch.where(masks > 0.5, logits, torch.full_like(logits, mask_value))


def compute_masked_probabilities(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, mask_value: float = -1e9) -> torch.Tensor:
    """
    Applies analytical pre-softmax action masks and computes normalized probability distribution.
    Guarantees that masked actions receive strictly 0.0 probability mass.
    """
    masked_logits = apply_action_masks(logits, mask, mask_value=mask_value)
    probs = F.softmax(masked_logits, dim=dim)
    # Ensure numerical precision clamping for masked values
    if isinstance(mask, (int, float, bool)):
        mask = torch.tensor(mask, device=logits.device, dtype=torch.float32)
    else:
        mask = mask.to(logits.device)
    return torch.where(mask > 0.5, probs, torch.zeros_like(probs))


def sample_masked_categorical(logits: torch.Tensor, mask: torch.Tensor, mask_value: float = -1e9) -> torch.Tensor:
    """
    Samples action index from masked categorical distribution.
    """
    probs = compute_masked_probabilities(logits, mask, dim=-1, mask_value=mask_value)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def build_action_masks(
    money: float,
    unlocked_quads: List[str],
    num_workers: int,
    shed: Optional[Dict[str, int]] = None,
    seeds: Optional[Dict[str, int]] = None,
    max_seed_buffer: int = MAX_SEED_BUFFER,
) -> Dict[str, Any]:
    """
    Computes comprehensive analytical pre-softmax action masks for all factorized policy heads:
    1. Spatial Crop Mask (1, 5, 10, 10): 1.0 for owned quadrant tiles (excluding shed), 0.0 elsewhere.
    2. Workforce Hiring Mask (1, 13): 1.0 if k >= num_workers and k * 20 <= money (or k == num_workers fallback).
    3. Land Expansion Mask (1, 1) / float: 1.0 if money >= next_quadrant_cost, 0.0 otherwise.
    4. Autonomous Seed Replenishment Mask (1, 5): 1.0 if money >= seed_cost and seed_inv < max_buffer, 0.0 otherwise.
    5. Market Liquidation Mask (1, 9): 1.0 if commodity quantity in shed > 0, 0.0 otherwise.
    """
    shed = dict(shed or {})
    seeds = dict(seeds or {})
    unlocked_set = set(unlocked_quads)

    # 1. Spatial Crop Mask (5 crops x 10 rows x 10 cols)
    crop_mask = torch.zeros(1, 5, 10, 10, dtype=torch.float32)
    for r in range(10):
        for c in range(10):
            quad = "NW" if r < 5 and c < 5 else ("NE" if r < 5 and c >= 5 else ("SW" if r >= 5 and c < 5 else "SE"))
            is_shed = (r, c) in SHED_TILES
            if quad in unlocked_set and not is_shed:
                crop_mask[0, :, r, c] = 1.0

    # 2. Workforce Hiring Mask (13 discrete logits: 0..12)
    workforce_mask = torch.zeros(1, 13, dtype=torch.float32)
    has_valid_hire = False
    for k in range(13):
        # Cannot fire existing workers; additional hires require wage reserve
        if k >= num_workers and (k * 20.0 <= money or k == num_workers):
            workforce_mask[0, k] = 1.0
            has_valid_hire = True
        elif k == num_workers:
            workforce_mask[0, k] = 1.0
            has_valid_hire = True

    if not has_valid_hire and 0 <= num_workers < 13:
        workforce_mask[0, num_workers] = 1.0

    # 3. Land Expansion Mask
    next_quad_cost = None
    if "NE" not in unlocked_set:
        next_quad_cost = QUADRANT_COSTS["NE"]
    elif "SW" not in unlocked_set:
        next_quad_cost = QUADRANT_COSTS["SW"]
    elif "SE" not in unlocked_set:
        next_quad_cost = QUADRANT_COSTS["SE"]

    land_expand_val = 1.0 if (next_quad_cost is not None and money >= next_quad_cost) else 0.0
    land_expand_mask = torch.tensor([[land_expand_val]], dtype=torch.float32)

    # 4. Autonomous Seed Replenishment Mask (5 crops)
    seed_mask = torch.zeros(1, 5, dtype=torch.float32)
    total_shed_items = sum(shed.values())
    for i, crop in enumerate(CROPS):
        cost = SEED_COSTS.get(crop, 20.0)
        cur_seeds = seeds.get(crop, 0)
        if money >= cost and cur_seeds < max_seed_buffer and total_shed_items < 100:
            seed_mask[0, i] = 1.0

    # 5. Market Liquidation Mask (9 commodities)
    market_mask = torch.zeros(1, 9, dtype=torch.float32)
    for i, prod in enumerate(PRODUCTS):
        if shed.get(prod, 0) > 0:
            market_mask[0, i] = 1.0

    return {
        "crop_spatial_mask": crop_mask,
        "workforce_mask": workforce_mask,
        "land_expand_mask": land_expand_val,
        "land_mask_tensor": land_expand_mask,
        "seed_replenish_mask": seed_mask,
        "market_mask": market_mask,
    }


def add_mask_safe_dirichlet_noise(
    logits: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
    epsilon: float = 0.25
) -> torch.Tensor:
    """
    Applies Dirichlet root exploration noise strictly over legal action masks.
    pi_noisy(a) = (1 - eps) * pi(a) + eps * Dir(alpha) on legal support.
    Masked actions (mask <= 0.5) receive strictly 0.0 probability.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        squeeze_needed = True
    else:
        squeeze_needed = False

    if isinstance(mask, (int, float, bool)):
        mask = torch.tensor([[mask]], device=logits.device, dtype=torch.float32)
    elif mask.dim() == 1:
        mask = mask.unsqueeze(0).to(logits.device)
    else:
        mask = mask.to(logits.device)

    batch_size = logits.shape[0]
    out_probs = torch.zeros_like(logits)

    for b in range(batch_size):
        b_logits = logits[b]
        b_mask = mask[b] if mask.shape[0] > b else mask[0]
        valid_indices = (b_mask > 0.5).nonzero(as_tuple=True)[0]
        
        if len(valid_indices) == 0:
            pass
        elif len(valid_indices) == 1:
            out_probs[b, valid_indices[0]] = 1.0
        else:
            sub_logits = b_logits[valid_indices]
            sub_probs = F.softmax(sub_logits, dim=-1)
            
            dirichlet = torch.distributions.Dirichlet(torch.full_like(sub_logits, alpha))
            noise = dirichlet.sample()
            
            noisy_sub_probs = (1.0 - epsilon) * sub_probs + epsilon * noise
            noisy_sub_probs = noisy_sub_probs / noisy_sub_probs.sum()
            out_probs[b, valid_indices] = noisy_sub_probs

    if squeeze_needed:
        return out_probs.squeeze(0)
    return out_probs


def get_cosine_entropy_coeff(
    step: int,
    total_steps: int = 100000,
    start_coeff: float = 0.05,
    end_coeff: float = 0.002
) -> float:
    """
    Computes cosine-annealed policy entropy loss weight: 0.05 -> 0.002.
    """
    fraction = min(1.0, max(0.0, float(step) / max(1, total_steps)))
    return end_coeff + 0.5 * (start_coeff - end_coeff) * (1.0 + math.cos(math.pi * fraction))


def get_gumbel_temperature(step: int) -> float:
    """
    Two-stage Gumbel evaluation temperature schedule:
    tau = 1.0 for t < 48 (first 2 days exploration)
    tau = 0.0 for t >= 48 (greedy deterministic execution)
    """
    return 1.0 if step < 48 else 0.0


def sample_gumbel_action(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    mask_value: float = -1e9
) -> int:
    """
    Samples discrete action with Gumbel noise and action masking.
    If temperature <= 1e-6: returns deterministic argmax over valid actions.
    If temperature > 0: adds Gumbel noise and returns argmax.
    """
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    if mask.dim() > 1:
        mask = mask.squeeze(0)

    masked_logits = torch.where(mask.to(logits.device) > 0.5, logits, torch.full_like(logits, mask_value))
    
    if temperature <= 1e-6:
        return int(torch.argmax(masked_logits).item())

    # Sample Gumbel noise: -log(-log(U))
    u = torch.rand_like(masked_logits).clamp(min=1e-7, max=1.0 - 1e-7)
    gumbel_noise = -torch.log(-torch.log(u))
    
    noisy_logits = (masked_logits / temperature) + gumbel_noise
    return int(torch.argmax(noisy_logits).item())

