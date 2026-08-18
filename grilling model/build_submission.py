"""
Automated Standalone Submission Packaging Routine.
Embeds FP16 neural network weights into single-file grilling model/submission.py.
"""

from typing import Any, Dict, Optional
import os
import io
import gzip
import base64
import math
import numpy as np
import torch
import torch.nn.functional as F

from src.network import ChampionFullNetwork


def add_mask_safe_dirichlet_noise(
    logits: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
    epsilon: float = 0.25
) -> torch.Tensor:
    """
    Applies Dirichlet root exploration noise strictly over legal action masks.
    pi_noisy(a) = (1 - eps) * pi(a) + eps * Dir(alpha) on legal support.
    """
    valid_indices = (mask > 0.5).nonzero(as_tuple=True)[1]
    if len(valid_indices) <= 1:
        probs = torch.zeros_like(logits)
        probs[0, valid_indices] = 1.0
        return probs

    # Sub-logits for valid indices
    sub_logits = logits[0, valid_indices]
    sub_probs = F.softmax(sub_logits, dim=-1)

    # Dirichlet noise
    dirichlet = torch.distributions.Dirichlet(torch.full_like(sub_logits, alpha))
    noise = dirichlet.sample()

    noisy_sub_probs = (1.0 - epsilon) * sub_probs + epsilon * noise
    noisy_sub_probs = noisy_sub_probs / noisy_sub_probs.sum()

    full_probs = torch.zeros_like(logits)
    full_probs[0, valid_indices] = noisy_sub_probs
    return full_probs


def get_cosine_entropy_coeff(
    step: int,
    total_steps: int = 100000,
    start_coeff: float = 0.05,
    end_coeff: float = 0.002
) -> float:
    """
    Computes cosine-annealed policy entropy loss weight: 0.05 -> 0.002.
    """
    fraction = min(1.0, float(step) / max(1, total_steps))
    return end_coeff + 0.5 * (start_coeff - end_coeff) * (1.0 + math.cos(math.pi * fraction))


def serialize_weights_to_base85(weights_path: str) -> str:
    """
    Loads PyTorch weights, casts to FP16, compresses with gzip, and returns base85 string.
    """
    state_dict = torch.load(weights_path, map_location="cpu")
    # Cast float32 tensors to float16
    fp16_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            fp16_state_dict[k] = v.half()
        else:
            fp16_state_dict[k] = v

    buffer = io.BytesIO()
    torch.save(fp16_state_dict, buffer)
    compressed = gzip.compress(buffer.getvalue())
    return base64.b85encode(compressed).decode("utf-8")


SUBMISSION_TEMPLATE = '''"""
Kaggriculture AI Competition: Autonomous Champion Bot.
Single-File Standalone Tournament Submission.
Hierarchical 2-Tier Architecture with FiLM SE-ResNet Trunk and Hungarian Micro Solver.
"""

import io
import gzip
import base64
import math
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# Catalogs
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

CROP_SPECS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
}

BASE_PRICES = {
    "WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0, "STRAWBERRY": 120.0, "MELON": 250.0,
    "EGG": 50.0, "MILK": 160.0, "WOOL": 200.0, "FERTILIZER": 100.0,
}

BASE_INVENTORY = {
    "WHEAT": 100.0, "CARROT": 100.0, "TOMATO": 100.0, "STRAWBERRY": 100.0, "MELON": 100.0,
    "EGG": 100.0, "MILK": 100.0, "WOOL": 100.0, "FERTILIZER": 100.0,
}

SHOPS_CATALOG = [
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"
]

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]


def symlog_val(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def get_quadrant(r: int, c: int) -> str:
    if r < 5:
        return "NW" if c < 5 else "NE"
    else:
        return "SW" if c < 5 else "SE"


def encode_observation(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    opp_farm = farms[1 - player] if (1 - player) < len(farms) else {}

    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}
    town = obs.get("town", {}) or {}

    step = float(obs.get("step", 0))
    day = float(obs.get("day", 0))
    hour = float(obs.get("hour", 0))

    my_money = float(my_farm.get("money", 3000.0))
    opp_money = float(opp_farm.get("money", 3000.0))

    tiles = my_farm.get("tiles", [])
    unlocked_quads = set(my_farm.get("unlocked_quadrants", ["NW"]))
    opp_unlocked_quads = set(opp_farm.get("unlocked_quadrants", ["NW"]))

    farmer_pos = my_farm.get("farmer", [4, 4])
    hands_pos = my_farm.get("hands", [])

    spatial = np.zeros((24, 10, 10), dtype=np.float32)

    for r in range(10):
        for c in range(10):
            if (r in (4, 5)) and (c in (4, 5)):
                spatial[16, r, c] = 1.0
            if r in (0, 9) or c in (0, 9):
                spatial[17, r, c] = 0.5
            if (r in (4, 5)) and (c in (4, 5)):
                spatial[17, r, c] = 1.0

            q = get_quadrant(r, c)
            if q == "NW":
                spatial[19, r, c] = 0.0
            elif q == "NE":
                spatial[19, r, c] = 0.25
            elif q == "SW":
                spatial[19, r, c] = 0.50
            else:
                spatial[19, r, c] = 1.00

            min_shed_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in SHED_TILES)
            spatial[20, r, c] = float(min_shed_dist) / 18.0
            spatial[21, r, c] = (abs(r - 4.5) + abs(c - 4.5)) / 10.0
            spatial[22, r, c] = (r - 4.5) / 5.0
            spatial[23, r, c] = 1.0 if q in unlocked_quads else 0.0

    planted_counts = {c: 0 for c in CROPS}
    animal_counts = {a: 0 for a in ANIMALS}

    for r in range(min(10, len(tiles))):
        row = tiles[r]
        for c in range(min(10, len(row))):
            t = row[c]
            if t == "LOCKED":
                continue
            q = get_quadrant(r, c)
            if t is None:
                if q in unlocked_quads and (r, c) not in SHED_TILES:
                    spatial[10, r, c] = 1.0
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    if crop in CROPS:
                        crop_idx = CROPS.index(crop)
                        spatial[crop_idx, r, c] = 1.0
                        planted_counts[crop] = planted_counts.get(crop, 0) + 1
                    age = max(0.0, day - float(t.get("planted_day", day)))
                    spatial[5, r, c] = min(age / 12.0, 1.0)
                    spatial[6, r, c] = 1.0 if t.get("watered_today", False) else 0.0
                    fert_until = float(t.get("fertilized_until_day", -1))
                    spatial[7, r, c] = 1.0 if fert_until >= day else 0.0
                elif k == "COOP":
                    spatial[8, r, c] = 1.0
                    an = t.get("animal")
                    if an in ANIMALS:
                        spatial[11 + ANIMALS.index(an), r, c] = 1.0
                        animal_counts[an] = animal_counts.get(an, 0) + 1
                        spatial[14, r, c] = 0.0 if t.get("fed_today", False) else 1.0
                        spatial[15, r, c] = min(float(t.get("yield_units", 0)) / 4.0, 1.0)
                    else:
                        spatial[10, r, c] = 0.5
                elif k == "PASTURE":
                    spatial[9, r, c] = 1.0
                    an = t.get("animal")
                    if an in ANIMALS:
                        spatial[11 + ANIMALS.index(an), r, c] = 1.0
                        animal_counts[an] = animal_counts.get(an, 0) + 1
                        spatial[14, r, c] = 0.0 if t.get("fed_today", False) else 1.0
                        spatial[15, r, c] = min(float(t.get("yield_units", 0)) / 4.0, 1.0)
                    else:
                        spatial[10, r, c] = 0.5

    if isinstance(farmer_pos, (list, tuple)) and len(farmer_pos) >= 2:
        fx, fy = int(farmer_pos[0]), int(farmer_pos[1])
        if 0 <= fy < 10 and 0 <= fx < 10:
            spatial[18, fy, fx] += 1.0

    if isinstance(hands_pos, list):
        for h in hands_pos:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                hx, hy = int(h[0]), int(h[1])
                if 0 <= hy < 10 and 0 <= hx < 10:
                    spatial[18, hy, hx] += 1.0

    scalar = np.zeros(72, dtype=np.float32)
    scalar[0] = step / 720.0
    scalar[1] = hour / 24.0
    scalar[2] = day / 30.0
    scalar[3] = symlog_val(my_money) / 15.0
    scalar[4] = symlog_val(opp_money) / 15.0
    scalar[5] = np.clip((my_money - opp_money) / 10000.0, -10.0, 10.0)

    num_hands = len(hands_pos) if isinstance(hands_pos, list) else 0
    scalar[6] = ((num_hands + 1) * 20.0) / 260.0
    scalar[7] = float(my_farm.get("hires_today", 0)) / 5.0
    scalar[8] = float(len(unlocked_quads)) / 4.0
    scalar[9] = 1.0 if "NE" in unlocked_quads else 0.0
    scalar[10] = 1.0 if "SW" in unlocked_quads else 0.0

    prices = market.get("prices", {}) or {}
    for i, prod in enumerate(PRODUCTS):
        p_val = float(prices.get(prod, BASE_PRICES.get(prod, 25.0)))
        scalar[11 + i] = min(p_val / 500.0, 2.0)
        p_base = BASE_PRICES.get(prod, 25.0)
        scalar[20 + i] = min(p_val / max(p_base, 1.0), 5.0)

    mkt_inv = market.get("inventory", {}) or {}
    for i, prod in enumerate(PRODUCTS):
        cur_inv = float(mkt_inv.get(prod, BASE_INVENTORY.get(prod, 100.0)))
        base_inv = BASE_INVENTORY.get(prod, 100.0)
        scalar[29 + i] = min(cur_inv / max(base_inv, 1.0), 5.0)

    unlocked_shops = set(town.get("unlocked_shops", []) or [])
    for i, shop in enumerate(SHOPS_CATALOG):
        scalar[38 + i] = 1.0 if shop in unlocked_shops else 0.0

    shed = private.get("shed", {}) or {}
    total_shed = 0.0
    for i, prod in enumerate(PRODUCTS):
        s_count = float(shed.get(prod, 0))
        total_shed += s_count
        scalar[46 + i] = min(s_count / 25.0, 4.0)

    scalar[55] = min(total_shed / 100.0, 2.0)

    seeds = private.get("seeds", {}) or {}
    for i, crop in enumerate(CROPS):
        seed_count = float(seeds.get(crop, 0))
        scalar[56 + i] = min(seed_count / 10.0, 5.0)

    for i, crop in enumerate(CROPS):
        scalar[61 + i] = min(float(planted_counts.get(crop, 0)) / 25.0, 4.0)

    for i, an in enumerate(ANIMALS):
        scalar[66 + i] = min(float(animal_counts.get(an, 0)) / 10.0, 2.0)

    opp_hands = opp_farm.get("hands", [])
    opp_num_hands = len(opp_hands) if isinstance(opp_hands, list) else 0
    scalar[69] = min(opp_money / 10000.0, 10.0)
    scalar[70] = float(len(opp_unlocked_quads)) / 4.0
    scalar[71] = float(opp_num_hands + 1) / 13.0

    return spatial, scalar


# Neural Network Architecture
class SEBlock(nn.Module):
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
    def __init__(self, channels: int = 64, econ_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.film_proj = nn.Linear(econ_dim, 2 * channels)

    def forward(self, x: torch.Tensor, z_econ: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
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
    def __init__(self, spatial_in: int = 24, scalar_in: int = 72, channels: int = 64, econ_dim: int = 128, global_dim: int = 256):
        super().__init__()
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_in, econ_dim),
            nn.LayerNorm(econ_dim),
            nn.ReLU(inplace=True),
            nn.Linear(econ_dim, econ_dim),
            nn.LayerNorm(econ_dim),
            nn.ReLU(inplace=True),
        )
        self.in_conv = nn.Sequential(
            nn.Conv2d(spatial_in, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res1 = FiLMSEBlock(channels, econ_dim)
        self.res2 = FiLMSEBlock(channels, econ_dim)
        self.res3 = FiLMSEBlock(channels, econ_dim)
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
    def __init__(self, global_dim: int = 256, num_bins: int = 1001):
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
        return self.value_head(z_global), self.win_head(z_global)


class ChampionPolicyNetwork(nn.Module):
    def __init__(self, spatial_channels: int = 64, global_dim: int = 256):
        super().__init__()
        self.crop_head = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 5, kernel_size=1)
        )
        self.livestock_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )
        self.workforce_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 13)
        )
        self.land_head = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        self.seed_head = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 5)
        )
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


# Micro Solver & Guardrails
def get_manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_step_towards(curr: Tuple[int, int], target: Tuple[int, int], occupied: Optional[Set[Tuple[int, int]]] = None) -> Tuple[str, Tuple[int, int]]:
    cx, cy = curr
    tx, ty = target
    if cx == tx and cy == ty:
        return "PASS", curr

    occupied = occupied or set()
    moves: List[Tuple[str, Tuple[int, int]]] = []
    dx = tx - cx
    dy = ty - cy
    if abs(dx) >= abs(dy):
        if dx > 0:
            moves.append(("EAST", (cx + 1, cy)))
        elif dx < 0:
            moves.append(("WEST", (cx - 1, cy)))
        if dy > 0:
            moves.append(("SOUTH", (cx, cy + 1)))
        elif dy < 0:
            moves.append(("NORTH", (cx, cy - 1)))
    else:
        if dy > 0:
            moves.append(("SOUTH", (cx, cy + 1)))
        elif dy < 0:
            moves.append(("NORTH", (cx, cy - 1)))
        if dx > 0:
            moves.append(("EAST", (cx + 1, cy)))
        elif dx < 0:
            moves.append(("WEST", (cx - 1, cy)))

    for direction, next_pos in moves:
        nx, ny = next_pos
        if 0 <= nx < 10 and 0 <= ny < 10 and next_pos not in occupied:
            return direction, next_pos

    for direction, (nx, ny) in [("NORTH", (cx, cy - 1)), ("SOUTH", (cx, cy + 1)), ("EAST", (cx + 1, cy)), ("WEST", (cx - 1, cy))]:
        if 0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in occupied:
            return direction, (nx, ny)

    return "PASS", curr


def apply_market_guardrails(
    market_fractions: np.ndarray,
    obs: Dict[str, Any],
    seed_replenish_logits: Optional[np.ndarray] = None,
    land_expand_logit: float = 0.0,
    workforce_logit_idx: int = 0
) -> List[List[Any]]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}

    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    money = float(my_farm.get("money", 0.0))
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    unlocked_quads = set(my_farm.get("unlocked_quadrants", ["NW"]))
    hires_today = int(my_farm.get("hires_today", 0))

    tiles = my_farm.get("tiles", [])
    living_cows = 0
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if isinstance(t, dict) and t.get("animal") == "COW":
                living_cows += 1

    market_orders: List[List[Any]] = []
    cow_feed_reservation = max(4, living_cows * 2) if day < 28 else 0

    for i, prod in enumerate(PRODUCTS):
        qty = shed.get(prod, 0)
        if qty <= 0:
            continue
        frac = float(market_fractions[i]) if i < len(market_fractions) else 0.0
        if day >= 28:
            frac = 1.0

        if prod == "WHEAT":
            avail = max(0, qty - cow_feed_reservation)
        else:
            avail = qty

        sell_qty = int(math.floor(avail * frac)) if frac < 0.99 else avail
        if sell_qty > 0:
            market_orders.append(["SELL", prod, sell_qty])
            shed[prod] -= sell_qty

    remaining_shed_total = sum(shed.values())
    if hour == 23 and remaining_shed_total > 100:
        excess = remaining_shed_total - 95
        sorted_prods = sorted(shed.keys(), key=lambda p: float(prices.get(p, BASE_PRICES.get(p, 25.0))), reverse=True)
        for prod in sorted_prods:
            if excess <= 0:
                break
            can_dump = shed.get(prod, 0)
            if prod == "WHEAT" and day < 28:
                can_dump = max(0, can_dump - cow_feed_reservation)
            dump_qty = min(can_dump, excess)
            if dump_qty > 0:
                existing = [o for o in market_orders if o[0] == "SELL" and o[1] == prod]
                if existing:
                    existing[0][2] += dump_qty
                else:
                    market_orders.append(["SELL", prod, dump_qty])
                shed[prod] -= dump_qty
                excess -= dump_qty

    if day <= 16 and land_expand_logit > 0.0:
        if "NE" not in unlocked_quads and money >= 1200:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked_quads and "NE" in unlocked_quads and money >= 2400 and day <= 10:
            market_orders.append(["BUY_LAND"])
            money -= 2000

    num_hands = len(my_farm.get("hands", []))
    if day < 28 and hour <= 1 and hires_today < 2 and workforce_logit_idx > num_hands:
        if money >= (num_hands + 2) * 20:
            market_orders.append(["HIRE"])
            money -= 20

    if seed_replenish_logits is not None and day < 27 and money >= 50:
        for i, crop in enumerate(CROPS):
            logit = float(seed_replenish_logits[i])
            cur_seeds = seeds.get(crop, 0)
            cost = CROP_SPECS[crop]["seed"]
            if logit > 0.0 and cur_seeds < 6 and money >= cost * 2:
                buy_qty = min(4, int(money // cost))
                if buy_qty > 0:
                    market_orders.append(["BUY_SEED", crop, buy_qty])
                    money -= buy_qty * cost

    return market_orders[:10]


def solve_micro_actions(
    obs: Dict[str, Any],
    crop_heatmaps: Optional[np.ndarray] = None,
    livestock_quotas: Optional[np.ndarray] = None
) -> Tuple[List[Any], List[List[Any]]]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    day = int(obs.get("day", 0))

    farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
    hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
    tiles = my_farm.get("tiles", [])

    units = [farmer_pos] + hands_pos
    num_units = len(units)
    unit_actions: List[Optional[List[Any]]] = [None] * num_units
    occupied_destinations: Set[Tuple[int, int]] = set()

    for u_idx, u_pos in enumerate(units):
        ux, uy = u_pos
        u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
        if isinstance(u_tile, dict):
            k = u_tile.get("kind")
            an = u_tile.get("animal")
            if k in ["COOP", "PASTURE"] and an:
                if not u_tile.get("fed_today", False):
                    unit_actions[u_idx] = ["FEED"]
                    occupied_destinations.add(u_pos)
                    continue
                if not u_tile.get("cared_today", False):
                    unit_actions[u_idx] = ["CARE"]
                    occupied_destinations.add(u_pos)
                    continue
                if u_tile.get("yield_units", 0) > 0:
                    unit_actions[u_idx] = ["HARVEST"]
                    occupied_destinations.add(u_pos)
                    continue
            elif k == "PLANT":
                crop = u_tile.get("crop", "CARROT")
                cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                age = day - u_tile.get("planted_day", day)
                yield_u = u_tile.get("yield_units", 0)
                if (not cspec["ongoing"] and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                    cspec["ongoing"] and age >= cspec["first_yield_day"] and yield_u > 0
                ):
                    unit_actions[u_idx] = ["HARVEST"]
                    occupied_destinations.add(u_pos)
                    continue
                if not u_tile.get("watered_today", False):
                    unit_actions[u_idx] = ["WATER"]
                    occupied_destinations.add(u_pos)
                    continue
            elif k == "WEED":
                unit_actions[u_idx] = ["DIG"]
                occupied_destinations.add(u_pos)
                continue

    tasks: List[Tuple[Tuple[int, int], float, str, Optional[str]]] = []
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            pos = (c, r)
            if isinstance(t, dict):
                k = t.get("kind")
                an = t.get("animal")
                if k in ["COOP", "PASTURE"] and an:
                    if not t.get("fed_today", False):
                        tasks.append((pos, 25.0, "FEED", None))
                    elif not t.get("cared_today", False):
                        tasks.append((pos, 18.0, "CARE", None))
                    elif t.get("yield_units", 0) > 0:
                        tasks.append((pos, 20.0, "HARVEST_ANIMAL", None))
                elif k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                    age = day - t.get("planted_day", day)
                    yield_u = t.get("yield_units", 0)
                    if (not cspec["ongoing"] and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                        cspec["ongoing"] and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        tasks.append((pos, 22.0, "HARVEST_CROP", None))
                    elif not t.get("watered_today", False):
                        tasks.append((pos, 16.0, "WATER_CROP", None))
                elif k == "WEED":
                    tasks.append((pos, 8.0, "DIG_WEED", None))
            elif t is None and pos not in SHED_TILES and day < 27:
                best_crop = "CARROT"
                priority_bonus = 5.0
                if crop_heatmaps is not None:
                    c_logits = crop_heatmaps[:, r, c]
                    best_crop_idx = int(np.argmax(c_logits))
                    best_crop = CROPS[best_crop_idx]
                    priority_bonus += float(c_logits[best_crop_idx])
                tasks.append((pos, priority_bonus, "PLANT_TILE", best_crop))

    unassigned_unit_indices = [i for i, act in enumerate(unit_actions) if act is None]
    if unassigned_unit_indices and tasks:
        num_free = len(unassigned_unit_indices)
        num_tasks = len(tasks)
        cost_matrix = np.zeros((num_free, num_tasks), dtype=np.float32)

        for i_idx, u_idx in enumerate(unassigned_unit_indices):
            u_pos = units[u_idx]
            for j_idx, (t_pos, priority, _, _) in enumerate(tasks):
                dist = get_manhattan_dist(u_pos, t_pos)
                cost_matrix[i_idx, j_idx] = float(dist) - priority

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i_idx, j_idx in zip(row_ind, col_ind):
            u_idx = unassigned_unit_indices[i_idx]
            u_pos = units[u_idx]
            t_pos, _, task_type, best_crop = tasks[j_idx]
            if u_pos == t_pos:
                if task_type == "PLANT_TILE" and best_crop:
                    unit_actions[u_idx] = ["PLANT", best_crop]
                elif task_type == "FEED":
                    unit_actions[u_idx] = ["FEED"]
                elif task_type == "CARE":
                    unit_actions[u_idx] = ["CARE"]
                elif task_type in ["HARVEST_ANIMAL", "HARVEST_CROP"]:
                    unit_actions[u_idx] = ["HARVEST"]
                elif task_type == "DIG_WEED":
                    unit_actions[u_idx] = ["DIG"]
                else:
                    unit_actions[u_idx] = ["PASS"]
                occupied_destinations.add(u_pos)
            else:
                direction, next_pos = get_step_towards(u_pos, t_pos, occupied_destinations)
                unit_actions[u_idx] = [direction]
                occupied_destinations.add(next_pos)

    for u_idx in range(num_units):
        if unit_actions[u_idx] is None:
            u_pos = units[u_idx]
            if u_pos not in SHED_TILES:
                nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                direction, next_pos = get_step_towards(u_pos, nearest_shed, occupied_destinations)
                unit_actions[u_idx] = [direction]
                occupied_destinations.add(next_pos)
            else:
                unit_actions[u_idx] = ["PASS"]
                occupied_destinations.add(u_pos)

    farmer_action = unit_actions[0] if unit_actions else ["PASS"]
    hands_actions = unit_actions[1:] if len(unit_actions) > 1 else []
    return farmer_action, hands_actions


# Standalone Model & Weights Loader
WEIGHTS_BASE85 = "{WEIGHTS_BASE85_PAYLOAD}"


class StandaloneChampionAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.network = ChampionFullNetwork().to(self.device)
        self.network.eval()
        if WEIGHTS_BASE85:
            try:
                raw_bytes = gzip.decompress(base64.b85decode(WEIGHTS_BASE85))
                state_dict = torch.load(io.BytesIO(raw_bytes), map_location=self.device)
                # Convert fp16 back to float32
                f32_state = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
                self.network.load_state_dict(f32_state)
            except Exception as e:
                pass

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        x_spatial_np, x_scalar_np = encode_observation(obs)
        x_spatial = torch.from_numpy(x_spatial_np).unsqueeze(0).to(self.device)
        x_scalar = torch.from_numpy(x_scalar_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.network(x_spatial, x_scalar)

        crop_heatmaps = outputs["crop_heatmaps"].squeeze(0).cpu().numpy()
        livestock_quotas = outputs["livestock_quotas"].squeeze(0).cpu().numpy()
        workforce_logits = outputs["workforce_logits"].squeeze(0)
        land_expand_logit = outputs["land_expand_logit"].item()
        seed_replenish_logits = outputs["seed_replenish_logits"].squeeze(0).cpu().numpy()
        market_fractions = outputs["market_fractions"].squeeze(0).cpu().numpy()

        player = obs.get("player", 0)
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        money = float(my_farm.get("money", 0.0))
        unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
        num_workers = len(my_farm.get("hands", []))

        workforce_idx = int(torch.argmax(workforce_logits).item())
        if workforce_idx * 20 > money and workforce_idx > num_workers:
            workforce_idx = num_workers

        market_orders = apply_market_guardrails(
            market_fractions=market_fractions,
            obs=obs,
            seed_replenish_logits=seed_replenish_logits,
            land_expand_logit=land_expand_logit,
            workforce_logit_idx=workforce_idx
        )

        farmer_act, hands_acts = solve_micro_actions(
            obs=obs,
            crop_heatmaps=crop_heatmaps,
            livestock_quotas=livestock_quotas
        )

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders,
        }


_standalone_agent = StandaloneChampionAgent()


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _standalone_agent(obs, config)
'''


def build_submission_file(output_path: str, weights_path: Optional[str] = None):
    """
    Builds the standalone single-file submission.py script.
    """
    base85_payload = ""
    if weights_path is not None and os.path.exists(weights_path):
        base85_payload = serialize_weights_to_base85(weights_path)

    content = SUBMISSION_TEMPLATE.replace("{WEIGHTS_BASE85_PAYLOAD}", base85_payload)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "submission.py")
    ckpt_file = os.path.join(os.path.dirname(__file__), "weights", "champion_weights.pt")
    if not os.path.exists(ckpt_file):
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        net = ChampionFullNetwork()
        torch.save(net.state_dict(), ckpt_file)
    build_submission_file(out_file, ckpt_file)
    print(f"Built standalone submission bot at {out_file}")
