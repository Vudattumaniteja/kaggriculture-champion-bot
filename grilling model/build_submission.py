"""
Automated Standalone Submission Packaging Routine.
Embeds base85-compressed FP16 neural network weights into single-file grilling model/submission.py.
Completely self-contained, complying strictly with Kaggle agent(obs, config=None) contract.
"""

from typing import Any, Dict, Optional, Tuple, Set, List
import os
import io
import gzip
import base64
import math
import numpy as np
import torch
import torch.nn.functional as F

from src.network import (
    ChampionFullNetwork,
    add_mask_safe_dirichlet_noise,
    get_cosine_entropy_coeff,
    get_gumbel_temperature,
    sample_gumbel_action,
)


def serialize_weights_to_base85(weights_path: str) -> str:
    """
    Loads PyTorch weights, casts floating point tensors to FP16, compresses with gzip, and returns base85 string.
    """
    state_dict = torch.load(weights_path, map_location="cpu")
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
Hierarchical 2-Tier Architecture with FiLM SE-ResNet Trunk, Gumbel Action Sampling, and Hungarian Micro Solver.
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

# Catalogs & Constants
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

SHED_TILES = {(4, 4), (5, 4), (4, 5), (5, 5)}

PASTURE_LOCS = [
    # NW (6)
    (4, 4), (3, 4), (4, 3), (3, 3), (2, 4), (4, 2),
    # NE (5)
    (5, 4), (6, 4), (6, 3), (5, 2), (7, 4),
    # SW (3)
    (4, 5), (3, 5), (2, 5),
]
PASTURE_SET = set(PASTURE_LOCS)

NW_WHEAT_TILES = {
    (0, 0), (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1),
}


def symlog_val(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x)


def get_quadrant(r: int, c: int) -> str:
    if r < 5:
        return "NW" if c < 5 else "NE"
    else:
        return "SW" if c < 5 else "SE"


def get_manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_move(curr: Tuple[int, int], target: Tuple[int, int]) -> str:
    cx, cy = curr
    tx, ty = target
    if cx == tx and cy == ty:
        return "PASS"
    dx = tx - cx
    dy = ty - cy
    if abs(dx) >= abs(dy):
        if dx > 0:
            return "EAST"
        elif dx < 0:
            return "WEST"
    if dy > 0:
        return "SOUTH"
    elif dy < 0:
        return "NORTH"
    return "PASS"


def get_step_towards(
    curr: Tuple[int, int],
    target: Tuple[int, int],
    reserved_dests: Optional[Set[Tuple[int, int]]] = None,
    unmoved_positions: Optional[Set[Tuple[int, int]]] = None,
) -> Tuple[str, Tuple[int, int]]:
    cx, cy = curr
    tx, ty = target
    reserved = reserved_dests if reserved_dests is not None else set()
    unmoved = unmoved_positions if unmoved_positions is not None else set()

    moves = []
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

    all_dirs = [("NORTH", (cx, cy - 1)), ("SOUTH", (cx, cy + 1)), ("EAST", (cx + 1, cy)), ("WEST", (cx - 1, cy))]
    for d, pos in all_dirs:
        if (d, pos) not in moves:
            moves.append((d, pos))

    for direction, next_pos in moves:
        nx, ny = next_pos
        if 0 <= nx < 10 and 0 <= ny < 10:
            if next_pos not in reserved and next_pos not in unmoved:
                return direction, next_pos

    if curr not in reserved:
        return "PASS", curr

    for direction, next_pos in moves:
        nx, ny = next_pos
        if 0 <= nx < 10 and 0 <= ny < 10:
            if next_pos not in reserved:
                return direction, next_pos

    for direction, next_pos in all_dirs:
        nx, ny = next_pos
        if 0 <= nx < 10 and 0 <= ny < 10:
            if next_pos not in reserved:
                return direction, next_pos

    for direction, next_pos in all_dirs:
        nx, ny = next_pos
        if 0 <= nx < 10 and 0 <= ny < 10:
            return direction, next_pos

    return "PASS", curr


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

            min_shed_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in [(4, 4), (5, 4), (4, 5), (5, 5)])
            spatial[20, r, c] = float(min_shed_dist) / 18.0
            spatial[21, r, c] = (abs(r - 4.5) + abs(c - 4.5)) / 10.0
            spatial[22, r, c] = (r - 4.5) / 5.0
            spatial[23, r, c] = 1.0 if q in unlocked_quads else 0.0

    for r in range(min(10, len(tiles))):
        row = tiles[r]
        for c in range(min(10, len(row))):
            t = row[c]
            if t == "LOCKED":
                continue
            q = get_quadrant(r, c)
            if t is None:
                if q in unlocked_quads and (r, c) not in [(4, 4), (5, 4), (4, 5), (5, 5)]:
                    spatial[10, r, c] = 1.0
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    if crop in CROPS:
                        crop_idx = CROPS.index(crop)
                        spatial[crop_idx, r, c] = 1.0
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
                        spatial[14, r, c] = 0.0 if t.get("fed_today", False) else 1.0
                        spatial[15, r, c] = min(float(t.get("yield_units", 0)) / 4.0, 1.0)
                    else:
                        spatial[10, r, c] = 0.5
                elif k == "PASTURE":
                    spatial[9, r, c] = 1.0
                    an = t.get("animal")
                    if an in ANIMALS:
                        spatial[11 + ANIMALS.index(an), r, c] = 1.0
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

    # Scalar features (72 dims)
    scalar = np.zeros(72, dtype=np.float32)
    scalar[0] = step / 720.0
    scalar[1] = day / 30.0
    scalar[2] = hour / 24.0
    scalar[3] = math.sin(2.0 * math.pi * hour / 24.0)
    scalar[4] = math.cos(2.0 * math.pi * hour / 24.0)
    scalar[5] = math.sin(2.0 * math.pi * day / 30.0)
    scalar[6] = math.cos(2.0 * math.pi * day / 30.0)

    scalar[7] = symlog_val(my_money) / 15.0
    scalar[8] = symlog_val(opp_money) / 15.0
    scalar[9] = symlog_val(my_money - opp_money) / 15.0
    scalar[10] = float(len(hands_pos)) * 20.0 / 240.0

    prices = market.get("prices", {}) or {}
    for i, prod in enumerate(PRODUCTS):
        p = float(prices.get(prod, BASE_PRICES.get(prod, 25.0)))
        bp = BASE_PRICES.get(prod, 25.0)
        scalar[11 + i] = p / (bp * 2.0)
        scalar[20 + i] = (p - bp) / bp

    return spatial, scalar


# Neural Network Definition
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


def get_gumbel_temperature(step: int) -> float:
    return 1.0 if step < 48 else 0.0


def sample_gumbel_action(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0, mask_value: float = -1e9) -> int:
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    if mask.dim() > 1:
        mask = mask.squeeze(0)
    masked_logits = torch.where(mask.to(logits.device) > 0.5, logits, torch.full_like(logits, mask_value))
    if temperature <= 1e-6:
        return int(torch.argmax(masked_logits).item())
    u = torch.rand_like(masked_logits).clamp(min=1e-7, max=1.0 - 1e-7)
    gumbel_noise = -torch.log(-torch.log(u))
    noisy_logits = (masked_logits / temperature) + gumbel_noise
    return int(torch.argmax(noisy_logits).item())


class MicroTask:
    __slots__ = ("task_type", "pos", "utility", "extra")
    def __init__(self, task_type: str, pos: Tuple[int, int], utility: float, extra: Any = None):
        self.task_type = task_type
        self.pos = pos
        self.utility = utility
        self.extra = extra


def _get_target_hires(day: int) -> int:
    if day <= 6:
        return 5
    elif day <= 8:
        return 8
    elif day <= 28:
        return 12
    else:
        return 8


def apply_market_guardrails(
    market_fractions: np.ndarray,
    obs: Dict[str, Any],
    seed_replenish_logits: Optional[np.ndarray] = None,
    land_expand_logit: float = 0.0,
    workforce_logit_idx: int = 0,
    livestock_quotas: Optional[np.ndarray] = None,
) -> List[List[Any]]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}
    market_prices = market.get("prices", {}) or {}

    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", int(obs.get("step", 0)) % 24))
    step = int(obs.get("step", 0))
    money = float(my_farm.get("money", 0.0))
    tiles = my_farm.get("tiles", [])
    unlocked_quads = list(my_farm.get("unlocked_quadrants", ["NW"]))
    hires_today = int(my_farm.get("hires_today", 0))

    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    inventories = [dict(inv) for inv in (private.get("inventories", []) or [])]
    shed_total = sum(shed.values())

    living_cows = 0
    living_sheep = 0
    crop_tiles_count = 0
    wheat_tiles_count = 0
    straw_tiles_count = 0

    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if isinstance(t, dict):
                k = t.get("kind")
                animal = t.get("animal")
                if k == "PASTURE":
                    if animal == "COW":
                        living_cows += 1
                    elif animal == "SHEEP":
                        living_sheep += 1
                elif k == "PLANT":
                    crop_tiles_count += 1
                    crop_name = t.get("crop")
                    if crop_name == "WHEAT":
                        wheat_tiles_count += 1
                    elif crop_name == "STRAWBERRY":
                        straw_tiles_count += 1

    total_living_animals = living_cows + living_sheep
    cows_in_shed = shed.get("COW", 0)
    sheep_in_shed = shed.get("SHEEP", 0)
    unplaced_shed_animals = cows_in_shed + sheep_in_shed
    cows_in_inv = sum(inv.get("COW", 0) for inv in inventories)
    sheep_in_inv = sum(inv.get("SHEEP", 0) for inv in inventories)
    total_animals_owned = total_living_animals + unplaced_shed_animals + cows_in_inv + sheep_in_inv

    hire_orders: List[List[Any]] = []
    feed_orders: List[List[Any]] = []
    sell_orders: List[List[Any]] = []
    land_orders: List[List[Any]] = []
    seed_orders: List[List[Any]] = []
    animal_orders: List[List[Any]] = []

    # A. Day 0 Turn 0 Initialization ($2,929 spend, $71 buffer)
    if day == 0 and step <= 1:
        for _ in range(5):
            hire_orders.append(["HIRE"])
        hires_today += 5
        money -= 24

        animal_orders.append(["BUY_ANIMAL", "COW", 2])
        money -= 800
        animal_orders.append(["BUY_ANIMAL", "SHEEP", 2])
        money -= 1000

        seed_orders.append(["BUY_SEED", "WHEAT", 7])
        money -= 70
        seed_orders.append(["BUY_SEED", "MELON", 12])
        money -= 960
        feed_orders.append(["BUY_PRODUCT", "WHEAT", 3])
        money -= 75

    elif day == 0 and step == 2 and shed.get("WHEAT", 0) >= 2:
        sell_orders.append(["SELL", "WHEAT", 2])
        money += 50
        shed["WHEAT"] -= 2

    else:
        # 1. Multi-Turn Daily Hiring (Hours 0..3)
        target_hires = max(_get_target_hires(day), workforce_logit_idx)
        if hour <= 3 and hires_today < target_hires and money >= 2:
            hires_needed = target_hires - hires_today
            hire_chunk = min(hires_needed, 5)
            for _ in range(hire_chunk):
                hire_orders.append(["HIRE"])
            hires_today += hire_chunk

        # 2. Critical Feed On-Demand (Whenever shed wheat < total_animals + 2)
        wheat_in_shed = shed.get("WHEAT", 0)
        if day <= 27 and total_living_animals > 0 and wheat_in_shed < max(4, total_living_animals + 2) and money >= 25:
            needed_feed = max(2, total_living_animals * 2 - wheat_in_shed)
            buy_feed = min(needed_feed, int(money // 25), 8)
            if buy_feed > 0:
                feed_orders.append(["BUY_PRODUCT", "WHEAT", buy_feed])
                money -= buy_feed * 25
                shed["WHEAT"] = wheat_in_shed + buy_feed

        # 3. Market Sales Guardrail
        cow_feed_reservation = living_cows * 2 if day < 28 else 0
        safe_wheat = max(6, (total_living_animals + 2) if day < 28 else 0)

        if day >= 28:
            for prod in PRODUCTS:
                cnt = shed.get(prod, 0)
                if cnt > 0:
                    sell_orders.append(["SELL", prod, cnt])
                    shed[prod] = 0
        else:
            p_melon = market_prices.get("MELON", 250)
            p_milk = market_prices.get("MILK", 160)
            p_wool = market_prices.get("WOOL", 200)
            p_straw = market_prices.get("STRAWBERRY", 120)
            p_fert = market_prices.get("FERTILIZER", 95)
            p_wheat = market_prices.get("WHEAT", 25)

            near_capacity = (shed_total >= 85)
            max_policy_frac = float(np.max(market_fractions)) if len(market_fractions) > 0 else 0.0

            if max_policy_frac > 0.01:
                # Melons
                melon_cnt = shed.get("MELON", 0)
                if melon_cnt > 0:
                    sell_m = min(melon_cnt, 4)
                    sell_orders.append(["SELL", "MELON", sell_m])
                    money += sell_m * p_melon
                    shed["MELON"] -= sell_m

                # Milk
                milk_cnt = shed.get("MILK", 0)
                if milk_cnt > 0 and (p_milk >= 75 or near_capacity or money < 300):
                    sell_m = min(milk_cnt, 3)
                    sell_orders.append(["SELL", "MILK", sell_m])
                    money += sell_m * p_milk
                    shed["MILK"] -= sell_m

                # Wool
                wool_cnt = shed.get("WOOL", 0)
                if wool_cnt > 0 and (p_wool >= 95 or near_capacity or money < 300):
                    sell_w = min(wool_cnt, 2)
                    sell_orders.append(["SELL", "WOOL", sell_w])
                    money += sell_w * p_wool
                    shed["WOOL"] -= sell_w

                # Strawberries
                straw_cnt = shed.get("STRAWBERRY", 0)
                if straw_cnt > 0 and (p_straw >= 60 or near_capacity or money < 300):
                    sell_s = min(straw_cnt, 3)
                    sell_orders.append(["SELL", "STRAWBERRY", sell_s])
                    money += sell_s * p_straw
                    shed["STRAWBERRY"] -= sell_s

                # Wheat: Keep safe reserve for feeding, drip-feed excess
                wheat_cnt = shed.get("WHEAT", 0)
                wheat_frac = float(market_fractions[0]) if len(market_fractions) > 0 else 0.0
                if wheat_frac > 0.99:
                    avail_wheat = max(0, wheat_cnt - cow_feed_reservation)
                    if avail_wheat > 0:
                        sell_orders.append(["SELL", "WHEAT", avail_wheat])
                        money += avail_wheat * p_wheat
                        shed["WHEAT"] -= avail_wheat
                elif wheat_cnt > safe_wheat + 2 and (p_wheat >= 15 or near_capacity):
                    sell_wheat = min(wheat_cnt - safe_wheat, 3)
                    sell_orders.append(["SELL", "WHEAT", sell_wheat])
                    money += sell_wheat * p_wheat
                    shed["WHEAT"] -= sell_wheat

                # Fertilizer
                fert_cnt = shed.get("FERTILIZER", 0)
                if fert_cnt > 2 and (p_fert >= 65 or near_capacity or money < 300):
                    sell_f = min(fert_cnt - 2, 3)
                    sell_orders.append(["SELL", "FERTILIZER", sell_f])
                    money += sell_f * p_fert
                    shed["FERTILIZER"] -= sell_f
                elif day <= 9 and fert_cnt > 0 and money < 250:
                    sell_f = min(fert_cnt, 2)
                    sell_orders.append(["SELL", "FERTILIZER", sell_f])
                    money += sell_f * p_fert
                    shed["FERTILIZER"] -= sell_f

                for prod in ["CARROT", "TOMATO", "EGG"]:
                    cnt = shed.get(prod, 0)
                    if cnt > 0:
                        sell_c = min(cnt, 2)
                        sell_orders.append(["SELL", prod, sell_c])
                        money += sell_c * market_prices.get(prod, 35)
                        shed[prod] -= sell_c

        # 4. Midnight Shed Overflow Guardrail (hour == 23 and total > 100)
        remaining_shed_total = sum(shed.values())
        if hour == 23 and remaining_shed_total > 100:
            excess = remaining_shed_total - 100
            sorted_prods = sorted(
                [p for p in shed.keys() if shed[p] > 0],
                key=lambda p: float(market_prices.get(p, BASE_PRICES.get(p, 25.0))),
                reverse=True,
            )
            for prod in sorted_prods:
                if excess <= 0:
                    break
                can_dump = shed.get(prod, 0)
                if prod == "WHEAT" and day < 28:
                    can_dump = max(0, can_dump - cow_feed_reservation)
                dump_qty = min(can_dump, excess)
                if dump_qty > 0:
                    existing = [o for o in sell_orders if o[0] == "SELL" and o[1] == prod]
                    if existing:
                        existing[0][2] += dump_qty
                    else:
                        sell_orders.append(["SELL", prod, dump_qty])
                    shed[prod] -= dump_qty
                    excess -= dump_qty

            if excess > 0 and shed.get("WHEAT", 0) > 0:
                dump_wheat = min(shed["WHEAT"], excess)
                existing = [o for o in sell_orders if o[0] == "SELL" and o[1] == "WHEAT"]
                if existing:
                    existing[0][2] += dump_wheat
                else:
                    sell_orders.append(["SELL", "WHEAT", dump_wheat])
                shed["WHEAT"] -= dump_wheat
                excess -= dump_wheat

        # 5. Land Expansion (TOP PRIORITY over animal buying: NE @ Day 6, SW @ Day 11)
        if "NE" not in unlocked_quads and (money >= 1050 and day >= 6 or (land_expand_logit > 0.0 and money >= 1000)):
            land_orders.append(["BUY_LAND"])
            money -= 1000
            unlocked_quads.append("NE")
        elif "SW" not in unlocked_quads and "NE" in unlocked_quads and (money >= 2050 and day >= 11 or (land_expand_logit > 0.0 and money >= 2000)):
            land_orders.append(["BUY_LAND"])
            money -= 2000
            unlocked_quads.append("SW")

        # 6. Strawberry & Wheat Seed Purchasing (High priority on Day 6+)
        if day < 26:
            wheat_seeds = seeds.get("WHEAT", 0)
            target_wheat_seeds = 12 if day <= 24 else 25
            if (wheat_seeds + wheat_tiles_count) < target_wheat_seeds and money >= 40:
                buy_w = min(target_wheat_seeds - (wheat_seeds + wheat_tiles_count), int((money - 30) // 10), 6)
                if buy_w > 0:
                    seed_orders.append(["BUY_SEED", "WHEAT", buy_w])
                    money -= buy_w * 10
                    seeds["WHEAT"] = wheat_seeds + buy_w

            straw_seeds = seeds.get("STRAWBERRY", 0)
            target_straw = 12 if day <= 8 else (20 if day <= 10 else (38 if day <= 24 else 20))

            if "NE" in unlocked_quads and (straw_tiles_count + straw_seeds) < target_straw and money >= 110:
                needed_s = target_straw - (straw_tiles_count + straw_seeds)
                buy_s = min(needed_s, int((money - 40) // 100), 6)
                if buy_s > 0:
                    seed_orders.append(["BUY_SEED", "STRAWBERRY", buy_s])
                    money -= buy_s * 100
                    seeds["STRAWBERRY"] = straw_seeds + buy_s

        # 7. Herd Scaling (10 Cows + 4 Sheep through Day 18)
        if day <= 18 and unplaced_shed_animals <= 2:
            target_herd_cows = 2 if day < 3 else (3 if day < 5 else (4 if day < 7 else (6 if day < 9 else (8 if day < 11 else 10))))
            target_herd_sheep = 2 if day < 7 else 4

            if livestock_quotas is not None:
                q_cows = int(round(livestock_quotas[1])) if len(livestock_quotas) > 1 else 0
                q_sheep = int(round(livestock_quotas[2])) if len(livestock_quotas) > 2 else 0
                target_herd_cows = max(target_herd_cows, q_cows)
                target_herd_sheep = max(target_herd_sheep, q_sheep)

            rem_cows = max(0, min(10, target_herd_cows) - (living_cows + cows_in_shed + cows_in_inv))
            rem_sheep = max(0, min(4, target_herd_sheep) - (living_sheep + sheep_in_shed + sheep_in_inv))

            min_buffer = 1100.0 if (day == 6 and "NE" not in unlocked_quads) else 250.0
            avail_animal_cash = max(0.0, money - min_buffer)

            if rem_sheep > 0 and avail_animal_cash >= 500:
                animal_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500
                sheep_in_shed += 1
                avail_animal_cash -= 500
            elif rem_cows > 0 and avail_animal_cash >= 400:
                buy_c = min(rem_cows, int(avail_animal_cash // 400), 2)
                if buy_c > 0:
                    animal_orders.append(["BUY_ANIMAL", "COW", buy_c])
                    money -= buy_c * 400
                    cows_in_shed += buy_c
                    avail_animal_cash -= buy_c * 400

    all_market_orders = hire_orders + feed_orders + sell_orders + land_orders + seed_orders + animal_orders
    return all_market_orders[:10]


def solve_micro_actions(
    obs: Dict[str, Any],
    crop_heatmaps: Optional[np.ndarray] = None,
    livestock_quotas: Optional[np.ndarray] = None,
    lambda_logit: float = 1.0,
) -> Tuple[List[Any], List[List[Any]]]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}

    day = int(obs.get("day", 0))
    farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
    hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
    tiles = my_farm.get("tiles", [])

    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    inventories = [dict(inv) for inv in (private.get("inventories", []) or [])]

    living_animals = []
    living_cows = 0
    living_sheep = 0
    empty_pastures = []
    crop_tiles = []
    empty_tiles = []
    weed_tiles = []

    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            pos = (c, r)
            t = tiles[r][c]
            if t is None:
                empty_tiles.append(pos)
            elif isinstance(t, dict):
                k = t.get("kind")
                animal = t.get("animal")
                if k in ["PASTURE", "COOP"]:
                    if animal == "COW":
                        living_cows += 1
                        living_animals.append((c, r, "COW", t))
                    elif animal == "SHEEP":
                        living_sheep += 1
                        living_animals.append((c, r, "SHEEP", t))
                    elif animal is None:
                        empty_pastures.append(pos)
                elif k == "PLANT":
                    crop_tiles.append((c, r, t))
                elif k == "WEED":
                    weed_tiles.append(pos)

    total_living_animals = living_cows + living_sheep
    total_structures = len(living_animals) + len(empty_pastures)
    cows_in_shed = shed.get("COW", 0)
    sheep_in_shed = shed.get("SHEEP", 0)
    unplaced_shed_animals = cows_in_shed + sheep_in_shed

    cows_in_inv = sum(inv.get("COW", 0) for inv in inventories)
    sheep_in_inv = sum(inv.get("SHEEP", 0) for inv in inventories)
    fert_in_inv = sum(inv.get("FERTILIZER", 0) for inv in inventories)
    total_animals_owned = total_living_animals + unplaced_shed_animals + cows_in_inv + sheep_in_inv

    tasks = []

    # 1. Animal Critical Feed
    for ax, ay, aname, atile in living_animals:
        if not atile.get("fed_today", False):
            tasks.append(MicroTask("FEED", (ax, ay), utility=1200.0, extra={"animal": aname}))

    # 2. Crop Critical Watering
    for cx, cy, ctile in crop_tiles:
        if not ctile.get("watered_today", False):
            tasks.append(MicroTask("WATER_CROP", (cx, cy), utility=1180.0))

    # 3. Place Animals into Pastures
    if unplaced_shed_animals > 0 or cows_in_inv > 0 or sheep_in_inv > 0:
        for ep in empty_pastures:
            tasks.append(MicroTask("PLACE_PASTURE_ANIMAL", ep, utility=1100.0))

    # 4. Build Pastures on Dedicated Locations
    pastures_needed = max(0, total_animals_owned - total_structures)
    if pastures_needed > 0 and len(empty_tiles) > 0:
        candidate_spots = [p for p in PASTURE_LOCS if p in empty_tiles]
        if not candidate_spots:
            candidate_spots = sorted([p for p in empty_tiles if p not in NW_WHEAT_TILES], key=lambda p: get_manhattan_dist(p, (4, 4)))
        for p in candidate_spots[:pastures_needed]:
            tasks.append(MicroTask("BUILD_PASTURE", p, utility=1050.0))

    # 5. Crop Harvesting
    for cx, cy, ctile in crop_tiles:
        pos = (cx, cy)
        crop = ctile.get("crop", "WHEAT")
        age = day - ctile.get("planted_day", day)
        yield_units = ctile.get("yield_units", 0)

        if crop == "WHEAT":
            if age >= 4 or (day >= 28 and yield_units > 0):
                tasks.append(MicroTask("HARVEST_CROP", pos, utility=850.0))
        elif crop == "MELON":
            if age >= 10 or (day >= 28 and yield_units > 0):
                tasks.append(MicroTask("HARVEST_CROP", pos, utility=890.0))
        elif crop == "STRAWBERRY":
            if yield_units > 0:
                tasks.append(MicroTask("HARVEST_CROP", pos, utility=880.0))
        else:
            if yield_units > 0 or age >= 4:
                tasks.append(MicroTask("HARVEST_CROP", pos, utility=700.0))

    # 6. Fertilizer Application
    total_fert_available = shed.get("FERTILIZER", 0) + fert_in_inv
    if total_fert_available > 0 and day <= 27:
        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            crop = ctile.get("crop", "")
            age = day - ctile.get("planted_day", day)
            fert_until = ctile.get("fertilized_until_day", -1)
            if fert_until < day:
                if crop == "MELON" and age <= 10:
                    tasks.append(MicroTask("FERTILIZE_CROP", pos, utility=860.0))
                elif crop == "STRAWBERRY":
                    tasks.append(MicroTask("FERTILIZE_CROP", pos, utility=855.0))

    # 7. Livestock Care, Manure & Harvest
    for ax, ay, aname, atile in living_animals:
        pos = (ax, ay)
        if not atile.get("cared_today", False):
            tasks.append(MicroTask("CARE", pos, utility=850.0, extra={"animal": aname}))
        if atile.get("yield_units", 0) > 0:
            tasks.append(MicroTask("HARVEST_ANIMAL", pos, utility=840.0, extra={"animal": aname}))
        if atile.get("fertilizer_available", False):
            tasks.append(MicroTask("COLLECT_FERTILIZER", pos, utility=820.0, extra={"animal": aname}))

    # 8. Planting Chores on Empty Tiles
    wheat_seeds_avail = seeds.get("WHEAT", 0)
    nw_wheat_empty = [p for p in NW_WHEAT_TILES if p in empty_tiles and p not in PASTURE_SET]
    if day <= 26 and wheat_seeds_avail > 0 and nw_wheat_empty:
        for p in nw_wheat_empty[:wheat_seeds_avail]:
            tasks.append(MicroTask("PLANT_CROP", p, utility=810.0, extra="WHEAT"))

    straw_seeds_avail = seeds.get("STRAWBERRY", 0)
    straw_empty_tiles = [p for p in empty_tiles if p not in NW_WHEAT_TILES and p not in PASTURE_SET]
    if day <= 26 and straw_seeds_avail > 0 and straw_empty_tiles:
        for p in straw_empty_tiles[:straw_seeds_avail]:
            extra_logit = 0.0
            if crop_heatmaps is not None:
                px, py = p
                if 0 <= py < 10 and 0 <= px < 10:
                    extra_logit = float(crop_heatmaps[3, py, px])
            tasks.append(MicroTask("PLANT_CROP", p, utility=800.0 + lambda_logit * extra_logit, extra="STRAWBERRY"))

    # 9. Digging Weeds
    for wx, wy in weed_tiles:
        tasks.append(MicroTask("DIG_WEED", (wx, wy), utility=350.0))

    all_workers = [(0, farmer_pos, inventories[0] if len(inventories) > 0 else {})]
    for h_idx, h_pos in enumerate(hands_pos):
        w_idx = h_idx + 1
        w_inv = inventories[w_idx] if w_idx < len(inventories) else {}
        all_workers.append((w_idx, h_pos, w_inv))

    num_workers = len(all_workers)
    num_tasks = len(tasks)
    worker_actions = [["PASS"] for _ in range(num_workers)]

    if num_workers > 0 and num_tasks > 0:
        cost_matrix = np.zeros((num_workers, num_tasks), dtype=np.float64)

        for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
            is_at_shed = w_pos in SHED_TILES
            has_cow = w_inv.get("COW", 0) > 0
            has_sheep = w_inv.get("SHEEP", 0) > 0
            has_animal = has_cow or has_sheep
            has_wheat = w_inv.get("WHEAT", 0) > 0
            has_fert = w_inv.get("FERTILIZER", 0) > 0

            for j, task in enumerate(tasks):
                t_pos = task.pos
                raw_dist = get_manhattan_dist(w_pos, t_pos)
                effective_dist = float(raw_dist)

                if has_animal:
                    if task.task_type == "PLACE_PASTURE_ANIMAL":
                        effective_dist = float(raw_dist) - 50.0
                    else:
                        cost_matrix[i, j] = 99999.0
                        continue

                elif task.task_type == "PLACE_PASTURE_ANIMAL":
                    if not is_at_shed:
                        dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                        dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                        effective_dist = dist_to_shed + dist_shed_to_task + 2.0

                elif task.task_type == "FEED":
                    if has_wheat:
                        effective_dist = float(raw_dist) - 2.0
                    elif not is_at_shed:
                        dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                        dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                        effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                elif task.task_type == "FERTILIZE_CROP":
                    if has_fert:
                        effective_dist = float(raw_dist) - 2.0
                    elif not is_at_shed:
                        dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                        dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                        effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                cost_matrix[i, j] = (effective_dist * 4.0) - (task.utility * 0.1)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_task_map = {row: col for row, col in zip(row_ind, col_ind)}
    else:
        assigned_task_map = {}

    reserved_destinations = set()
    unmoved_units = set(range(num_workers))
    units = [w[1] for w in all_workers]

    # Standing operations
    for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
        if w_pos in reserved_destinations:
            continue

        wx, wy = w_pos
        w_tile = tiles[wy][wx] if wy < len(tiles) and wx < len(tiles[wy]) else None
        is_at_shed = w_pos in SHED_TILES
        assigned_task = tasks[assigned_task_map[i]] if i in assigned_task_map else None

        if is_at_shed:
            has_produce = any(w_inv.get(p, 0) > 0 for p in ["STRAWBERRY", "MELON", "MILK", "WOOL", "EGG"])
            has_unwanted_wheat = (w_inv.get("WHEAT", 0) > 0 and (not assigned_task or assigned_task.task_type != "FEED"))
            has_unwanted_fert = (w_inv.get("FERTILIZER", 0) > 0 and (not assigned_task or assigned_task.task_type != "FERTILIZE_CROP"))

            if has_produce or has_unwanted_wheat or has_unwanted_fert:
                worker_actions[i] = ["DROP"]
                reserved_destinations.add(w_pos)
                unmoved_units.discard(i)
                continue

            if assigned_task:
                if assigned_task.task_type == "PLACE_PASTURE_ANIMAL" and w_inv.get("SHEEP", 0) == 0 and w_inv.get("COW", 0) == 0:
                    if shed.get("SHEEP", 0) > 0:
                        worker_actions[i] = ["PICKUP", "SHEEP", 1]
                        reserved_destinations.add(w_pos)
                        unmoved_units.discard(i)
                        continue
                    elif shed.get("COW", 0) > 0:
                        worker_actions[i] = ["PICKUP", "COW", 1]
                        reserved_destinations.add(w_pos)
                        unmoved_units.discard(i)
                        continue

                elif assigned_task.task_type == "FEED" and w_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    worker_actions[i] = ["PICKUP", "WHEAT", min(4, shed.get("WHEAT", 0))]
                    reserved_destinations.add(w_pos)
                    unmoved_units.discard(i)
                    continue

                elif assigned_task.task_type == "FERTILIZE_CROP" and w_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    worker_actions[i] = ["PICKUP", "FERTILIZER", min(4, shed.get("FERTILIZER", 0))]
                    reserved_destinations.add(w_pos)
                    unmoved_units.discard(i)
                    continue

        if isinstance(w_tile, dict):
            k = w_tile.get("kind")
            animal = w_tile.get("animal")

            if k in ["PASTURE", "COOP"]:
                if animal is not None:
                    if not w_tile.get("fed_today", False) and w_inv.get("WHEAT", 0) > 0:
                        worker_actions[i] = ["FEED"]
                        reserved_destinations.add(w_pos)
                        unmoved_units.discard(i)
                        continue
                    if not w_tile.get("cared_today", False):
                        worker_actions[i] = ["CARE"]
                        reserved_destinations.add(w_pos)
                        unmoved_units.discard(i)
                        continue
                    if w_tile.get("fertilizer_available", False):
                        worker_actions[i] = ["COLLECT_FERTILIZER"]
                        reserved_destinations.add(w_pos)
                        unmoved_units.discard(i)
                        continue
                    if w_tile.get("yield_units", 0) > 0:
                        worker_actions[i] = ["HARVEST"]
                        reserved_destinations.add(w_pos)
                        unmoved_units.discard(i)
                        continue
                else:
                    if assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "PLACE_PASTURE_ANIMAL":
                        if w_inv.get("SHEEP", 0) > 0:
                            worker_actions[i] = ["PLACE", "SHEEP"]
                            reserved_destinations.add(w_pos)
                            unmoved_units.discard(i)
                            continue
                        elif w_inv.get("COW", 0) > 0:
                            worker_actions[i] = ["PLACE", "COW"]
                            reserved_destinations.add(w_pos)
                            unmoved_units.discard(i)
                            continue

            elif k == "PLANT":
                crop = w_tile.get("crop", "WHEAT")
                age = day - w_tile.get("planted_day", day)
                yield_u = w_tile.get("yield_units", 0)
                fert_until = w_tile.get("fertilized_until_day", -1)

                if not w_tile.get("watered_today", False):
                    worker_actions[i] = ["WATER"]
                    reserved_destinations.add(w_pos)
                    unmoved_units.discard(i)
                    continue

                if fert_until < day and w_inv.get("FERTILIZER", 0) > 0:
                    if (crop == "MELON" and age <= 10) or (crop == "STRAWBERRY" and day <= 27):
                        worker_actions[i] = ["FERTILIZE"]
                        reserved_destinations.add(w_pos)
                        unmoved_units.discard(i)
                        continue

                if (crop == "WHEAT" and (age >= 4 or (day >= 28 and yield_u > 0))) or \
                   (crop == "MELON" and (age >= 10 or (day >= 28 and yield_u > 0))) or \
                   (crop == "STRAWBERRY" and yield_u > 0) or \
                   (yield_u > 0 or age >= 4):
                    worker_actions[i] = ["HARVEST"]
                    reserved_destinations.add(w_pos)
                    unmoved_units.discard(i)
                    continue

            elif k == "WEED":
                worker_actions[i] = ["DIG"]
                reserved_destinations.add(w_pos)
                unmoved_units.discard(i)
                continue

        elif w_tile is None:
            if assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "BUILD_PASTURE":
                worker_actions[i] = ["BUILD_PASTURE"]
                reserved_destinations.add(w_pos)
                unmoved_units.discard(i)
                continue

            if (assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "PLANT_CROP") or (assigned_task is None and day <= 26):
                if w_pos not in PASTURE_SET:
                    if day == 0:
                        if seeds.get("MELON", 0) > 0:
                            worker_actions[i] = ["PLANT", "MELON"]
                            seeds["MELON"] -= 1
                            reserved_destinations.add(w_pos)
                            unmoved_units.discard(i)
                            continue
                        elif seeds.get("WHEAT", 0) > 0:
                            worker_actions[i] = ["PLANT", "WHEAT"]
                            seeds["WHEAT"] -= 1
                            reserved_destinations.add(w_pos)
                            unmoved_units.discard(i)
                            continue
                    else:
                        if w_pos in NW_WHEAT_TILES:
                            if seeds.get("WHEAT", 0) > 0:
                                worker_actions[i] = ["PLANT", "WHEAT"]
                                seeds["WHEAT"] -= 1
                                reserved_destinations.add(w_pos)
                                unmoved_units.discard(i)
                                continue
                        else:
                            if seeds.get("STRAWBERRY", 0) > 0:
                                worker_actions[i] = ["PLANT", "STRAWBERRY"]
                                seeds["STRAWBERRY"] -= 1
                                reserved_destinations.add(w_pos)
                                unmoved_units.discard(i)
                                continue
                            elif seeds.get("WHEAT", 0) > 0:
                                worker_actions[i] = ["PLANT", "WHEAT"]
                                seeds["WHEAT"] -= 1
                                reserved_destinations.add(w_pos)
                                unmoved_units.discard(i)
                                continue

    # Moving units (Collision-Free Routing)
    for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
        if i not in unmoved_units:
            continue

        assigned_task = tasks[assigned_task_map[i]] if i in assigned_task_map else None
        is_at_shed = w_pos in SHED_TILES
        unmoved_positions = {units[v] for v in unmoved_units if v != i}

        target_pos = None
        if assigned_task:
            needs_shed = False
            if assigned_task.task_type == "FEED" and w_inv.get("WHEAT", 0) == 0:
                needs_shed = True
            elif assigned_task.task_type == "FERTILIZE_CROP" and w_inv.get("FERTILIZER", 0) == 0:
                needs_shed = True
            elif assigned_task.task_type == "PLACE_PASTURE_ANIMAL" and (w_inv.get("COW", 0) == 0 and w_inv.get("SHEEP", 0) == 0):
                needs_shed = True

            if needs_shed and not is_at_shed:
                target_pos = min(SHED_TILES, key=lambda s: get_manhattan_dist(w_pos, s))
            else:
                target_pos = assigned_task.pos
        else:
            if not is_at_shed:
                target_pos = min(SHED_TILES, key=lambda s: get_manhattan_dist(w_pos, s))

        if target_pos is not None:
            direction, next_pos = get_step_towards(w_pos, target_pos, reserved_destinations, unmoved_positions)
            worker_actions[i] = [direction]
            reserved_destinations.add(next_pos)
        else:
            direction, next_pos = get_step_towards(w_pos, (4, 4), reserved_destinations, unmoved_positions)
            worker_actions[i] = [direction]
            reserved_destinations.add(next_pos)

        unmoved_units.discard(i)

    farmer_action = worker_actions[0] if worker_actions else ["PASS"]
    hands_actions = worker_actions[1:] if len(worker_actions) > 1 else []

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
                f32_state = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
                self.network.load_state_dict(f32_state)
            except Exception:
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

        workforce_mask = torch.zeros(13, dtype=torch.float32)
        has_valid_hire = False
        for k in range(13):
            if k >= num_workers and (k * 20.0 <= money or k == num_workers):
                workforce_mask[k] = 1.0
                has_valid_hire = True
            elif k == num_workers:
                workforce_mask[k] = 1.0
                has_valid_hire = True
        if not has_valid_hire and 0 <= num_workers < 13:
            workforce_mask[num_workers] = 1.0

        step = int(obs.get("step", 0))
        tau = get_gumbel_temperature(step)
        workforce_idx = sample_gumbel_action(workforce_logits, workforce_mask, temperature=tau)

        market_orders = apply_market_guardrails(
            market_fractions=market_fractions,
            obs=obs,
            seed_replenish_logits=seed_replenish_logits,
            land_expand_logit=land_expand_logit,
            workforce_logit_idx=workforce_idx,
            livestock_quotas=livestock_quotas,
        )

        farmer_act, hands_acts = solve_micro_actions(
            obs=obs,
            crop_heatmaps=crop_heatmaps,
            livestock_quotas=livestock_quotas,
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
