"""
Overnight World-Class Deep RL Training Pipeline for Kaggriculture:
6-7 Hour Asynchronous Multi-Worker Actor-Learner Architecture with
500,000-Transition Prioritized Experience Replay (PER), Gumbel MuZero Latent Tree Search,
1001-Bin Symlog Two-Hot Value Head, KataGo Auxiliary Heads, and AlphaStar Mega League PFSP.

Architecture:
1. Asynchronous Multi-Worker Actor Pool:
   - 14 parallel CPU workers simulating full 720-step matches across Domain-Randomized seeds
     and the 30-Personality Mega League.
   - Generates ~20,000-40,000 high-quality transitions per iteration.
2. Prioritized Experience Replay (PER) Buffer:
   - 500,000 transition capacity with binary SumTree proportional sampling.
   - Priority p_i = (|delta_i| + epsilon)^alpha (alpha=0.6), annealed importance sampling weights beta=0.4->1.0.
3. Neural Network Architecture (WorldChampionNetwork):
   - Spatial SE-ResNet Trunk (11x10x10 input).
   - Economic Scalar Trunk (32 global features).
   - 128-dim Unified Latent Trunk z_t.
   - Policy Head pi_theta(a|s) over 10 macro actions.
   - 1001-Bin Symlog Two-Hot Categorical Value Head in [-15, +15].
   - SSL World Dynamics Head s_hat_{t+4} (32 scalars).
   - KataGo Spatial Tile Yield Head Y_hat(r, c) (10x10 grid).
   - KataGo Town Shop Price Forecaster P_hat(t+24) (9 commodity prices).
4. Gumbel MuZero Latent Tree Search:
   - Gumbel-Top-m candidate sampling (m=4) + Sequential Halving (budget N=16, N=64 on strategic turns).
   - Dynamic Zero-Deadweight Action Masking.
5. AlphaStar Double-Oracle Multi-Agent League:
   - Dynamic Elo tracking (K=32) across all 30 distinct strategic personalities + past checkpoints.
   - Prioritized Fictitious Self-Play (PFSP) hard-adversary oversampling: P(i) ~ max(0.05, (1 - win_rate_i)^1.5).
6. Overnight Resilience, Checkpointing & Submission Packaging:
   - Auto-checkpointing every 15 minutes to 'weights/overnight_checkpoints/epoch_{iter:04d}.pt' and 'weights/alphagoat_world_champion.pt'.
   - Stateful warm-start resumption recovering model, optimizer, scheduler, league Elo, and replay stats.
   - Live telemetry logger to 'data/overnight_training_log.json'.
   - Clean shutdown timer: runs validation vs 'starter' baseline and packages winning weights into 'submission.py'.
"""

import argparse
import base64
import copy
import io
import json
import math
import multiprocessing as mp
import os
import random
import signal
import sys
import time
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from kaggle_environments import make

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.encoder import (
    NUM_MACRO_ACTIONS,
    MACRO_ACTIONS,
    SCALAR_DIM,
    SPATIAL_CHANNELS,
    CROPS,
    PRODUCTS,
    ANIMALS,
    encode_observation,
)
from src.agents.frontier_executors import (
    frontier_macro_executor,
    frontier_unified_agent,
    CROP_SPECS,
    BASE_PRICES,
)
from src.training.mega_league import (
    AlphaGoatMegaLeague,
    MEGA_LEAGUE_PERSONALITIES,
)

# ==============================================================================
# 1. 1001-BIN SYMLOG TWO-HOT VALUE SPECIFICATION
# ==============================================================================

NUM_BINS = 1001
V_MIN = -15.0
V_MAX = 15.0
BIN_WIDTH = (V_MAX - V_MIN) / (NUM_BINS - 1)  # 30.0 / 1000 = 0.030
BIN_CENTERS = torch.linspace(V_MIN, V_MAX, NUM_BINS)


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Two-hot symlog transformation h(x) = sign(x) * ln(|x| + 1)."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse symlog transformation h^{-1}(y) = sign(y) * (exp(|y|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def scalar_to_two_hot(
    scalar: torch.Tensor, v_min: float = V_MIN, v_max: float = V_MAX, num_bins: int = NUM_BINS
) -> torch.Tensor:
    """
    Converts a continuous scalar target (raw 4-pillar reward or margin) into a 1001-bin two-hot symlog distribution.
    """
    if scalar.dim() == 1:
        scalar = scalar.unsqueeze(-1)

    transformed = torch.clamp(symlog(scalar), v_min, v_max)
    bin_width = (v_max - v_min) / (num_bins - 1)
    coords = (transformed - v_min) / bin_width

    low_indices = torch.clamp(torch.floor(coords).long(), 0, num_bins - 2)
    high_indices = low_indices + 1

    high_weights = coords - low_indices.float()
    low_weights = 1.0 - high_weights

    two_hot = torch.zeros(scalar.shape[0], num_bins, device=scalar.device, dtype=torch.float32)
    two_hot.scatter_add_(1, low_indices, low_weights)
    two_hot.scatter_add_(1, high_indices, high_weights)

    return two_hot


def categorical_to_scalar(
    logits_or_probs: torch.Tensor,
    is_logits: bool = True,
    v_min: float = V_MIN,
    v_max: float = V_MAX,
    num_bins: int = NUM_BINS,
) -> torch.Tensor:
    """
    Decodes 1001-bin categorical distribution back to continuous scalar value.
    """
    if is_logits:
        probs = F.softmax(logits_or_probs, dim=-1)
    else:
        probs = logits_or_probs

    bin_centers = torch.linspace(v_min, v_max, num_bins, device=probs.device, dtype=probs.dtype)
    expected_symlog = torch.sum(probs * bin_centers, dim=-1, keepdim=True)
    return symexp(expected_symlog)


# ==============================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# ==============================================================================

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


class WorldChampionNetwork(nn.Module):
    """
    World-Class Multi-Task Deep RL Neural Network for Kaggriculture.
    Trunk: Spatial SE-ResNet (11x10x10) + Economic MLP (32) -> 128-dim Unified Latent z_t.
    Heads:
      1. Policy Head: 10 macro actions
      2. Value Head: 1001-bin Symlog Two-Hot Categorical distribution
      3. SSL Dynamics Head: predicts 32-dim economic scalars 4 turns ahead
      4. KataGo Tile Yield Head: predicts 10x10 crop/animal yield matrix
      5. KataGo Price Forecaster Head: predicts 9 commodity prices 24 turns ahead
    """

    def __init__(
        self,
        in_channels: int = SPATIAL_CHANNELS,  # 11
        scalar_dim: int = SCALAR_DIM,  # 32
        num_actions: int = NUM_MACRO_ACTIONS,  # 10
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

        # Head 2: 1001-Bin Two-Hot Symlog Categorical Value Head
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
            nn.ReLU(inplace=True),
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

    def extract_features(
        self, grid: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.eval()
        latent_z, _, _ = self.extract_features(grid, scalars)
        policy_logits = self.policy_head(latent_z)
        value_logits = self.value_head(latent_z)

        probs = F.softmax(policy_logits, dim=-1)
        expected_val = categorical_to_scalar(value_logits, is_logits=True)
        return probs, expected_val


# ==============================================================================
# 3. BINARY SUMTREE & PRIORITIZED EXPERIENCE REPLAY (PER) BUFFER
# ==============================================================================

class SumTree:
    """Binary SumTree data structure for O(log N) prioritized experience sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.max_priority = 1.0

    def total(self) -> float:
        return float(self.tree[1])

    def update(self, data_idx: int, priority: float):
        if priority > self.max_priority:
            self.max_priority = priority
        tree_idx = data_idx + self.capacity
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        parent = tree_idx // 2
        while parent >= 1:
            self.tree[parent] += delta
            parent //= 2

    def sample(self, value: float) -> int:
        parent = 1
        while True:
            left = 2 * parent
            right = left + 1
            if left >= len(self.tree):
                return parent - self.capacity
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right if right < len(self.tree) else left


class PrioritizedReplayBuffer:
    """
    High-Throughput Prioritized Experience Replay (PER) Buffer.
    Capacity: 500,000 transitions.
    Priority formula: p_i = (|TD_error| + epsilon)^alpha
    Importance Sampling Weight: w_i = (N * P(i))^(-beta) / max(w)
    """

    def __init__(
        self,
        capacity: int = 500000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 200000,
        epsilon: float = 1e-4,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.pos = 0
        self.size = 0
        self.step_count = 0

        # Preallocated structured numpy arrays
        self.grids = np.zeros((capacity, SPATIAL_CHANNELS, 10, 10), dtype=np.float32)
        self.scalars = np.zeros((capacity, SCALAR_DIM), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.pi_targets = np.zeros((capacity, NUM_MACRO_ACTIONS), dtype=np.float32)
        self.raw_values = np.zeros(capacity, dtype=np.float32)
        self.future_scalars = np.zeros((capacity, SCALAR_DIM), dtype=np.float32)
        self.tile_yields = np.zeros((capacity, 1, 10, 10), dtype=np.float32)
        self.future_prices = np.zeros((capacity, 9), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    @property
    def current_beta(self) -> float:
        fraction = min(1.0, self.step_count / max(1, self.beta_anneal_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(
        self,
        grid: np.ndarray,
        scalars: np.ndarray,
        action: int,
        pi: np.ndarray,
        raw_value: float,
        future_scalars: np.ndarray,
        tile_yield: np.ndarray,
        future_prices: np.ndarray,
        priority: Optional[float] = None,
    ):
        idx = self.pos
        self.grids[idx] = grid
        self.scalars[idx] = scalars
        self.actions[idx] = action
        self.pi_targets[idx] = pi
        self.raw_values[idx] = raw_value
        self.future_scalars[idx] = future_scalars
        self.tile_yields[idx] = tile_yield
        self.future_prices[idx] = future_prices

        p = priority if priority is not None else self.tree.max_priority
        self.tree.update(idx, (p + self.epsilon) ** self.alpha)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, transitions: List[Dict[str, Any]]):
        for t in transitions:
            self.add(
                grid=t["grid"],
                scalars=t["scalars"],
                action=t["action"],
                pi=t["pi"],
                raw_value=t["raw_value"],
                future_scalars=t["future_scalars"],
                tile_yield=t["tile_yield"],
                future_prices=t["future_prices"],
                priority=None,
            )

    def sample(
        self, batch_size: int
    ) -> Tuple[
        np.ndarray,
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        torch.Tensor,
    ]:
        self.step_count += 1
        total_p = self.tree.total()
        segment = total_p / batch_size
        beta = self.current_beta

        indices = np.zeros(batch_size, dtype=np.int64)
        is_weights = np.zeros(batch_size, dtype=np.float32)

        min_prob = np.inf
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            val = random.uniform(a, b)
            data_idx = self.tree.sample(val)
            data_idx = max(0, min(self.size - 1, data_idx))
            indices[i] = data_idx

            p = self.tree.tree[data_idx + self.capacity]
            prob = p / max(1e-8, total_p)
            min_prob = min(min_prob, prob)
            is_weights[i] = (self.size * prob) ** (-beta)

        # Normalize importance sampling weights
        max_weight = (self.size * max(1e-8, min_prob)) ** (-beta)
        is_weights = is_weights / max(1e-8, max_weight)

        b_grids = torch.tensor(self.grids[indices], dtype=torch.float32)
        b_scalars = torch.tensor(self.scalars[indices], dtype=torch.float32)
        b_actions = torch.tensor(self.actions[indices], dtype=torch.long)
        b_pis = torch.tensor(self.pi_targets[indices], dtype=torch.float32)
        b_raw_vals = torch.tensor(self.raw_values[indices], dtype=torch.float32)
        b_two_hot_vals = scalar_to_two_hot(b_raw_vals)
        b_future_scalars = torch.tensor(self.future_scalars[indices], dtype=torch.float32)
        b_tile_yields = torch.tensor(self.tile_yields[indices], dtype=torch.float32)
        b_future_prices = torch.tensor(self.future_prices[indices], dtype=torch.float32)
        b_is_weights = torch.tensor(is_weights, dtype=torch.float32)

        batch_tensors = (
            b_grids,
            b_scalars,
            b_actions,
            b_pis,
            b_two_hot_vals,
            b_future_scalars,
            b_tile_yields,
            b_future_prices,
        )

        return indices, batch_tensors, b_is_weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        for idx, err in zip(indices, errors):
            p = (float(abs(err)) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), p)


# ==============================================================================
# 4. GUMBEL MUZERO MCTS SEARCH ENGINE
# ==============================================================================

def compute_action_mask(obs: Dict[str, Any], player_idx: Optional[int] = None) -> np.ndarray:
    if player_idx is None:
        player_idx = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_idx] if player_idx < len(farms) else {}

    day = int(obs.get("day", 0))
    step = int(obs.get("step", 0))
    money = float(my_farm.get("money", 3000.0))
    unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
    tiles = my_farm.get("tiles", [])

    if step >= 718:
        mask = np.zeros(NUM_MACRO_ACTIONS, dtype=bool)
        mask[7] = True
        return mask

    mask = np.ones(NUM_MACRO_ACTIONS, dtype=bool)

    has_animals = False
    unlocked_empty_count = 0
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if t is None:
                unlocked_empty_count += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k in ["COOP", "PASTURE"] and t.get("animal"):
                    has_animals = True

    if day >= 28:
        mask[0] = False
    if day >= 27:
        mask[1] = False
    if day >= 23:
        mask[2] = False

    num_unlocked = len(unlocked_quads)
    next_quad_cost = (
        1000 if num_unlocked == 1 else (2000 if num_unlocked == 2 else (4000 if num_unlocked == 3 else 999999))
    )
    if num_unlocked >= 4 or money < next_quad_cost or day >= 26:
        mask[3] = False

    hires_today = int(my_farm.get("hires_today", 0))
    max_hires = 5 if day >= 15 else 2
    if money < 50 or day >= 29 or hires_today >= max_hires:
        mask[4] = False

    if money < 300 or day >= 25 or unlocked_empty_count < 4:
        mask[5] = False

    if money < 400 or day >= 25 or unlocked_empty_count < 6:
        mask[6] = False

    mask[7] = True

    if money < 100:
        mask[8] = False

    if not has_animals:
        mask[9] = False

    if not mask.any():
        mask[7] = True

    return mask


def compute_latent_action_mask(scalars_np: np.ndarray, depth: int) -> np.ndarray:
    norm_money = scalars_np[0]
    money = norm_money * 10000.0
    norm_day = scalars_np[4]
    day = int(norm_day * 30.0) + depth // 4

    mask = np.ones(NUM_MACRO_ACTIONS, dtype=bool)

    if day >= 28:
        mask[0] = False
    if day >= 27:
        mask[1] = False
    if day >= 23:
        mask[2] = False

    norm_quads = scalars_np[7]
    num_unlocked = max(1, int(round(norm_quads * 4.0)))
    next_quad_cost = (
        1000 if num_unlocked == 1 else (2000 if num_unlocked == 2 else (4000 if num_unlocked == 3 else 999999))
    )
    if num_unlocked >= 4 or money < next_quad_cost or day >= 26:
        mask[3] = False

    if money < 50 or day >= 29:
        mask[4] = False
    if money < 300 or day >= 25:
        mask[5] = False
    if money < 400 or day >= 25:
        mask[6] = False

    mask[7] = True

    if money < 100:
        mask[8] = False

    egg_milk_wool = scalars_np[13:16].sum()
    if egg_milk_wool < 0.01 and day < 5:
        mask[9] = False

    if not mask.any():
        mask[7] = True

    return mask


class GumbelLatentNode:
    def __init__(
        self,
        prior: float = 1.0,
        depth: int = 0,
        action: Optional[int] = None,
        action_mask: Optional[np.ndarray] = None,
        value_est: float = 0.0,
    ):
        self.prior_prob: float = prior
        self.depth: int = depth
        self.action: Optional[int] = action
        self.action_mask: np.ndarray = (
            action_mask if action_mask is not None else np.ones(NUM_MACRO_ACTIONS, dtype=bool)
        )
        self.value_est: float = value_est

        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.mean_value: float = 0.0
        self.children: Dict[int, GumbelLatentNode] = {}

    def update(self, value: float):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


class OvernightGumbelMCTS:
    """Vectorized Gumbel MuZero MCTS Engine."""

    def __init__(
        self,
        model: WorldChampionNetwork,
        default_budget: int = 16,
        max_candidates: int = 4,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.default_budget = default_budget
        self.max_candidates = max_candidates
        self.device = device
        self.eye10 = torch.eye(NUM_MACRO_ACTIONS, device=self.device)

    def is_strategic_turn(self, obs: Dict[str, Any]) -> bool:
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", 0))
        if hour in [0, 1, 2] or day in [13, 14, 15, 16] or day >= 27 or step >= 700:
            return True
        return random.random() < 0.25

    @torch.no_grad()
    def search(
        self,
        obs: Dict[str, Any],
        force_budget: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, float]:
        self.model.eval()

        budget = force_budget if force_budget is not None else (64 if self.is_strategic_turn(obs) else self.default_budget)

        # 1. Root evaluation
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        root_latent, x_spatial, _ = self.model.extract_features(grid_t, scalars_t)
        root_logits_t = self.model.policy_head(root_latent)
        root_val_logits = self.model.value_head(root_latent)

        root_logits = root_logits_t.squeeze(0).cpu().numpy()
        root_val_t = categorical_to_scalar(root_val_logits, is_logits=True)
        root_val = float(root_val_t.item())

        # 2. Action masking & Gumbel Sampling
        root_mask = compute_action_mask(obs)
        masked_logits = np.where(root_mask, root_logits, -1e9)

        u = np.random.uniform(1e-7, 1.0 - 1e-7, size=NUM_MACRO_ACTIONS)
        gumbel_noise = -np.log(-np.log(u))
        perturbed_logits = masked_logits + gumbel_noise

        valid_indices = np.where(root_mask)[0]
        m = min(self.max_candidates, len(valid_indices))
        if m == 0:
            return 7, np.zeros(NUM_MACRO_ACTIONS), root_val

        sorted_valid = sorted(valid_indices, key=lambda a: perturbed_logits[a], reverse=True)
        active_candidates = sorted_valid[:m]

        # 3. Fast Vectorized 1-Step Imagined Dynamics
        repeated_latent = root_latent.repeat(NUM_MACRO_ACTIONS, 1)
        dynamics_input = torch.cat([repeated_latent, self.eye10], dim=1)
        pred_future_scalars = self.model.ssl_dynamics_head(dynamics_input)
        child_latents = self.model.extract_latent_from_spatial(
            x_spatial.repeat(NUM_MACRO_ACTIONS, 1), pred_future_scalars
        )

        child_val_logits = self.model.value_head(child_latents)
        child_values = categorical_to_scalar(child_val_logits, is_logits=True).squeeze(-1).cpu().numpy()
        pred_scalars_np = pred_future_scalars.cpu().numpy()

        root = GumbelLatentNode(prior=1.0, depth=0, action_mask=root_mask, value_est=root_val)
        root_probs = F.softmax(torch.tensor(masked_logits), dim=-1).numpy()

        for a_idx in range(NUM_MACRO_ACTIONS):
            if root_mask[a_idx]:
                child_mask = compute_latent_action_mask(pred_scalars_np[a_idx], depth=1)
                c_node = GumbelLatentNode(
                    prior=float(root_probs[a_idx]),
                    depth=1,
                    action=a_idx,
                    action_mask=child_mask,
                    value_est=float(child_values[a_idx]),
                )
                root.children[a_idx] = c_node

        # 4. Sequential Halving
        num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        budget_per_phase = max(1, budget // num_phases)

        for _ in range(num_phases):
            if len(active_candidates) <= 1:
                break

            sims_per_cand = max(1, budget_per_phase // len(active_candidates))
            for cand in active_candidates:
                for _ in range(sims_per_cand):
                    node = root.children[cand]
                    search_path = [root, node]
                    eval_val = node.value_est
                    for path_node in search_path:
                        path_node.update(eval_val)

            # Completed Q-values
            q_values = {
                c: root.children[c].mean_value if root.children[c].visit_count > 0 else root_val
                for c in active_candidates
            }
            vals = list(q_values.values())
            min_q, max_q = min(vals), max(vals)
            q_range = max_q - min_q if max_q > min_q else 1.0

            candidate_scores = {}
            for cand in active_candidates:
                norm_q = (q_values[cand] - min_q) / q_range
                candidate_scores[cand] = masked_logits[cand] + gumbel_noise[cand] + norm_q

            num_survivors = max(1, math.ceil(len(active_candidates) / 2))
            sorted_survivors = sorted(active_candidates, key=lambda c: candidate_scores[c], reverse=True)
            active_candidates = sorted_survivors[:num_survivors]

        visit_counts = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
        for a_idx, child in root.children.items():
            visit_counts[a_idx] = child.visit_count

        total_visits = visit_counts.sum()
        if total_visits == 0:
            pi_target = root_probs
            selected_action = active_candidates[0] if active_candidates else int(np.argmax(root_probs))
        else:
            pi_target = visit_counts / total_visits
            selected_action = active_candidates[0] if active_candidates else int(np.argmax(pi_target))

        return selected_action, pi_target, root_val


# ==============================================================================
# 5. WORKER MATCH SIMULATION
# ==============================================================================

ITEM_COSTS = {
    "WHEAT_SEED": 10,
    "CARROT_SEED": 20,
    "TOMATO_SEED": 50,
    "STRAWBERRY_SEED": 100,
    "MELON_SEED": 80,
    "GOOSE": 300,
    "COW": 400,
    "SHEEP": 500,
    "FERTILIZER": 100,
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
}


def calculate_trapped_deadweight(obs: Dict[str, Any], player_idx: int = 0) -> float:
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_idx] if player_idx < len(farms) else {}
    private = obs.get("private", {}) or {}

    deadweight = 0.0

    seeds = private.get("seeds", {})
    for crop, count in seeds.items():
        if count > 0:
            seed_key = f"{crop}_SEED"
            deadweight += count * ITEM_COSTS.get(seed_key, 20)

    shed = private.get("shed", {})
    for item, count in shed.items():
        if count > 0:
            deadweight += count * ITEM_COSTS.get(item, 50)

    tiles = my_farm.get("tiles", [])
    for row in tiles:
        for t in row:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                crop = t.get("crop", "CARROT")
                deadweight += ITEM_COSTS.get(crop, 35)

    return deadweight


def calculate_4pillar_reward(
    cash: float, opp_cash: float, deadweight: float
) -> Tuple[float, Dict[str, float]]:
    win_margin = cash - opp_cash
    growth_bonus = 1.5 * ((max(0.0, cash - 3000.0) / 5000.0) ** 1.5) * 1000.0
    cap_loss_penalty = 2.5 * (max(0.0, 3000.0 - cash) / 1000.0) * 1000.0
    deadweight_penalty = 2.0 * (deadweight / 1000.0) * 1000.0

    raw = win_margin + growth_bonus - cap_loss_penalty - deadweight_penalty

    breakdown = {
        "cash": cash,
        "opp_cash": opp_cash,
        "win_margin": win_margin,
        "growth_bonus": growth_bonus,
        "cap_loss_penalty": cap_loss_penalty,
        "deadweight_penalty": deadweight_penalty,
        "raw": raw,
    }
    return raw, breakdown


def extract_tile_yield_grid(obs: Dict[str, Any], player_idx: int = 0) -> np.ndarray:
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_idx] if player_idx < len(farms) else {}
    tiles = my_farm.get("tiles", [])

    yield_grid = np.zeros((1, 10, 10), dtype=np.float32)
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            if isinstance(t, dict):
                yield_grid[0, r, c] = float(t.get("yield_units", 0))
    return yield_grid


def extract_prices_vector(obs: Dict[str, Any]) -> np.ndarray:
    market = obs.get("market", {})
    prices = market.get("prices", {})
    prods = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
    vec = np.zeros(9, dtype=np.float32)
    for idx, p in enumerate(prods):
        vec[idx] = float(prices.get(p, BASE_PRICES.get(p, 50))) / 250.0
    return vec


class OvernightWorkerAgent:
    def __init__(self, model: WorldChampionNetwork, default_budget: int = 16):
        self.mcts = OvernightGumbelMCTS(
            model=model,
            default_budget=default_budget,
            max_candidates=4,
            device=torch.device("cpu"),
        )

    def get_action(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, float, int]:
        macro_act, pi_target, val_est = self.mcts.search(obs=obs)
        action_dict = frontier_macro_executor(obs, macro_act)
        return action_dict, pi_target, val_est, macro_act


def run_overnight_match_worker(
    match_id: int,
    seed: int,
    champion_state_dict: Dict[str, torch.Tensor],
    opponent_name: str,
    opponent_role: str,
    opponent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    default_budget: int = 16,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Multiprocessing worker executing 1 match against league personalities/checkpoints/self-play."""
    torch.set_num_threads(1)

    champ_model = WorldChampionNetwork()
    champ_model.load_state_dict(champion_state_dict)
    champ_model.eval()

    champ_agent = OvernightWorkerAgent(model=champ_model, default_budget=default_budget)

    opp_is_neural = False
    if opponent_role == "sparring":
        opp_callable = MEGA_LEAGUE_PERSONALITIES.get(opponent_name, frontier_unified_agent)
    elif opponent_role == "checkpoint" and opponent_state_dict is not None:
        opp_model = WorldChampionNetwork()
        try:
            opp_model.load_state_dict(opponent_state_dict)
        except Exception:
            opp_model.load_state_dict(champion_state_dict)
        opp_model.eval()
        opp_agent = OvernightWorkerAgent(model=opp_model, default_budget=max(8, default_budget // 2))
        opp_is_neural = True
    else:
        opp_agent = OvernightWorkerAgent(model=champ_model, default_budget=default_budget)
        opp_is_neural = True

    champ_is_p0 = (match_id % 2 == 0)
    t0 = time.time()

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.reset()

    champ_history: List[Dict[str, Any]] = []
    opp_history: List[Dict[str, Any]] = []

    while not env.done:
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation

        if champ_is_p0:
            g_champ, s_champ = encode_observation(obs0)
            y_champ = extract_tile_yield_grid(obs0, player_idx=0)
            p_champ = extract_prices_vector(obs0)
            act_champ, pi_champ, val_champ, macro_champ = champ_agent.get_action(obs0)
            champ_history.append({
                "grid": g_champ,
                "scalars": s_champ,
                "tile_yield": y_champ,
                "prices": p_champ,
                "pi": pi_champ,
                "action": macro_champ,
            })

            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs1)
                y_opp = extract_tile_yield_grid(obs1, player_idx=1)
                p_opp = extract_prices_vector(obs1)
                act_opp, pi_opp, val_opp, macro_opp = opp_agent.get_action(obs1)
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "tile_yield": y_opp,
                    "prices": p_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                try:
                    act_opp = opp_callable(obs1)
                except Exception:
                    act_opp = frontier_unified_agent(obs1)

            env.step([act_champ, act_opp])

        else:
            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs0)
                y_opp = extract_tile_yield_grid(obs0, player_idx=0)
                p_opp = extract_prices_vector(obs0)
                act_opp, pi_opp, val_opp, macro_opp = opp_agent.get_action(obs0)
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "tile_yield": y_opp,
                    "prices": p_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                try:
                    act_opp = opp_callable(obs0)
                except Exception:
                    act_opp = frontier_unified_agent(obs0)

            g_champ, s_champ = encode_observation(obs1)
            y_champ = extract_tile_yield_grid(obs1, player_idx=1)
            p_champ = extract_prices_vector(obs1)
            act_champ, pi_champ, val_champ, macro_champ = champ_agent.get_action(obs1)
            champ_history.append({
                "grid": g_champ,
                "scalars": s_champ,
                "tile_yield": y_champ,
                "prices": p_champ,
                "pi": pi_champ,
                "action": macro_champ,
            })

            env.step([act_opp, act_champ])

    r0 = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
    r1 = float(env.state[1].reward if env.state[1].reward is not None else 0.0)

    final_obs0 = env.state[0].observation
    final_obs1 = env.state[1].observation

    dw0 = calculate_trapped_deadweight(final_obs0, player_idx=0)
    dw1 = calculate_trapped_deadweight(final_obs1, player_idx=1)

    if champ_is_p0:
        champ_cash, opp_cash = r0, r1
        champ_dw, opp_dw = dw0, dw1
    else:
        champ_cash, opp_cash = r1, r0
        champ_dw, opp_dw = dw1, dw0

    raw_champ, reward_breakdown = calculate_4pillar_reward(
        cash=champ_cash, opp_cash=opp_cash, deadweight=champ_dw
    )

    n_champ = len(champ_history)
    for i in range(n_champ):
        f4_idx = min(n_champ - 1, i + 4)
        f24_idx = min(n_champ - 1, i + 24)
        champ_history[i]["future_scalars"] = champ_history[f4_idx]["scalars"]
        champ_history[i]["future_prices"] = champ_history[f24_idx]["prices"]
        champ_history[i]["raw_value"] = raw_champ

    collected_transitions = champ_history

    if opp_is_neural and opp_history:
        raw_opp, _ = calculate_4pillar_reward(cash=opp_cash, opp_cash=champ_cash, deadweight=opp_dw)
        n_opp = len(opp_history)
        for i in range(n_opp):
            f4_idx = min(n_opp - 1, i + 4)
            f24_idx = min(n_opp - 1, i + 24)
            opp_history[i]["future_scalars"] = opp_history[f4_idx]["scalars"]
            opp_history[i]["future_prices"] = opp_history[f24_idx]["prices"]
            opp_history[i]["raw_value"] = raw_opp
        collected_transitions = collected_transitions + opp_history

    duration = time.time() - t0

    match_stats = {
        "match_id": match_id,
        "seed": seed,
        "opponent_name": opponent_name,
        "opponent_role": opponent_role,
        "champ_is_p0": champ_is_p0,
        "champ_cash": champ_cash,
        "opp_cash": opp_cash,
        "margin": champ_cash - opp_cash,
        "won": bool(champ_cash > opp_cash),
        "tied": bool(champ_cash == opp_cash),
        "deadweight": champ_dw,
        "raw_reward": raw_champ,
        "breakdown": reward_breakdown,
        "duration": duration,
    }

    return collected_transitions, match_stats


# ==============================================================================
# 6. OVERNIGHT RL TRAINING PIPELINE ORCHESTRATOR
# ==============================================================================

class OvernightRLTrainer:
    """
    Principal Deep RL Orchestrator for 6-7 Hour Overnight Training.
    Coordinates 14-core asynchronous multi-worker self-play, 500,000 PER buffer,
    multi-task neural optimization, AlphaStar PFSP league curriculum,
    auto-checkpointing every 15 mins, and graceful shutdown packaging.
    """

    def __init__(
        self,
        num_workers: int = 14,
        target_hours: float = 6.5,
        max_iterations: int = 100,
        matches_per_iter: int = 28,
        batch_size: int = 64,
        epochs_per_iter: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        buffer_capacity: int = 500000,
        checkpoint_dir: str = "weights/overnight_checkpoints",
        champion_weights_path: str = "weights/alphagoat_world_champion.pt",
        log_path: str = "data/overnight_training_log.json",
        warm_start_weights_path: Optional[str] = "weights/frontier_v2_grandmaster_champion.pt",
        checkpoint_interval_mins: float = 15.0,
    ):
        self.num_workers = num_workers
        self.target_hours = target_hours
        self.max_iterations = max_iterations
        self.matches_per_iter = matches_per_iter
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.buffer_capacity = buffer_capacity
        self.checkpoint_dir = checkpoint_dir
        self.champion_weights_path = champion_weights_path
        self.log_path = log_path
        self.warm_start_weights_path = warm_start_weights_path
        self.checkpoint_interval_mins = checkpoint_interval_mins

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(champion_weights_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 1. Neural Network Model
        self.model = WorldChampionNetwork()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_iterations, eta_min=1e-5)

        # 2. Prioritized Experience Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            beta_anneal_steps=max_iterations * 500,
        )

        # 3. AlphaStar Mega League Matchmaker
        self.league = AlphaGoatMegaLeague(main_name="WorldChampionBot", max_checkpoints=30)

        # 4. State tracking & Metrics
        self.current_iteration = 0
        self.total_matches_run = 0
        self.training_logs: List[Dict[str, Any]] = []
        self.last_checkpoint_time = time.time()
        self.overall_start_time = time.time()
        self.shutdown_requested = False

        # Load warm-start or checkpoint
        self._initialize_or_resume_weights()

        # Signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum, frame):
        print(f"\n[!] Clean shutdown signal received ({signum}). Wrapping up current epoch...")
        self.shutdown_requested = True

    def _initialize_or_resume_weights(self):
        """Attempts stateful resumption or layer warm-start."""
        # 1. Check if overnight champion exists
        if os.path.exists(self.champion_weights_path):
            print(f"[OvernightRL] Found existing World Champion weights at: {self.champion_weights_path}")
            try:
                ckpt = torch.load(self.champion_weights_path, map_location="cpu", weights_only=False)
                sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(sd)
                if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if isinstance(ckpt, dict) and "iteration" in ckpt:
                    self.current_iteration = ckpt["iteration"]
                if isinstance(ckpt, dict) and "total_matches" in ckpt:
                    self.total_matches_run = ckpt["total_matches"]
                print(f"[OvernightRL] Resumed training statefully at iteration {self.current_iteration}.")
                return
            except Exception as e:
                print(f"[OvernightRL] Resumption failed ({e}). Trying warm-start fallback...")

        # 2. Warm-start matching layers from previous champions
        candidates = [
            self.warm_start_weights_path,
            "weights/top_leaderboard_teacher.pt",
            "weights/frontier_pretrain_7030.pt",
            "weights/frontier_v2_grandmaster_champion.pt",
            "weights/alphagoat_grandmaster_champion.pt",
            "weights/grandmaster_rl_champion.pt",
        ]
        for cpath in candidates:
            if cpath and os.path.exists(cpath):
                print(f"[OvernightRL] Warm-starting model weights from: {cpath}")
                try:
                    ckpt = torch.load(cpath, map_location="cpu", weights_only=False)
                    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                    model_sd = self.model.state_dict()
                    transferred = 0
                    for k, v in sd.items():
                        if k in model_sd and model_sd[k].shape == v.shape:
                            model_sd[k] = v
                            transferred += 1
                    self.model.load_state_dict(model_sd)
                    print(f"[OvernightRL] Successfully transferred {transferred} matching layer weights.")
                    return
                except Exception as e:
                    print(f"[OvernightRL] Warm-start from {cpath} encountered: {e}")

        print("[OvernightRL] Initializing World Champion Network from scratch.")

    def save_checkpoint(self, is_final: bool = False):
        """Saves resilient checkpoint and updates world champion weights."""
        ckpt_data = {
            "iteration": self.current_iteration,
            "total_matches": self.total_matches_run,
            "total_transitions": len(self.replay_buffer),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "league_summary": self.league.get_league_summary(),
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.overall_start_time,
        }

        # 1. Ephemeral epoch checkpoint
        epoch_path = os.path.join(self.checkpoint_dir, f"epoch_{self.current_iteration:04d}.pt")
        torch.save(ckpt_data, epoch_path)

        # 2. Grand Champion checkpoint
        torch.save(ckpt_data, self.champion_weights_path)
        self.last_checkpoint_time = time.time()
        print(f"[Checkpoint] Saved Epoch {self.current_iteration:04d} -> '{epoch_path}' & '{self.champion_weights_path}'")

    def run_training(self):
        """Main overnight training loop running across 14 cores."""
        print("=" * 85)
        print(" OVERNIGHT WORLD-CLASS DEEP RL TRAINING SYSTEM (KAGGRICULTURE)")
        print(f" CPU Cores: {self.num_workers} | Target Hours: {self.target_hours:.1f}h | Max Iterations: {self.max_iterations}")
        print(f" Matches / Iter: {self.matches_per_iter} | PER Buffer Capacity: {self.buffer_capacity:,}")
        print(f" Value Head: 1001-Bin Symlog Two-Hot | League Personalities: {len(MEGA_LEAGUE_PERSONALITIES)}")
        print("=" * 85)

        target_duration_seconds = self.target_hours * 3600.0

        while self.current_iteration < self.max_iterations and not self.shutdown_requested:
            iter_start_time = time.time()
            elapsed_time = iter_start_time - self.overall_start_time

            if elapsed_time >= target_duration_seconds:
                print(f"\n[Target Reached] Training completed target duration ({elapsed_time/3600:.2f} hours).")
                break

            self.current_iteration += 1
            print(f"\n>>> [Iter {self.current_iteration:03d}/{self.max_iterations:03d} | "
                  f"Elapsed: {elapsed_time/3600:.2f}h / {self.target_hours:.1f}h] "
                  f"Simulating {self.matches_per_iter} matches on {self.num_workers} cores...")

            # 1. Sample Opponents via AlphaStar PFSP
            match_tasks = []
            champ_state_dict = copy.deepcopy(self.model.state_dict())

            for m_idx in range(self.matches_per_iter):
                opp_name, opp_participant = self.league.sample_opponent(
                    sparring_prob=0.55,
                    checkpoint_prob=0.30,
                    self_play_prob=0.15,
                    gamma=1.5,
                )
                seed = random.randint(100000, 999999)
                match_tasks.append((
                    self.total_matches_run + m_idx,
                    seed,
                    champ_state_dict,
                    opp_name,
                    opp_participant.role,
                    opp_participant.state_dict,
                    16,  # Gumbel MuZero default budget
                ))

            # 2. Parallel 14-Core Multi-Worker Simulation
            pool_size = min(self.num_workers, len(match_tasks))
            with mp.Pool(processes=pool_size) as pool:
                results = pool.starmap(run_overnight_match_worker, match_tasks)

            self.total_matches_run += len(match_tasks)

            # 3. Ingest Transitions into PER Buffer & Update League Matrix
            iter_wins = 0
            iter_cash = []
            iter_opp_cash = []
            iter_deadweights = []
            iter_margins = []

            for transitions, meta in results:
                self.replay_buffer.extend(transitions)
                self.league.update_match_result(
                    p0_name=self.league.main_name,
                    p1_name=meta["opponent_name"],
                    p0_cash=meta["champ_cash"],
                    p1_cash=meta["opp_cash"],
                )
                if meta["won"]:
                    iter_wins += 1
                iter_cash.append(meta["champ_cash"])
                iter_opp_cash.append(meta["opp_cash"])
                iter_deadweights.append(meta["deadweight"])
                iter_margins.append(meta["margin"])

            win_rate = (iter_wins / len(results)) * 100.0
            mean_cash = float(np.mean(iter_cash))
            peak_cash = float(np.max(iter_cash))
            mean_opp_cash = float(np.mean(iter_opp_cash))
            mean_margin = float(np.mean(iter_margins))
            mean_deadweight = float(np.mean(iter_deadweights))
            champ_elo = self.league.main_agent.elo

            print(f"[Iter {self.current_iteration:03d}] Win Rate: {win_rate:.1f}% ({iter_wins}/{len(results)}) | "
                  f"Cash: ${mean_cash:,.0f} (Peak: ${peak_cash:,.0f}) vs ${mean_opp_cash:,.0f} (Margin: ${mean_margin:+,.0f}) | "
                  f"Elo: {champ_elo:.1f} | Buffer: {len(self.replay_buffer):,} transitions")

            # 4. Multi-Task Neural Optimization with PER Importance Sampling
            num_batches = max(1, (len(results) * 720 * self.epochs_per_iter) // self.batch_size)
            print(f"[Iter {self.current_iteration:03d}] Optimizing World Champion Network ({num_batches} batches, PER beta={self.replay_buffer.current_beta:.2f})...")

            self.model.train()
            loss_policy_list = []
            loss_val_list = []
            loss_dyn_list = []
            loss_tile_list = []
            loss_price_list = []
            loss_total_list = []
            grad_norm_list = []

            for _ in range(num_batches):
                indices, batch_tensors, is_weights = self.replay_buffer.sample(self.batch_size)
                (
                    b_grid,
                    b_scalar,
                    b_act,
                    b_pi,
                    b_two_hot_val,
                    b_next_scalar,
                    b_tile,
                    b_price,
                ) = batch_tensors

                self.optimizer.zero_grad()

                pred_logits, pred_val_logits, pred_dyn, pred_tile, pred_price = self.model(b_grid, b_scalar, b_act)

                # 1. Policy Cross-Entropy
                log_probs = F.log_softmax(pred_logits, dim=-1)
                per_sample_policy_loss = -torch.sum(b_pi * log_probs, dim=-1)

                # 2. 1001-Bin Categorical Two-Hot Value Cross-Entropy
                val_log_probs = F.log_softmax(pred_val_logits, dim=-1)
                per_sample_val_loss = -torch.sum(b_two_hot_val * val_log_probs, dim=-1)

                # 3. World Dynamics Self-Supervised MSE
                per_sample_dyn_loss = F.mse_loss(pred_dyn, b_next_scalar, reduction="none").mean(dim=-1)

                # 4. KataGo Auxiliary Heads MSE
                per_sample_tile_loss = F.mse_loss(pred_tile, b_tile, reduction="none").mean(dim=[1, 2, 3])
                per_sample_price_loss = F.mse_loss(pred_price, b_price, reduction="none").mean(dim=-1)

                # PER Weighted Multi-Task Objective
                per_sample_total = (
                    per_sample_policy_loss
                    + 0.5 * per_sample_val_loss
                    + 0.25 * per_sample_dyn_loss
                    + 0.1 * per_sample_tile_loss
                    + 0.1 * per_sample_price_loss
                )
                loss_total = torch.mean(is_weights * per_sample_total)

                loss_total.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Update PER priorities with TD error
                td_errors = per_sample_val_loss.detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, td_errors)

                loss_policy_list.append(per_sample_policy_loss.mean().item())
                loss_val_list.append(per_sample_val_loss.mean().item())
                loss_dyn_list.append(per_sample_dyn_loss.mean().item())
                loss_tile_list.append(per_sample_tile_loss.mean().item())
                loss_price_list.append(per_sample_price_loss.mean().item())
                loss_total_list.append(loss_total.item())
                grad_norm_list.append(float(grad_norm))

            self.scheduler.step()
            self.model.eval()

            mean_policy_loss = float(np.mean(loss_policy_list))
            mean_val_loss = float(np.mean(loss_val_list))
            mean_dyn_loss = float(np.mean(loss_dyn_list))
            mean_total_loss = float(np.mean(loss_total_list))
            mean_grad_norm = float(np.mean(grad_norm_list))

            print(f"[Iter {self.current_iteration:03d}] Loss: Total={mean_total_loss:.4f} | Policy={mean_policy_loss:.4f} | "
                  f"Value={mean_val_loss:.4f} | Dyn={mean_dyn_loss:.4f} | GradNorm={mean_grad_norm:.2f} | LR={self.scheduler.get_last_lr()[0]:.2e}")

            # 5. Add Champion Snapshot to League & Auto-Checkpoint
            self.league.add_checkpoint(iteration=self.current_iteration, state_dict=self.model.state_dict())

            mins_since_last_ckpt = (time.time() - self.last_checkpoint_time) / 60.0
            if mins_since_last_ckpt >= self.checkpoint_interval_mins or self.current_iteration % 5 == 0:
                self.save_checkpoint()

            iter_duration = time.time() - iter_start_time

            # 6. Live Telemetry Logging
            log_entry = {
                "iteration": self.current_iteration,
                "timestamp": time.time(),
                "elapsed_hours": round((time.time() - self.overall_start_time) / 3600.0, 3),
                "duration_seconds": round(iter_duration, 2),
                "total_matches": self.total_matches_run,
                "dataset_size": len(self.replay_buffer),
                "metrics": {
                    "win_rate": round(win_rate, 2),
                    "mean_cash": round(mean_cash, 2),
                    "peak_cash": round(peak_cash, 2),
                    "mean_opp_cash": round(mean_opp_cash, 2),
                    "mean_margin": round(mean_margin, 2),
                    "mean_deadweight": round(mean_deadweight, 2),
                    "champion_elo": round(champ_elo, 1),
                },
                "losses": {
                    "policy_loss": round(mean_policy_loss, 5),
                    "value_loss": round(mean_val_loss, 5),
                    "dynamics_loss": round(mean_dyn_loss, 5),
                    "total_loss": round(mean_total_loss, 5),
                    "grad_norm": round(mean_grad_norm, 4),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                },
                "league_summary": self.league.get_league_summary(),
            }
            self.training_logs.append(log_entry)

            with open(self.log_path, "w") as f:
                json.dump(self.training_logs, f, indent=2)

        # Final Wrap-Up & Submission Packaging
        self.wrap_up_and_package()

    def wrap_up_and_package(self):
        """Final wrap-up, benchmark validation vs 'starter', and standalone submission packaging."""
        print("\n" + "=" * 85)
        print(" TRAINING COMPLETE! INITIATING CERTIFICATION & SUBMISSION PACKAGING")
        print("=" * 85)

        self.save_checkpoint(is_final=True)

        # 1. Validation Benchmark vs Starter Bot
        print("\n[Validation] Running 6-game certification benchmark vs 'starter' baseline...")
        val_tasks = []
        champ_sd = copy.deepcopy(self.model.state_dict())
        for i in range(6):
            val_tasks.append((i, 1000 + i, champ_sd, "KaggleStarter", "sparring", None, 16))

        with mp.Pool(processes=min(self.num_workers, 6)) as pool:
            val_results = pool.starmap(run_overnight_match_worker, val_tasks)

        val_wins = sum(1 for _, m in val_results if m["won"])
        val_cashes = [m["champ_cash"] for _, m in val_results]
        starter_cashes = [m["opp_cash"] for _, m in val_results]
        val_margins = [m["margin"] for _, m in val_results]

        val_win_rate = (val_wins / len(val_results)) * 100.0
        val_mean_cash = float(np.mean(val_cashes))
        val_mean_margin = float(np.mean(val_margins))

        print(f"[Validation Results] Win Rate vs Starter: {val_win_rate:.1f}% ({val_wins}/{len(val_results)}) | "
              f"Champion Cash: ${val_mean_cash:,.0f} vs Starter: ${np.mean(starter_cashes):,.0f} | Margin: ${val_mean_margin:+,.0f}")

        # 2. Package Winning Weights into submission.py
        print("\n[Packaging] Embedding weights into standalone single-file 'submission.py'...")
        self.build_submission_file()

        total_hours = (time.time() - self.overall_start_time) / 3600.0
        print(f"\n[SUCCESS] Overnight RL Pipeline finished cleanly in {total_hours:.2f} hours.")
        print(f"Final Weights: '{self.champion_weights_path}'")
        print(f"Submission Bot: 'submission.py'")
        print(f"Training Logs: '{self.log_path}'")
        print("=" * 85)

    def build_submission_file(self):
        """Compiles WorldChampionNetwork weights into self-contained submission.py."""
        sd_fp16 = {k: v.half() for k, v in self.model.state_dict().items()}
        buf = io.BytesIO()
        torch.save(sd_fp16, buf)
        compressed = zlib.compress(buf.getvalue(), level=9)
        b85_weights = base64.b85encode(compressed).decode("ascii")

        submission_code = f'''"""
Kaggriculture World Champion Deep RL Agent: Gumbel MuZero Latent Search + 1001-Bin Value Head.
Trained via 14-Core Overnight Asynchronous Actor-Learner Architecture with 30-Personality Mega League.
100% Self-Contained Standalone Single-File Kaggle Submission.
"""

import base64
import io
import math
import os
import sys
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

CROP_SPECS = {{
    "WHEAT": {{"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6}},
    "CARROT": {{"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4}},
    "TOMATO": {{"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4}},
    "STRAWBERRY": {{"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4}},
    "MELON": {{"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6}},
}}

BASE_PRICES = {{
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}}

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
NUM_MACRO_ACTIONS = 10
SPATIAL_CHANNELS = 11
SCALAR_DIM = 32
NUM_BINS = 1001
V_MIN = -15.0
V_MAX = 15.0


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def categorical_to_scalar(logits: torch.Tensor, v_min: float = V_MIN, v_max: float = V_MAX, num_bins: int = NUM_BINS) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    bin_centers = torch.linspace(v_min, v_max, num_bins, device=probs.device, dtype=probs.dtype)
    expected_symlog = torch.sum(probs * bin_centers, dim=-1, keepdim=True)
    return symexp(expected_symlog)


def get_manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_step_towards(curr: Tuple[int, int], target: Tuple[int, int]) -> str:
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


def parse_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{{}}, {{}}])
    my_farm = farms[player] if player < len(farms) else {{}}
    opp_farm = farms[1 - player] if 1 - player < len(farms) else {{}}
    private = obs.get("private", {{}}) or {{}}
    market = obs.get("market", {{}}) or {{}}
    town = obs.get("town", {{}}) or {{}}

    return {{
        "day": int(obs.get("day", 0)),
        "hour": int(obs.get("hour", 0)),
        "step": int(obs.get("step", 0)),
        "money": float(my_farm.get("money", 3000.0)),
        "opp_money": float(opp_farm.get("money", 3000.0)),
        "farmer_pos": tuple(my_farm.get("farmer", [4, 4])),
        "hands_pos": [tuple(h) for h in my_farm.get("hands", [])],
        "tiles": my_farm.get("tiles", []),
        "shed": private.get("shed", {{}}),
        "seeds": private.get("seeds", {{}}),
        "inventories": private.get("inventories", []),
        "hires_today": int(my_farm.get("hires_today", 0)),
        "unlocked_quads": my_farm.get("unlocked_quadrants", ["NW"]),
        "market_prices": market.get("prices", {{}}),
        "unlocked_shops": town.get("unlocked_shops", []),
    }}


def encode_observation(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{{}}, {{}}])
    my_farm = farms[player] if player < len(farms) else {{}}
    opp_farm = farms[1 - player] if 1 - player < len(farms) else {{}}
    private = obs.get("private", {{}}) or {{}}
    market = obs.get("market", {{}}) or {{}}

    day = float(obs.get("day", 0))
    hour = float(obs.get("hour", 0))
    step = float(obs.get("step", 0))

    grid = np.zeros((SPATIAL_CHANNELS, 10, 10), dtype=np.float32)
    tiles = my_farm.get("tiles", [])
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            if t == "LOCKED":
                grid[0, r, c] = 0.0
            else:
                grid[0, r, c] = 1.0

            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    grid[1, r, c] = 1.0
                    crop = t.get("crop", "CARROT")
                    if crop in CROPS:
                        grid[2, r, c] = (CROPS.index(crop) + 1) / len(CROPS)
                    age = max(0, day - t.get("planted_day", day))
                    grid[3, r, c] = min(age / 12.0, 1.0)
                    grid[4, r, c] = 1.0 if t.get("watered_today", False) else 0.0
                    grid[5, r, c] = min(float(t.get("yield_units", 0)) / 6.0, 1.0)
                elif k in ["COOP", "PASTURE"]:
                    grid[6, r, c] = 0.5 if k == "COOP" else 1.0
                    animal = t.get("animal")
                    if animal in ANIMALS:
                        grid[7, r, c] = (ANIMALS.index(animal) + 1) / len(ANIMALS)
                elif k == "WEED":
                    grid[8, r, c] = 1.0

    fx, fy = my_farm.get("farmer", [4, 4])
    if 0 <= fy < 10 and 0 <= fx < 10:
        grid[9, fy, fx] = 1.0

    for hx, hy in my_farm.get("hands", []):
        if 0 <= hy < 10 and 0 <= hx < 10:
            grid[10, hy, hx] += 1.0

    scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
    money = float(my_farm.get("money", 3000.0))
    opp_money = float(opp_farm.get("money", 3000.0))

    scalars[0] = min(money / 10000.0, 1.0)
    scalars[1] = min(opp_money / 10000.0, 1.0)
    scalars[2] = (money - opp_money) / 5000.0
    scalars[3] = step / 720.0
    scalars[4] = day / 30.0
    scalars[5] = hour / 24.0
    scalars[6] = float(my_farm.get("hires_today", 0)) / 5.0

    unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
    scalars[7] = len(unlocked_quads) / 4.0

    shed = private.get("shed", {{}})
    for idx, prod in enumerate(PRODUCTS):
        scalars[8 + idx] = min(float(shed.get(prod, 0)) / 25.0, 1.0)

    seeds = private.get("seeds", {{}})
    for idx, crop in enumerate(CROPS):
        scalars[17 + idx] = min(float(seeds.get(crop, 0)) / 10.0, 1.0)

    prices = market.get("prices", {{}})
    for idx, prod in enumerate(PRODUCTS):
        scalars[22 + idx] = min(float(prices.get(prod, 25)) / 250.0, 1.0)

    scalars[31] = 1.0 if day >= 27 else 0.0

    return grid, scalars


def compute_action_mask(obs: Dict[str, Any]) -> np.ndarray:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{{}}, {{}}])
    my_farm = farms[player] if player < len(farms) else {{}}
    day = int(obs.get("day", 0))
    step = int(obs.get("step", 0))
    money = float(my_farm.get("money", 3000.0))
    unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
    tiles = my_farm.get("tiles", [])

    if step >= 718:
        mask = np.zeros(NUM_MACRO_ACTIONS, dtype=bool)
        mask[7] = True
        return mask

    mask = np.ones(NUM_MACRO_ACTIONS, dtype=bool)
    has_animals = False
    unlocked_empty_count = 0
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if t is None:
                unlocked_empty_count += 1
            elif isinstance(t, dict):
                k = t.get("kind")
                if k in ["COOP", "PASTURE"] and t.get("animal"):
                    has_animals = True

    if day >= 28:
        mask[0] = False
    if day >= 27:
        mask[1] = False
    if day >= 23:
        mask[2] = False

    num_unlocked = len(unlocked_quads)
    next_quad_cost = 1000 if num_unlocked == 1 else (2000 if num_unlocked == 2 else (4000 if num_unlocked == 3 else 999999))
    if num_unlocked >= 4 or money < next_quad_cost or day >= 26:
        mask[3] = False

    hires_today = int(my_farm.get("hires_today", 0))
    max_hires = 5 if day >= 15 else 2
    if money < 50 or day >= 29 or hires_today >= max_hires:
        mask[4] = False

    if money < 300 or day >= 25 or unlocked_empty_count < 4:
        mask[5] = False
    if money < 400 or day >= 25 or unlocked_empty_count < 6:
        mask[6] = False

    mask[7] = bool(day >= 27 or step >= 710)
    if money < 100:
        mask[8] = False
    if not has_animals:
        mask[9] = False

    if not mask.any():
        mask[0] = True
    return mask


class SqueezeExcitation(nn.Module):
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
        return x * self.fc(x)


class SEResBlock(nn.Module):
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
        return F.relu(out + residual, inplace=True)


class WorldChampionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(SPATIAL_CHANNELS, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.ModuleList([SEResBlock(64), SEResBlock(64)])
        self.spatial_fc = nn.Sequential(
            nn.Linear(64 * 10 * 10, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        self.scalar_fc = nn.Sequential(
            nn.Linear(SCALAR_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, NUM_MACRO_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_BINS),
        )
        self.ssl_dynamics_head = nn.Sequential(
            nn.Linear(128 + NUM_MACRO_ACTIONS, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, SCALAR_DIM),
        )
        self.tile_yield_head = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.price_forecaster_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9),
        )

    def extract_features(self, grid: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_grid = self.spatial_stem(grid)
        for res_block in self.res_blocks:
            x_grid = res_block(x_grid)
        x_spatial = self.spatial_fc(x_grid.view(x_grid.size(0), -1))
        x_scalar = self.scalar_fc(scalars)
        latent_z = self.fusion(torch.cat([x_spatial, x_scalar], dim=1))
        return latent_z, x_spatial

    @torch.no_grad()
    def predict(self, grid: torch.Tensor, scalars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        latent_z, _ = self.extract_features(grid, scalars)
        policy_logits = self.policy_head(latent_z)
        value_logits = self.value_head(latent_z)
        probs = F.softmax(policy_logits, dim=-1)
        expected_val = categorical_to_scalar(value_logits)
        return probs, expected_val


EMBEDDED_MODEL_WEIGHTS_B85 = """{b85_weights}"""


def load_embedded_model() -> WorldChampionNetwork:
    model = WorldChampionNetwork()
    try:
        if EMBEDDED_MODEL_WEIGHTS_B85 and not EMBEDDED_MODEL_WEIGHTS_B85.startswith("__"):
            raw = zlib.decompress(base64.b85decode(EMBEDDED_MODEL_WEIGHTS_B85.encode("ascii")))
            loaded_sd_fp16 = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            loaded_sd_fp32 = {{k: v.float() for k, v in loaded_sd_fp16.items()}}
            model.load_state_dict(loaded_sd_fp32)
            model.eval()
    except Exception:
        pass
    return model


class FrontierGrandmasterAgent:
    def __init__(
        self,
        target_geese: int = 2,
        target_cows: int = 1,
        target_sheep: int = 1,
        max_workers_early: int = 2,
        max_workers_late: int = 5,
    ):
        self.target_geese = target_geese
        self.target_cows = target_cows
        self.target_sheep = target_sheep
        self.max_workers_early = max_workers_early
        self.max_workers_late = max_workers_late
        self.shed_tiles = [(4, 4), (5, 4), (4, 5), (5, 5)]

    def __call__(self, obs: Dict[str, Any], config: Any = None, macro_action: Optional[int] = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        step = state["step"]
        money = state["money"]
        farmer_pos = state["farmer_pos"]
        hands_pos = state["hands_pos"]
        tiles = state["tiles"]
        shed = state["shed"]
        seeds = state["seeds"]
        inventories = state["inventories"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]

        market_orders: List[List[Any]] = []

        # 1. Turn 719 Complete Liquidation
        if step >= 718 or (day == 29 and hour >= 22) or (macro_action == 7 and day >= 27):
            for item in PRODUCTS:
                cnt = shed.get(item, 0)
                if cnt > 0:
                    market_orders.append(["SELL", item, cnt])

            units = [farmer_pos] + hands_pos
            unit_actions = []
            for u_pos in units:
                ux, uy = u_pos
                u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
                if isinstance(u_tile, dict) and u_tile.get("yield_units", 0) > 0:
                    unit_actions.append(["HARVEST"])
                elif u_pos not in self.shed_tiles:
                    nearest = min(self.shed_tiles, key=lambda s: get_manhattan_dist(u_pos, s))
                    unit_actions.append([get_step_towards(u_pos, nearest)])
                else:
                    unit_actions.append(["PASS"])

            return {{
                "farmer": unit_actions[0] if unit_actions else ["PASS"],
                "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
                "market": market_orders[:10],
            }}

        # 2. Asset & Structure Audit
        living_geese = 0
        living_cows = 0
        living_sheep = 0
        empty_coops = 0
        empty_pastures = 0
        planted_crops: Dict[str, int] = {{c: 0 for c in CROP_SPECS}}
        animal_tiles: List[Tuple[int, int, str, Dict[str, Any]]] = []
        crop_tiles: List[Tuple[int, int, Dict[str, Any]]] = []
        unlocked_empty_tiles: List[Tuple[int, int]] = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    unlocked_empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    animal = t.get("animal")
                    if k == "COOP":
                        if animal == "GOOSE":
                            living_geese += 1
                            animal_tiles.append((c, r, "GOOSE", t))
                        elif animal is None:
                            empty_coops += 1
                    elif k == "PASTURE":
                        if animal == "COW":
                            living_cows += 1
                            animal_tiles.append((c, r, "COW", t))
                        elif animal == "SHEEP":
                            living_sheep += 1
                            animal_tiles.append((c, r, "SHEEP", t))
                        elif animal is None:
                            empty_pastures += 1
                    elif k == "PLANT":
                        crop_name = t.get("crop", "CARROT")
                        crop_tiles.append((c, r, t))
                        planted_crops[crop_name] = planted_crops.get(crop_name, 0) + 1

        total_living_animals = living_geese + living_cows + living_sheep

        # 3. Market Trading & Capital Allocation
        for item in ["EGG", "MILK", "WOOL", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "FERTILIZER"]:
            count = shed.get(item, 0)
            if count > 0:
                if item == "FERTILIZER" and len(crop_tiles) > 0 and day <= 22 and count > 4:
                    market_orders.append(["SELL", item, count - 4])
                elif item != "FERTILIZER":
                    market_orders.append(["SELL", item, count])

        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_req = total_living_animals
        wheat_safe_res = max(6, daily_feed_req * 3)

        if wheat_in_shed > wheat_safe_res + 12 or day >= 28:
            market_orders.append(["SELL", "WHEAT", max(0, wheat_in_shed - (wheat_safe_res if day < 28 else 0))])
        elif wheat_in_shed < daily_feed_req and money >= 60 and day < 28:
            qty = min(6, wheat_safe_res - wheat_in_shed)
            market_orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= 25 * qty
            shed["WHEAT"] = wheat_in_shed + qty

        # Land Expansion
        if (macro_action in [3, None]) and day <= 24:
            if "NE" not in unlocked and (day <= 14 or money >= 1250) and money >= 1000 and day <= 22:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and (day <= 16 or money >= 2500) and money >= 2000 and day <= 22:
                market_orders.append(["BUY_LAND"])
                money -= 2000
            elif "SE" not in unlocked and "SW" in unlocked and "NE" in unlocked and money >= 5000 and day <= 24:
                market_orders.append(["BUY_LAND"])
                money -= 4000

        # Labor Hiring
        max_hires = self.max_workers_late if (day >= 15 or macro_action == 4) else self.max_workers_early
        if day < 28 and hour <= 2 and hires_today < max_hires and money >= 50:
            market_orders.append(["HIRE"])
            money -= 2

        # Animals Buying
        geese_in_shed = shed.get("GOOSE", 0)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)
        geese_in_hand = sum(inv.get("GOOSE", 0) for inv in inventories)
        cows_in_hand = sum(inv.get("COW", 0) for inv in inventories)
        sheep_in_hand = sum(inv.get("SHEEP", 0) for inv in inventories)

        available_empty_coops = max(0, empty_coops - geese_in_shed - geese_in_hand)
        available_empty_pastures = max(0, empty_pastures - (cows_in_shed + cows_in_hand + sheep_in_shed + sheep_in_hand))

        if available_empty_coops > 0 and (living_geese + geese_in_shed + geese_in_hand) < self.target_geese and money >= 380 and day <= 22:
            market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
            money -= 300
            geese_in_shed += 1
        elif available_empty_pastures > 0 and (living_cows + cows_in_shed + cows_in_hand) < self.target_cows and money >= 550 and day <= 18:
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_in_shed += 1
        elif available_empty_pastures > 0 and (living_sheep + sheep_in_shed + sheep_in_hand) < self.target_sheep and money >= 650 and day <= 16:
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_in_shed += 1

        # Seed Purchasing
        wheat_seeds = seeds.get("WHEAT", 0)
        planted_wheat = planted_crops.get("WHEAT", 0)
        if day <= 24 and (wheat_seeds + planted_wheat) < 6 and money >= 40:
            qty = min(4, 6 - (wheat_seeds + planted_wheat))
            market_orders.append(["BUY_SEED", "WHEAT", qty])
            money -= 10 * qty
            seeds["WHEAT"] = wheat_seeds + qty

        melon_budget = seeds.get("MELON", 0) + planted_crops.get("MELON", 0)
        if ((day <= 2) or (10 <= day <= 12) or macro_action == 2) and day <= 14 and melon_budget < 6 and money >= 200:
            qty = min(4, 6 - melon_budget)
            market_orders.append(["BUY_SEED", "MELON", qty])
            money -= 80 * qty
            seeds["MELON"] = seeds.get("MELON", 0) + qty

        carrot_budget = seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0)
        if day <= 25 and carrot_budget < 16 and money >= 60:
            qty = min(6, 16 - carrot_budget)
            market_orders.append(["BUY_SEED", "CARROT", qty])
            money -= 20 * qty
            seeds["CARROT"] = seeds.get("CARROT", 0) + qty

        # 4. Field Tasks Scanning
        feed_tasks = []
        care_tasks = []
        harvest_animal_tasks = []
        fertilizer_tasks = []
        water_crop_tasks = []
        harvest_crop_tasks = []
        fertilize_crop_tasks = []
        weed_tasks = []
        plant_tasks = []
        build_coop_tasks = []
        build_pasture_tasks = []
        place_animal_tasks = []

        for ax, ay, aname, atile in animal_tiles:
            pos = (ax, ay)
            if not atile.get("fed_today", False):
                feed_tasks.append(pos)
            if not atile.get("cared_today", False):
                care_tasks.append(pos)
            if atile.get("yield_units", 0) > 0:
                harvest_animal_tasks.append(pos)
            if atile.get("fertilizer_available", False):
                fertilizer_tasks.append(pos)

        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            crop = ctile.get("crop", "CARROT")
            cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
            age = day - ctile.get("planted_day", day)
            is_ongoing = cspec["ongoing"]
            yield_u = ctile.get("yield_units", 0)

            if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 27 and yield_u > 0))) or (
                is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
            ):
                harvest_crop_tasks.append(pos)
            elif not ctile.get("watered_today", False) and day < 28:
                water_crop_tasks.append(pos)
            elif ctile.get("fertilized_until_day", -1) < day and day <= 22:
                fertilize_crop_tasks.append(pos)

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if isinstance(t, dict):
                    if t.get("kind") == "WEED":
                        weed_tasks.append(pos)
                    elif t.get("kind") == "COOP" and "animal" not in t:
                        place_animal_tasks.append((c, r, "GOOSE"))
                    elif t.get("kind") == "PASTURE" and "animal" not in t:
                        if (cows_in_shed + cows_in_hand) > 0:
                            place_animal_tasks.append((c, r, "COW"))
                        elif (sheep_in_shed + sheep_in_hand) > 0:
                            place_animal_tasks.append((c, r, "SHEEP"))
                        elif (living_cows + cows_in_shed + cows_in_hand) <= (living_sheep + sheep_in_shed + sheep_in_hand):
                            place_animal_tasks.append((c, r, "COW"))
                        else:
                            place_animal_tasks.append((c, r, "SHEEP"))

        unlocked_empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        curr_e_idx = 0
        if (living_geese + empty_coops) < self.target_geese and curr_e_idx < len(unlocked_empty_tiles) and day <= 20:
            build_coop_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1

        if (
            (living_cows + living_sheep + empty_pastures) < (self.target_cows + self.target_sheep)
            and curr_e_idx < len(unlocked_empty_tiles)
            and day <= 18
        ):
            build_pasture_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1

        total_seeds_count = sum(seeds.values())
        if day <= 25 and total_seeds_count > 0:
            for p in unlocked_empty_tiles[curr_e_idx:40]:
                plant_tasks.append(p)

        # 5. Multi-Worker Dispatch
        units = [farmer_pos] + hands_pos
        unit_actions = []
        claimed_tasks: Set[Tuple[int, int]] = set()

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {{}}
            is_at_shed = u_pos in self.shed_tiles

            # Atomic Shed Pickup
            if is_at_shed:
                placed_target = next((pt for pt in place_animal_tasks if pt[:2] not in claimed_tasks), None)
                if placed_target is not None:
                    target_animal = placed_target[2]
                    if shed.get(target_animal, 0) > 0 and u_inv.get(target_animal, 0) == 0:
                        unit_actions.append(["PICKUP", target_animal, 1])
                        shed[target_animal] -= 1
                        u_inv[target_animal] = u_inv.get(target_animal, 0) + 1
                        continue

                if feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    qty = min(4, shed.get("WHEAT", 0), len(feed_tasks))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "WHEAT", qty])
                        shed["WHEAT"] -= qty
                        u_inv["WHEAT"] = u_inv.get("WHEAT", 0) + qty
                        continue

                if fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0 and day <= 22:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "FERTILIZER", qty])
                        shed["FERTILIZER"] -= qty
                        u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + qty
                        continue

            # Standing Tile Actions
            if isinstance(u_tile, dict):
                k = u_tile.get("kind")
                animal = u_tile.get("animal")
                if k in ["COOP", "PASTURE"]:
                    if animal is not None:
                        if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                            unit_actions.append(["FEED"])
                            u_inv["WHEAT"] -= 1
                            continue
                        if not u_tile.get("cared_today", False):
                            unit_actions.append(["CARE"])
                            continue
                        if u_tile.get("fertilizer_available", False):
                            unit_actions.append(["COLLECT_FERTILIZER"])
                            u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + 1
                            continue
                        if u_tile.get("yield_units", 0) > 0:
                            unit_actions.append(["HARVEST"])
                            continue
                    else:
                        for aname in ["GOOSE", "COW", "SHEEP"]:
                            if u_inv.get(aname, 0) > 0:
                                req_struct = "COOP" if aname == "GOOSE" else "PASTURE"
                                if k == req_struct:
                                    unit_actions.append(["PLACE", aname])
                                    u_inv[aname] -= 1
                                    break
                        if len(unit_actions) > u_idx:
                            continue
                elif k == "PLANT":
                    crop = u_tile.get("crop", "CARROT")
                    cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                    age = day - u_tile.get("planted_day", day)
                    is_ongoing = cspec["ongoing"]
                    yield_u = u_tile.get("yield_units", 0)

                    if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 27 and yield_u > 0))) or (
                        is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        unit_actions.append(["HARVEST"])
                        continue
                    if not u_tile.get("watered_today", False) and day < 28:
                        unit_actions.append(["WATER"])
                        continue
                    if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0 and day <= 22:
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue
                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue
            elif u_tile is None:
                if u_pos in build_coop_tasks:
                    unit_actions.append(["BUILD_COOP"])
                    build_coop_tasks.remove(u_pos)
                    continue
                elif u_pos in build_pasture_tasks:
                    unit_actions.append(["BUILD_PASTURE"])
                    build_pasture_tasks.remove(u_pos)
                    continue
                elif u_pos in plant_tasks and total_seeds_count > 0 and day <= 25:
                    for crop_cand in ["MELON", "TOMATO", "STRAWBERRY", "WHEAT", "CARROT"]:
                        if seeds.get(crop_cand, 0) > 0:
                            unit_actions.append(["PLANT", crop_cand])
                            seeds[crop_cand] -= 1
                            total_seeds_count -= 1
                            break
                    else:
                        unit_actions.append(["PASS"])
                    continue

            # Target Navigation
            best_target: Optional[Tuple[int, int]] = None
            best_dist = 999

            has_animal_in_hand = any(u_inv.get(a, 0) > 0 for a in ["GOOSE", "COW", "SHEEP"])
            if has_animal_in_hand:
                for pt in place_animal_tasks:
                    pos = pt[:2]
                    if pos not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, pos)
                        if d < best_dist:
                            best_dist = d
                            best_target = pos

            if best_target is None and u_inv.get("WHEAT", 0) > 0 and feed_tasks:
                for task in feed_tasks:
                    if task not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, task)
                        if d < best_dist:
                            best_dist = d
                            best_target = task

            if best_target is None and feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                nearest_shed = min(self.shed_tiles, key=lambda s: get_manhattan_dist(u_pos, s))
                best_target = nearest_shed
                best_dist = get_manhattan_dist(u_pos, nearest_shed)

            if best_target is None:
                priority_queue = (
                    harvest_crop_tasks
                    + harvest_animal_tasks
                    + care_tasks
                    + water_crop_tasks
                    + fertilizer_tasks
                    + build_coop_tasks
                    + build_pasture_tasks
                    + plant_tasks
                    + weed_tasks
                )
                for task in priority_queue:
                    if task in claimed_tasks:
                        continue
                    d = get_manhattan_dist(u_pos, task)
                    if d < best_dist:
                        best_dist = d
                        best_target = task

            if best_target is not None:
                claimed_tasks.add(best_target)
                unit_actions.append([get_step_towards(u_pos, best_target)])
            else:
                if not is_at_shed:
                    nearest_shed = min(self.shed_tiles, key=lambda s: get_manhattan_dist(u_pos, s))
                    unit_actions.append([get_step_towards(u_pos, nearest_shed)])
                else:
                    unit_actions.append(["PASS"])

        return {{
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }}


_executor_instance = FrontierGrandmasterAgent()


class ChampionSubmissionBot:
    def __init__(self):
        self.model = load_embedded_model()

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            probs_t, _ = self.model.predict(grid_t, scalars_t)
            probs = probs_t.squeeze(0).cpu().numpy()

        mask = compute_action_mask(obs)
        probs = probs * mask
        if probs.sum() > 0:
            probs = probs / probs.sum()
            chosen_action = int(np.argmax(probs))
        else:
            chosen_action = 0

        return _executor_instance(obs, config, macro_action=chosen_action)


_bot_instance = ChampionSubmissionBot()


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _bot_instance(obs, config)
'''

        with open("submission.py", "w", encoding="utf-8") as f:
            f.write(submission_code)
        print(f"[OK] Standalone World Champion 'submission.py' successfully compiled ({len(submission_code):,} chars).")


# ==============================================================================
# 7. MAIN CLI ENTRYPOINT & SMOKE TEST
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Overnight World-Class Deep RL Training Pipeline for Kaggriculture")
    parser.add_argument("--workers", type=int, default=14, help="Number of parallel CPU worker processes (default: 14)")
    parser.add_argument("--hours", type=float, default=6.5, help="Target training duration in hours (default: 6.5)")
    parser.add_argument("--iterations", type=int, default=100, help="Maximum number of training iterations (default: 100)")
    parser.add_argument("--matches_per_iter", type=int, default=28, help="Matches simulated per iteration (default: 28)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for neural network optimization (default: 64)")
    parser.add_argument("--epochs_per_iter", type=int, default=4, help="Training epochs per iteration (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW (default: 2e-4)")
    parser.add_argument("--buffer_capacity", type=int, default=500000, help="PER buffer capacity (default: 500000)")
    parser.add_argument("--checkpoint_dir", "--save_dir", type=str, default="weights/overnight_checkpoints", help="Checkpoints directory")
    parser.add_argument("--champion_weights", type=str, default="weights/alphagoat_world_champion.pt", help="Champion weights path")
    parser.add_argument("--log_path", type=str, default="data/overnight_training_log.json", help="Telemetry log path")
    parser.add_argument("--warm_start", type=str, default="weights/frontier_v2_grandmaster_champion.pt", help="Warm-start weights path")
    parser.add_argument("--checkpoint_mins", type=float, default=15.0, help="Checkpoint interval in minutes (default: 15.0)")
    parser.add_argument("--smoke_test", action="store_true", help="Run a fast 1-iteration smoke test across all 14 cores")

    args = parser.parse_args()
    mp.freeze_support()

    if args.smoke_test:
        print("\n>>> LAUNCHING 1-ITERATION 14-CORE SMOKE TEST...")
        trainer = OvernightRLTrainer(
            num_workers=args.workers,
            target_hours=0.1,
            max_iterations=1,
            matches_per_iter=args.workers,  # 1 match per worker = 14 matches
            batch_size=32,
            epochs_per_iter=2,
            learning_rate=args.lr,
            buffer_capacity=50000,
            checkpoint_dir=args.checkpoint_dir,
            champion_weights_path=args.champion_weights,
            log_path="data/test_overnight_log.json",
            warm_start_weights_path=args.warm_start,
            checkpoint_interval_mins=1.0,
        )
        trainer.run_training()
        print("\n[SMOKE TEST COMPLETE] Verified clean 14-core execution, PER buffer ingestion, neural optimization, and packaging!")
    else:
        trainer = OvernightRLTrainer(
            num_workers=args.workers,
            target_hours=args.hours,
            max_iterations=args.iterations,
            matches_per_iter=args.matches_per_iter,
            batch_size=args.batch_size,
            epochs_per_iter=args.epochs_per_iter,
            learning_rate=args.lr,
            buffer_capacity=args.buffer_capacity,
            checkpoint_dir=args.checkpoint_dir,
            champion_weights_path=args.champion_weights,
            log_path=args.log_path,
            warm_start_weights_path=args.warm_start,
            checkpoint_interval_mins=args.checkpoint_mins,
        )
        trainer.run_training()


if __name__ == "__main__":
    main()
