"""
Gumbel MuZero Latent Monte Carlo Tree Search (MCTS) Engine for Kaggriculture.

Components:
1. Gumbel-Top-m (m=4) candidate sampling + Sequential Halving (budget N=16).
2. Completed Q-values with normalized scale transformation.
3. Playout-Cap Randomization: Fast N=8 (75% routine turns) vs Deep N=64 (25% strategic turns).
4. Dynamic Zero-Deadweight Action Masking (Root and Latent space).
5. Vectorized ultra-fast 1-step dynamics imagination using FrontierV2Network.
"""

import math
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.encoder import NUM_MACRO_ACTIONS, MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS, encode_observation
from src.models.frontier_v2_network import FrontierV2Network, categorical_to_scalar

# ==============================================================================
# 1. DYNAMIC ACTION MASKING
# ==============================================================================

def compute_action_mask(obs: Dict[str, Any], player_idx: Optional[int] = None) -> np.ndarray:
    """
    Computes dynamic action validity mask over the 10 macro actions for the current observation.
    Zeroes out illegal, unprofitable, or deadweight-generating macro actions.
    """
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

    # 0: FARM_CARROTS_INTENSIVE (Carrots take 3 days)
    if day >= 28:
        mask[0] = False

    # 1: FARM_WHEAT_EXPANSION (Wheat takes 4 days)
    if day >= 27:
        mask[1] = False

    # 2: FARM_DIVERSIFIED (Melon 10d, Strawberry 10d, Tomato 8d)
    if day >= 23:
        mask[2] = False

    # 3: BUY_LAND_EXPANSION
    num_unlocked = len(unlocked_quads)
    next_quad_cost = 1000 if num_unlocked == 1 else (2000 if num_unlocked == 2 else (4000 if num_unlocked == 3 else 999999))
    if num_unlocked >= 4 or money < next_quad_cost or day >= 26:
        mask[3] = False

    # 4: HIRE_EXTRA_LABOR
    hires_today = int(my_farm.get("hires_today", 0))
    max_hires = 5 if day >= 15 else 2
    if money < 50 or day >= 29 or hires_today >= max_hires:
        mask[4] = False

    # 5: BUILD_GOOSE_COOP
    if money < 300 or day >= 25 or unlocked_empty_count < 4:
        mask[5] = False

    # 6: BUILD_PASTURE_LIVESTOCK
    if money < 400 or day >= 25 or unlocked_empty_count < 6:
        mask[6] = False

    # 7: HARVEST_AND_LIQUIDATE_ALL (Always valid)
    mask[7] = True

    # 8: MARKET_ARBITRAGE_TRADE
    if money < 100:
        mask[8] = False

    # 9: LIVESTOCK_CARE_FEED
    if not has_animals:
        mask[9] = False

    if not mask.any():
        mask[7] = True

    return mask


def compute_latent_action_mask(scalars_np: np.ndarray, depth: int) -> np.ndarray:
    """
    Computes dynamic action validity mask from predicted latent economic scalars.
    """
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
    next_quad_cost = 1000 if num_unlocked == 1 else (2000 if num_unlocked == 2 else (4000 if num_unlocked == 3 else 999999))
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


# ==============================================================================
# 2. GUMBEL MUZERO MCTS NODE & ENGINE
# ==============================================================================

class GumbelLatentNode:
    """Node in the Gumbel MuZero Latent Tree."""
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
        self.action_mask: np.ndarray = action_mask if action_mask is not None else np.ones(NUM_MACRO_ACTIONS, dtype=bool)
        self.value_est: float = value_est

        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.mean_value: float = 0.0
        self.children: Dict[int, GumbelLatentNode] = {}

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def update(self, value: float):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


class GumbelMuZeroMCTS:
    """
    Ultra-Fast Vectorized Gumbel MuZero Latent MCTS Engine.
    Executes Gumbel-Top-m Sequential Halving over imagined latent representations.
    """
    def __init__(
        self,
        model: FrontierV2Network,
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
        """Determines if current turn is strategic (N=64) or routine (N=8)."""
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
        temperature: float = 0.0,
        force_budget: Optional[int] = None,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Executes Gumbel MuZero Latent Tree Search from current observation.
        """
        self.model.eval()

        if force_budget is not None:
            budget = force_budget
        else:
            budget = 64 if self.is_strategic_turn(obs) else 8

        # 1. Root Evaluation
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        root_latent, x_spatial, _ = self.model.extract_features(grid_t, scalars_t)
        root_logits_t = self.model.policy_head(root_latent)
        root_val_logits = self.model.value_head(root_latent)

        root_logits = root_logits_t.squeeze(0).cpu().numpy()
        root_val_t = categorical_to_scalar(root_val_logits, is_logits=True)
        root_val = float(root_val_t.item())

        # 2. Dynamic Action Masking & Gumbel Sampling
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

        # 3. Fast Vectorized 1-Step Imagined Dynamics in Latent Space
        repeated_latent = root_latent.repeat(NUM_MACRO_ACTIONS, 1)
        dynamics_input = torch.cat([repeated_latent, self.eye10], dim=1)
        pred_future_scalars = self.model.ssl_dynamics_head(dynamics_input)  # (10, 32)
        child_latents = self.model.extract_latent_from_spatial(x_spatial.repeat(NUM_MACRO_ACTIONS, 1), pred_future_scalars)

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

        # 4. Sequential Halving Allocation
        num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
        budget_per_phase = max(1, budget // num_phases)

        for phase in range(num_phases):
            if len(active_candidates) <= 1:
                break

            sims_per_candidate = max(1, budget_per_phase // len(active_candidates))
            for cand in active_candidates:
                for _ in range(sims_per_candidate):
                    node = root.children[cand]
                    search_path = [root, node]
                    eval_val = node.value_est
                    for path_node in search_path:
                        path_node.update(eval_val)

            # Completed Q-values
            q_values = {c: root.children[c].mean_value if root.children[c].visit_count > 0 else root_val for c in active_candidates}
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
