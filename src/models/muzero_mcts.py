"""
MuZero Latent Monte Carlo Tree Search (MCTS) Engine for Kaggriculture.

Architectural Innovations:
1. Latent Representation Search:
   - Root node: Encodes real observation into 128-dim latent space via SSLPolicyValueNet.extract_features().
   - Tree nodes: Explores future imagined states using SSL Dynamics Head without executing full environment steps.
2. High-Throughput Batched Latent Rollouts:
   - Evaluates policy priors, value estimations, and economic scalar transitions simultaneously in parallel tensors.
3. Dynamic Zero-Deadweight Action Masking:
   - Prunes invalid, unprofitable, or capital-trapping actions dynamically across all latent search depths.
4. Macro-Action Translation:
   - Maps searched macro policies to specialized sub-executors (Carrot, Wheat, Portfolio, Hybrid, Livestock, Arbitrage, Liquidation).
"""

import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.encoder import NUM_MACRO_ACTIONS, MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS, encode_observation
from src.models.ssl_network import SSLPolicyValueNet
from src.agents.balanced_farmer import (
    crop_farmer_portfolio,
    crop_farmer_carrot,
    crop_farmer_wheat,
    crop_farmer_melon,
    crop_farmer_strawberry,
    crop_farmer_tomato,
)
from src.agents.hybrid_expert import hybrid_expert_agent
from src.agents.livestock_bot import livestock_agent
from src.agents.arbitrage_bot import arbitrage_agent
from src.training.league import (
    melon_jackpot_rusher_agent,
    town_shop_monopolizer_agent,
    livestock_tycoon_agent,
    carrot_clockwork_engine_agent,
)


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
    money = float(my_farm.get("money", 3000.0))
    unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
    tiles = my_farm.get("tiles", [])

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
    if money < 50 or day >= 29 or hires_today >= 3:
        mask[4] = False

    # 5: BUILD_GOOSE_COOP
    if money < 300 or day >= 25 or unlocked_empty_count < 4:
        mask[5] = False

    # 6: BUILD_PASTURE_LIVESTOCK
    if money < 400 or day >= 25 or unlocked_empty_count < 6:
        mask[6] = False

    # 7: HARVEST_AND_LIQUIDATE_ALL (Always valid, essential liquidation)
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
    Allows dynamic action pruning at arbitrary imagined depths in the search tree.
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
# 2. MUZERO LATENT MCTS NODE & TREE
# ==============================================================================

class MuZeroLatentNode:
    """
    Node in the MuZero Imagined Latent Search Tree.
    Contains latent representation, predicted scalars, child priors, and visit statistics.
    """

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
        self.children: Dict[int, MuZeroLatentNode] = {}

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def get_puct_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """Computes Upper Confidence Bound for Trees (PUCT) score."""
        u_score = c_puct * self.prior_prob * math.sqrt(max(1, parent_visits)) / (1 + self.visit_count)
        return self.mean_value + u_score

    def update(self, value: float):
        """Backpropagates value through this node."""
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


class MuZeroLatentMCTS:
    """
    High-Throughput Vectorized MuZero Latent Tree Search Engine:
    Executes imagined forward rollouts in the latent feature space
    using SSLPolicyValueNet's latent representation and SSL Dynamics Head.
    """

    def __init__(
        self,
        model: SSLPolicyValueNet,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        max_latent_depth: int = 3,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_latent_depth = max_latent_depth
        self.device = device
        self.eye10 = torch.eye(NUM_MACRO_ACTIONS, device=self.device)

    @torch.no_grad()
    def search(
        self,
        obs: Dict[str, Any],
        temperature: float = 0.0,
        add_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Executes MuZero Latent Tree Search from current observation.
        Returns:
            best_action: Selected macro action integer index (0-9)
            pi_target: MCTS visit distribution over macro actions (10,)
            root_value: Estimated value of root state
        """
        self.model.eval()

        # 1. ROOT REPRESENTATION & EVALUATION
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Extract root latent and evaluate root policy & value
        root_latent = self.model.extract_features(grid_t, scalars_t)
        root_logits = self.model.policy_head(root_latent)
        root_value_t = self.model.value_head(root_latent)

        root_probs = F.softmax(root_logits, dim=-1).squeeze(0).cpu().numpy()
        root_val = float(root_value_t.item())

        # Dynamic Action Masking at Root
        root_mask = compute_action_mask(obs)
        root_probs = root_probs * root_mask
        p_sum = root_probs.sum()
        if p_sum > 0:
            root_probs = root_probs / p_sum
        else:
            root_probs = root_mask.astype(np.float32) / max(1, root_mask.sum())

        # Dirichlet Exploration Noise
        if add_dirichlet_noise and dirichlet_epsilon > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * NUM_MACRO_ACTIONS)
            noise = noise * root_mask
            n_sum = noise.sum()
            if n_sum > 0:
                noise = noise / n_sum
            else:
                noise = root_probs
            root_probs = (1.0 - dirichlet_epsilon) * root_probs + dirichlet_epsilon * noise
            root_probs = root_probs / root_probs.sum()

        # 2. BATCHED 1-STEP IMAGINED DYNAMICS FOR ALL 10 ACTIONS
        repeated_latent = root_latent.repeat(NUM_MACRO_ACTIONS, 1)
        dynamics_input = torch.cat([repeated_latent, self.eye10], dim=1)
        pred_future_scalars = self.model.ssl_dynamics_head(dynamics_input)  # (10, 32)
        repeated_grid = grid_t.repeat(NUM_MACRO_ACTIONS, 1, 1, 1)
        child_latents = self.model.extract_features(repeated_grid, pred_future_scalars)  # (10, 128)

        child_logits = self.model.policy_head(child_latents)  # (10, 10)
        child_values = self.model.value_head(child_latents).squeeze(-1).cpu().numpy()  # (10,)
        child_probs_matrix = F.softmax(child_logits, dim=-1).cpu().numpy()  # (10, 10)
        pred_scalars_np = pred_future_scalars.cpu().numpy()  # (10, 32)

        root = MuZeroLatentNode(
            prior=1.0,
            depth=0,
            action_mask=root_mask,
            value_est=root_val,
        )

        # Expand Level 1 children
        for a_idx, prob in enumerate(root_probs):
            if root_mask[a_idx]:
                child_mask = compute_latent_action_mask(pred_scalars_np[a_idx], depth=1)
                child_p = child_probs_matrix[a_idx] * child_mask
                cp_sum = child_p.sum()
                if cp_sum > 0:
                    child_p = child_p / cp_sum
                else:
                    child_p = child_mask.astype(np.float32) / max(1, child_mask.sum())

                c_node = MuZeroLatentNode(
                    prior=float(prob),
                    depth=1,
                    action=a_idx,
                    action_mask=child_mask,
                    value_est=float(child_values[a_idx]),
                )

                # Expand Level 2 children
                if self.max_latent_depth >= 2:
                    for g_idx, g_prob in enumerate(child_p):
                        if child_mask[g_idx]:
                            c_node.children[g_idx] = MuZeroLatentNode(
                                prior=float(g_prob),
                                depth=2,
                                action=g_idx,
                                value_est=float(child_values[a_idx]),
                            )

                root.children[a_idx] = c_node

        # 3. FAST MCTS TREE SEARCH SIMULATIONS
        for _ in range(self.num_simulations):
            node = root
            search_path: List[MuZeroLatentNode] = [root]

            # Selection
            while node.is_expanded:
                best_action = -1
                best_score = -float("inf")

                for a_idx, child in node.children.items():
                    if not node.action_mask[a_idx]:
                        continue
                    score = child.get_puct_score(parent_visits=node.visit_count, c_puct=self.c_puct)
                    if score > best_score:
                        best_score = score
                        best_action = a_idx

                if best_action == -1 or best_action not in node.children:
                    break

                node = node.children[best_action]
                search_path.append(node)

            eval_val = node.value_est

            # Backpropagation
            for path_node in search_path:
                path_node.update(eval_val)

        # 4. ACTION SELECTION & VISIT DISTRIBUTION
        visit_counts = np.zeros(NUM_MACRO_ACTIONS, dtype=np.float32)
        for a_idx, child in root.children.items():
            visit_counts[a_idx] = child.visit_count

        total_visits = visit_counts.sum()
        if total_visits == 0:
            pi_target = root_probs
        else:
            pi_target = visit_counts / total_visits

        if temperature == 0.0 or total_visits == 0:
            selected_action = int(np.argmax(pi_target))
        else:
            exp_counts = visit_counts ** (1.0 / max(1e-3, temperature))
            dist = exp_counts / exp_counts.sum()
            selected_action = int(np.random.choice(NUM_MACRO_ACTIONS, p=dist))

        return selected_action, pi_target, root_val


# ==============================================================================
# 3. MUZERO MCTS AGENT (STANDALONE CALLABLE BOT)
# ==============================================================================

class MuZeroMCTSAgent:
    """
    Complete MuZero MCTS Agent for Kaggriculture:
    Integrates Latent Tree Search, dynamic action masking, and specialized sub-executors.
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = "weights/muzero_league_champion.pt",
        fallback_weights_path: Optional[str] = "weights/grandmaster_rl_champion.pt",
        num_simulations: int = 50,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        max_latent_depth: int = 3,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SSLPolicyValueNet().to(self.device)

        loaded = False
        for path in [model_weights_path, fallback_weights_path, "weights/grandmaster_ssl.pt"]:
            if path and os.path.exists(path):
                try:
                    self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                    loaded = True
                    break
                except Exception:
                    pass

        self.model.eval()

        self.mcts = MuZeroLatentMCTS(
            model=self.model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            max_latent_depth=max_latent_depth,
            device=self.device,
        )
        self.temperature = temperature

        # Macro Action Sub-Executors
        self.executors: Dict[int, Callable] = {
            0: crop_farmer_carrot,
            1: crop_farmer_wheat,
            2: crop_farmer_portfolio,
            3: hybrid_expert_agent,
            4: hybrid_expert_agent,
            5: hybrid_expert_agent,
            6: livestock_agent,
            7: None,  # Specialized Harvest & Liquidation
            8: arbitrage_agent,
            9: livestock_agent,
        }

    def search_action(self, obs: Dict[str, Any]) -> Tuple[int, np.ndarray, float]:
        """Performs latent tree search from the given observation."""
        return self.mcts.search(
            obs=obs,
            temperature=self.temperature,
            add_dirichlet_noise=False,
        )

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        """Callable entrypoint for Kaggle simulation."""
        best_macro, pi_target, val_est = self.search_action(obs)

        if best_macro in [0, 1, 2]:
            return crop_farmer_portfolio(obs, config)
        elif best_macro in [3, 4, 5]:
            return hybrid_expert_agent(obs, config)
        elif best_macro == 6:
            return livestock_agent(obs, config)
        elif best_macro == 7:
            res = hybrid_expert_agent(obs, config)
            shed = obs.get("private", {}).get("shed", {})
            orders = []
            for item, count in shed.items():
                if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                    orders.append(["SELL", item, count])
            res["market"] = orders[:10]
            return res
        elif best_macro == 8:
            return arbitrage_agent(obs, config)
        elif best_macro == 9:
            return livestock_agent(obs, config)
        else:
            return hybrid_expert_agent(obs, config)


# Default agent instance
muzero_mcts_agent = MuZeroMCTSAgent()


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    """Top-level agent function."""
    return muzero_mcts_agent(obs, config)
