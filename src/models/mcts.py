"""
Monte Carlo Tree Search (MCTS) Search Engine driven by Policy-Value Networks.
Implements PUCT selection, leaf expansion, value backpropagation, and macro-action selection.
"""

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from src.models.encoder import NUM_MACRO_ACTIONS, encode_observation
from src.models.network import PolicyValueNet
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


class MCTSNode:
    def __init__(self, prior: float = 0.0):
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.mean_value: float = 0.0
        self.prior_prob: float = prior
        self.children: Dict[int, MCTSNode] = {}

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def get_puct_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        u_score = c_puct * self.prior_prob * math.sqrt(max(1, parent_visits)) / (1 + self.visit_count)
        return self.mean_value + u_score

    def expand(self, action_probs: np.ndarray):
        for action_idx, prob in enumerate(action_probs):
            self.children[action_idx] = MCTSNode(prior=float(prob))

    def update(self, value: float):
        self.visit_count += 1
        self.total_value += value
        self.mean_value = self.total_value / self.visit_count


class AlphaZeroMCTSAgent:
    def __init__(
        self,
        model_weights_path: Optional[str] = "weights/pretrained_alphazero.pt",
        model_cls: Type[torch.nn.Module] = PolicyValueNet,
        num_simulations: int = 25,
        c_puct: float = 1.5,
        temperature: float = 0.0,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_cls().to(self.device)

        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=True))
            self.model.eval()

        self.crop_executor = crop_farmer_portfolio
        self.hybrid_executor = hybrid_expert_agent

    def search_best_action(self, obs: Dict[str, Any]) -> Tuple[int, np.ndarray, float]:
        """Runs MCTS search from the current observation root node."""
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_probs, root_value = self.model.predict(grid_t, scalars_t)

        probs_np = action_probs.squeeze(0).cpu().numpy()
        val_float = float(root_value.item())

        root = MCTSNode(prior=1.0)
        root.expand(probs_np)

        for _ in range(self.num_simulations):
            node = root
            best_action = -1
            best_score = -float("inf")
            for action_idx, child in node.children.items():
                score = child.get_puct_score(parent_visits=node.visit_count, c_puct=self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action = action_idx

            if best_action != -1:
                selected_child = node.children[best_action]
                selected_child.update(val_float)
                root.update(val_float)

        visit_counts = np.array([root.children[a].visit_count for a in range(NUM_MACRO_ACTIONS)])
        if self.temperature == 0.0 or visit_counts.sum() == 0:
            best_action = int(np.argmax(probs_np))
        else:
            exp_counts = visit_counts ** (1.0 / self.temperature)
            action_dist = exp_counts / exp_counts.sum()
            best_action = int(np.random.choice(len(action_dist), p=action_dist))

        return best_action, probs_np, val_float

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        best_action, probs, val = self.search_best_action(obs)

        if best_action in [0, 1, 2]:
            return self.crop_executor(obs, config)
        elif best_action in [3, 4, 5, 6]:
            return self.hybrid_executor(obs, config)
        elif best_action == 7:
            res = self.crop_executor(obs, config)
            shed = obs.get("private", {}).get("shed", {})
            orders = []
            for item, count in shed.items():
                if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                    orders.append(["SELL", item, count])
            res["market"] = orders[:10]
            return res
        else:
            return self.crop_executor(obs, config)


def compute_action_mask(obs: Dict[str, Any], player_idx: Optional[int] = None) -> np.ndarray:
    """Computes dynamic action validity mask over the 10 macro actions."""
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


class SSLMCTSAgent:
    """
    Self-Supervised Learning MCTS Agent driven by SSLPolicyValueNet.
    Performs MCTS planning over macro-action space with dynamic action masking
    and translates chosen policies across specialized sub-executors.
    """

    def __init__(
        self,
        model_weights_path: Optional[str] = "weights/grandmaster_ssl.pt",
        num_simulations: int = 60,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        use_action_mask: bool = True,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_action_mask = use_action_mask

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SSLPolicyValueNet().to(self.device)

        if model_weights_path and os.path.exists(model_weights_path):
            self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device, weights_only=True))
            self.model.eval()

        # Dedicated expert sub-executors
        self.crop_portfolio = crop_farmer_portfolio
        self.crop_carrot = crop_farmer_carrot
        self.crop_wheat = crop_farmer_wheat
        self.crop_melon = crop_farmer_melon
        self.crop_strawberry = crop_farmer_strawberry
        self.crop_tomato = crop_farmer_tomato
        self.hybrid_expert = hybrid_expert_agent
        self.livestock_bot = livestock_agent
        self.arbitrage_bot = arbitrage_agent

    def search_best_action(self, obs: Dict[str, Any]) -> Tuple[int, np.ndarray, float]:
        """Runs MCTS search with dynamic action masking from the current observation root node."""
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_probs, root_value = self.model.predict(grid_t, scalars_t)

        probs_np = action_probs.squeeze(0).cpu().numpy()
        val_float = float(root_value.item())

        mask = compute_action_mask(obs) if self.use_action_mask else np.ones(NUM_MACRO_ACTIONS, dtype=bool)
        probs_np = probs_np * mask
        p_sum = probs_np.sum()
        if p_sum > 0:
            probs_np = probs_np / p_sum
        else:
            probs_np = mask.astype(np.float32) / max(1, mask.sum())

        root = MCTSNode(prior=1.0)
        root.expand(probs_np)

        for _ in range(self.num_simulations):
            node = root
            best_action = -1
            best_score = -float("inf")
            for action_idx, child in node.children.items():
                if self.use_action_mask and not mask[action_idx]:
                    continue
                score = child.get_puct_score(parent_visits=node.visit_count, c_puct=self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action = action_idx

            if best_action != -1:
                selected_child = node.children[best_action]
                selected_child.update(val_float)
                root.update(val_float)

        visit_counts = np.array([root.children[a].visit_count for a in range(NUM_MACRO_ACTIONS)], dtype=np.float32)
        if self.temperature == 0.0 or visit_counts.sum() == 0:
            best_action = int(np.argmax(probs_np))
        else:
            exp_counts = visit_counts ** (1.0 / self.temperature)
            action_dist = exp_counts / exp_counts.sum()
            best_action = int(np.random.choice(len(action_dist), p=action_dist))

        return best_action, probs_np, val_float

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        best_action, probs, val = self.search_best_action(obs)

        # Macro-Action Translation:
        # 0: FARM_CARROTS_INTENSIVE -> Intensive carrot planting & rapid harvest
        # 1: FARM_WHEAT_EXPANSION   -> Wheat production for staple / feed reserves
        # 2: FARM_DIVERSIFIED       -> Portfolio multi-crop (Melon, Tomato, Strawberry, Carrot)
        # 3: BUY_LAND_EXPANSION     -> Land expansion priority + hybrid farming
        # 4: HIRE_EXTRA_LABOR       -> Labor recruitment + hybrid chores
        # 5: BUILD_GOOSE_COOP       -> Coop & Goose production
        # 6: BUILD_PASTURE_LIVESTOCK-> Pasture & Cow/Sheep production
        # 7: HARVEST_AND_LIQUIDATE  -> Full harvest & complete market inventory sell-off
        # 8: MARKET_ARBITRAGE_TRADE -> Unlocked town shop arbitrage trading
        # 9: LIVESTOCK_CARE_FEED    -> Animal feeding, daily care, fertilizer harvesting
        if best_action == 0:
            return self.crop_carrot(obs, config)
        elif best_action == 1:
            return self.crop_wheat(obs, config)
        elif best_action == 2:
            return self.crop_portfolio(obs, config)
        elif best_action in [3, 4]:
            return self.hybrid_expert(obs, config)
        elif best_action == 5:
            return self.hybrid_expert(obs, config)
        elif best_action == 6:
            return self.livestock_bot(obs, config)
        elif best_action == 7:
            res = self.hybrid_expert(obs, config)
            shed = obs.get("private", {}).get("shed", {})
            orders = []
            for item, count in shed.items():
                if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                    orders.append(["SELL", item, count])
            res["market"] = orders[:10]
            return res
        elif best_action == 8:
            return self.arbitrage_bot(obs, config)
        elif best_action == 9:
            return self.livestock_bot(obs, config)
        else:
            return self.hybrid_expert(obs, config)

