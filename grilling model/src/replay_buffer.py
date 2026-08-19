"""
500k Verified Prioritized Experience Replay (PER) Buffer with Binary SumTree
and Causal Action Logger for Hungarian Dispatcher and Market Guardrails.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import os
import numpy as np
import torch

from src.encoder import encode_observation, CROPS, PRODUCTS


class SumTree:
    """
    Binary SumTree data structure for O(log N) proportional priority sampling.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: List[Optional[Any]] = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _retrieve(self, idx: int, s: float) -> int:
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx

            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                if right < len(self.tree):
                    idx = right
                else:
                    idx = left

    def total_priority(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Any):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        if idx != 0:
            self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


def extract_verified_macro_actions(
    obs: Dict[str, Any],
    farmer_action: List[Any],
    hands_actions: List[List[Any]],
    market_orders: List[List[Any]],
    raw_intents: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts verified executed macro actions (a_eff) from actual Hungarian dispatch
    and market guardrail executions to prevent replay buffer corruption.
    """
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if (0 <= player < len(farms) and isinstance(farms[player], dict)) else {}

    # 1. Target Crop Allocations (5 x 10 x 10)
    target_crop = np.zeros((5, 10, 10), dtype=np.float32)
    tiles = my_farm.get("tiles", [])
    for r in range(min(10, len(tiles))):
        row = tiles[r]
        if not isinstance(row, (list, tuple)):
            continue
        for c in range(min(10, len(row))):
            t = row[c]
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                crop = t.get("crop")
                if crop in CROPS:
                    target_crop[CROPS.index(crop), r, c] = 1.0

    # Include in-step executed PLANT actions
    if isinstance(farmer_action, (list, tuple)) and len(farmer_action) >= 2 and farmer_action[0] == "PLANT":
        crop = str(farmer_action[1]).upper()
        if crop in CROPS:
            farmer_pos = my_farm.get("farmer", [4, 4])
            if isinstance(farmer_pos, (list, tuple)) and len(farmer_pos) >= 2:
                fx, fy = int(farmer_pos[0]), int(farmer_pos[1])
                if 0 <= fy < 10 and 0 <= fx < 10:
                    target_crop[CROPS.index(crop), fy, fx] = 1.0

    hands_pos_list = my_farm.get("hands", [])
    if isinstance(hands_actions, (list, tuple)):
        for h_idx, h_act in enumerate(hands_actions):
            if isinstance(h_act, (list, tuple)) and len(h_act) >= 2 and h_act[0] == "PLANT":
                crop = str(h_act[1]).upper()
                if crop in CROPS and h_idx < len(hands_pos_list):
                    h_pos = hands_pos_list[h_idx]
                    if isinstance(h_pos, (list, tuple)) and len(h_pos) >= 2:
                        hx, hy = int(h_pos[0]), int(h_pos[1])
                        if 0 <= hy < 10 and 0 <= hx < 10:
                            target_crop[CROPS.index(crop), hy, hx] = 1.0

    # 2. Verified Workforce Level
    hands = my_farm.get("hands", [])
    num_hands = len(hands) if isinstance(hands, (list, tuple)) else 0
    has_hire = False
    if isinstance(market_orders, (list, tuple)):
        for order in market_orders:
            if isinstance(order, (list, tuple)) and len(order) > 0 and order[0] == "HIRE":
                has_hire = True
                break
    target_workforce = num_hands + 1 if has_hire else num_hands

    # 3. Verified Land Expansion
    has_buy_land = False
    if isinstance(market_orders, (list, tuple)):
        for order in market_orders:
            if isinstance(order, (list, tuple)) and len(order) > 0 and order[0] in ("BUY_LAND", "UNLOCK_QUADRANT"):
                has_buy_land = True
                break
    target_land = 1.0 if has_buy_land else 0.0

    # 4. Verified Seed Replenishment (5 dims)
    target_seed = np.zeros(5, dtype=np.float32)
    if isinstance(market_orders, (list, tuple)):
        for order in market_orders:
            if isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] in ("BUY", "BUY_SEED"):
                crop_name = str(order[1]).replace("_SEED", "").upper()
                if crop_name in CROPS:
                    try:
                        qty = float(order[2] or 0)
                        target_seed[CROPS.index(crop_name)] += qty
                    except (ValueError, TypeError):
                        pass

    # 5. Verified Market Liquidation Fractions (9 dims)
    target_market = np.zeros(9, dtype=np.float32)
    shed = (obs.get("private", {}) or {}).get("shed", {})
    if not shed and isinstance(my_farm.get("shed"), dict):
        shed = my_farm.get("shed", {})

    sold_counts = {p: 0.0 for p in PRODUCTS}
    if isinstance(market_orders, (list, tuple)):
        for order in market_orders:
            if isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == "SELL":
                prod = str(order[1]).upper()
                if prod in PRODUCTS:
                    try:
                        sold_counts[prod] += float(order[2] or 0)
                    except (ValueError, TypeError):
                        pass

    for p_idx, prod in enumerate(PRODUCTS):
        in_shed = float(shed.get(prod, 0) or 0)
        if in_shed > 0:
            target_market[p_idx] = min(1.0, max(0.0, sold_counts[prod] / in_shed))
        else:
            target_market[p_idx] = 0.0

    result = {
        "target_crop": target_crop,
        "target_workforce": target_workforce,
        "target_land": target_land,
        "target_seed": target_seed,
        "target_market": target_market,
    }
    if raw_intents is not None:
        result["raw_intents"] = raw_intents

    return result


class CausalActionLogger:
    """
    Logs verified executed causal actions (a_eff) dispatched by the Hungarian engine
    and market guardrails to prevent replay buffer corruption from unexecutable intents.
    """
    @staticmethod
    def log_step(
        obs: Dict[str, Any],
        farmer_action: List[Any],
        hands_actions: List[List[Any]],
        market_orders: List[List[Any]],
        raw_intents: Optional[Dict[str, Any]] = None,
        x_spatial: Optional[np.ndarray] = None,
        x_scalar: Optional[np.ndarray] = None,
        reward: float = 0.0,
        next_obs: Optional[Dict[str, Any]] = None,
        done: bool = False,
        value_target: Optional[float] = None,
    ) -> Dict[str, Any]:
        if x_spatial is None or x_scalar is None:
            x_sp, x_sc = encode_observation(obs)
            x_spatial = x_spatial if x_spatial is not None else x_sp
            x_scalar = x_scalar if x_scalar is not None else x_sc

        verified = extract_verified_macro_actions(
            obs=obs,
            farmer_action=farmer_action,
            hands_actions=hands_actions,
            market_orders=market_orders,
            raw_intents=raw_intents,
        )

        transition = {
            "x_spatial": x_spatial,
            "x_scalar": x_scalar,
            "target_crop": verified["target_crop"],
            "target_workforce": verified["target_workforce"],
            "target_land": verified["target_land"],
            "target_seed": verified["target_seed"],
            "target_market": verified["target_market"],
            "reward": float(reward),
            "done": bool(done),
            "step": int(obs.get("step", 0)),
        }

        if value_target is not None:
            transition["target_value"] = float(value_target)

        if raw_intents is not None:
            transition["raw_intents"] = raw_intents

        return transition


class PrioritizedReplayBuffer:
    """
    500k Prioritized Experience Replay (PER) buffer with importance sampling correction,
    fast batch sampling, disk serialization, and causal macro action logging.
    """
    def __init__(
        self,
        capacity: int = 500000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 100000,
        epsilon: float = 1e-5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self.step_count = 0
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self) -> int:
        return self.tree.size

    def add(self, transition: Dict[str, Any], td_error: Optional[float] = None):
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

        self.tree.add(priority, transition)

    def add_verified_step(
        self,
        obs: Dict[str, Any],
        farmer_action: List[Any],
        hands_actions: List[List[Any]],
        market_orders: List[List[Any]],
        raw_intents: Optional[Dict[str, Any]] = None,
        x_spatial: Optional[np.ndarray] = None,
        x_scalar: Optional[np.ndarray] = None,
        reward: float = 0.0,
        done: bool = False,
        value_target: Optional[float] = None,
        td_error: Optional[float] = None,
    ):
        transition = CausalActionLogger.log_step(
            obs=obs,
            farmer_action=farmer_action,
            hands_actions=hands_actions,
            market_orders=market_orders,
            raw_intents=raw_intents,
            x_spatial=x_spatial,
            x_scalar=x_scalar,
            reward=reward,
            done=done,
            value_target=value_target,
        )
        self.add(transition, td_error=td_error)

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], List[int], np.ndarray]:
        batch_data: List[Dict[str, Any]] = []
        tree_indices: List[int] = []
        priorities: List[float] = []

        total_p = self.tree.total_priority()
        if total_p <= 0.0:
            total_p = 1.0

        segment = total_p / batch_size

        # Anneal beta
        self.step_count += 1
        fraction = min(1.0, float(self.step_count) / max(1, self.beta_steps))
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            if data is None:
                idx, p, data = self.tree.get(np.random.uniform(0, total_p))
            batch_data.append(data)
            tree_indices.append(idx)
            priorities.append(max(p, self.epsilon))

        # Compute importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        N = len(self)
        probs = np.array(priorities, dtype=np.float64) / total_p
        weights = (N * probs) ** (-self.beta)
        max_w = np.max(weights)
        if max_w > 0:
            weights = (weights / max_w).astype(np.float32)
        else:
            weights = np.ones(batch_size, dtype=np.float32)

        # Collate batch dictionary
        keys = batch_data[0].keys()
        collated_batch = {k: np.array([item[k] for item in batch_data]) for k in keys}

        return collated_batch, tree_indices, weights

    def update_priorities(self, tree_indices: List[int], td_errors: Union[np.ndarray, List[float], torch.Tensor]):
        if isinstance(td_errors, torch.Tensor):
            errors = td_errors.detach().cpu().numpy()
        else:
            errors = np.asarray(td_errors)

        for idx, err in zip(tree_indices, errors):
            priority = (abs(float(err)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def save_checkpoint(self, path: str):
        """
        Serializes replay buffer state to disk.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_steps": self.beta_steps,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "max_priority": self.max_priority,
            "tree_array": self.tree.tree,
            "tree_data": self.tree.data[:self.tree.size],
            "tree_write": self.tree.write,
            "tree_size": self.tree.size,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Resumes replay buffer state from disk.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.capacity = int(checkpoint["capacity"])
        self.alpha = float(checkpoint["alpha"])
        self.beta = float(checkpoint["beta"])
        self.beta_start = float(checkpoint["beta_start"])
        self.beta_end = float(checkpoint["beta_end"])
        self.beta_steps = int(checkpoint["beta_steps"])
        self.epsilon = float(checkpoint["epsilon"])
        self.step_count = int(checkpoint["step_count"])
        self.max_priority = float(checkpoint["max_priority"])

        self.tree = SumTree(self.capacity)
        self.tree.tree = np.array(checkpoint["tree_array"], dtype=np.float64)
        saved_data = checkpoint["tree_data"]
        self.tree.data = [None] * self.capacity
        for i, d in enumerate(saved_data):
            self.tree.data[i] = d
        self.tree.write = int(checkpoint["tree_write"])
        self.tree.size = int(checkpoint["tree_size"])

    def get_state(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_steps": self.beta_steps,
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "max_priority": self.max_priority,
            "tree_array": self.tree.tree.copy(),
            "tree_data": list(self.tree.data[:self.tree.size]),
            "tree_write": self.tree.write,
            "tree_size": self.tree.size,
        }

    def load_state(self, state: Dict[str, Any]):
        self.capacity = int(state["capacity"])
        self.alpha = float(state["alpha"])
        self.beta = float(state["beta"])
        self.beta_start = float(state["beta_start"])
        self.beta_end = float(state["beta_end"])
        self.beta_steps = int(state["beta_steps"])
        self.epsilon = float(state["epsilon"])
        self.step_count = int(state["step_count"])
        self.max_priority = float(state["max_priority"])

        self.tree = SumTree(self.capacity)
        self.tree.tree = np.array(state["tree_array"], dtype=np.float64)
        saved_data = state["tree_data"]
        self.tree.data = [None] * self.capacity
        for i, d in enumerate(saved_data):
            self.tree.data[i] = d
        self.tree.write = int(state["tree_write"])
        self.tree.size = int(state["tree_size"])

