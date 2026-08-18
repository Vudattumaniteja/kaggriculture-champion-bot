"""
500k Verified Prioritized Experience Replay (PER) Buffer with Binary SumTree.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import torch


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
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

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


class PrioritizedReplayBuffer:
    """
    500k Prioritized Experience Replay (PER) buffer with importance sampling correction.
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
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

        self.tree.add(priority, transition)

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
                # Fallback to random entry
                idx, p, data = self.tree.get(np.random.uniform(0, total_p))
            batch_data.append(data)
            tree_indices.append(idx)
            priorities.append(max(p, self.epsilon))

        # Compute importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        N = len(self)
        probs = np.array(priorities, dtype=np.float64) / total_p
        weights = (N * probs) ** (-self.beta)
        weights = (weights / np.max(weights)).astype(np.float32)

        # Collate batch dictionary
        keys = batch_data[0].keys()
        collated_batch = {k: np.array([item[k] for item in batch_data]) for k in keys}

        return collated_batch, tree_indices, weights

    def update_priorities(self, tree_indices: List[int], td_errors: np.ndarray):
        for idx, err in zip(tree_indices, td_errors):
            priority = (abs(float(err)) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
