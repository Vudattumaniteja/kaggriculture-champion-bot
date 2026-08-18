"""
Grandmaster Replay Dataset Parser with D4 Coordinate Augmentation & AWIL Scorer.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from src.encoder import encode_observation, CROPS, PRODUCTS, ANIMALS, CROP_SPECS


def get_phase_bucket(step: int) -> int:
    """Returns the seasonal phase bucket index (0..4) for a given step (0..719)."""
    day = step // 24
    if day <= 5:
        return 0
    elif day <= 12:
        return 1
    elif day <= 18:
        return 2
    elif day <= 24:
        return 3
    else:
        return 4


def apply_d4_augmentation(
    spatial: np.ndarray,
    crop_target: np.ndarray,
    scalar: np.ndarray,
    market_fractions: np.ndarray,
    transform_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies one of the 8 Dihedral Group (D4) geometric transformations to spatial tensors.
    Keeps economic scalar vectors and non-spatial market fractions invariant.
    """
    if transform_idx is None:
        transform_idx = int(np.random.randint(0, 8))

    aug_spatial = spatial.copy()
    aug_crop_target = crop_target.copy()

    # D4 transformations:
    # 0: identity
    # 1: rot90 (1)
    # 2: rot90 (2)
    # 3: rot90 (3)
    # 4: flip horizontal (axis=2)
    # 5: flip vertical (axis=1)
    # 6: transpose (rot90 + flip)
    # 7: anti-transpose (rot270 + flip)
    if transform_idx == 1:
        aug_spatial = np.rot90(aug_spatial, k=1, axes=(1, 2)).copy()
        aug_crop_target = np.rot90(aug_crop_target, k=1, axes=(1, 2)).copy()
    elif transform_idx == 2:
        aug_spatial = np.rot90(aug_spatial, k=2, axes=(1, 2)).copy()
        aug_crop_target = np.rot90(aug_crop_target, k=2, axes=(1, 2)).copy()
    elif transform_idx == 3:
        aug_spatial = np.rot90(aug_spatial, k=3, axes=(1, 2)).copy()
        aug_crop_target = np.rot90(aug_crop_target, k=3, axes=(1, 2)).copy()
    elif transform_idx == 4:
        aug_spatial = np.flip(aug_spatial, axis=2).copy()
        aug_crop_target = np.flip(aug_crop_target, axis=2).copy()
    elif transform_idx == 5:
        aug_spatial = np.flip(aug_spatial, axis=1).copy()
        aug_crop_target = np.flip(aug_crop_target, axis=1).copy()
    elif transform_idx == 6:
        aug_spatial = np.rot90(np.flip(aug_spatial, axis=2), k=1, axes=(1, 2)).copy()
        aug_crop_target = np.rot90(np.flip(aug_crop_target, axis=2), k=1, axes=(1, 2)).copy()
    elif transform_idx == 7:
        aug_spatial = np.rot90(np.flip(aug_spatial, axis=1), k=1, axes=(1, 2)).copy()
        aug_crop_target = np.rot90(np.flip(aug_crop_target, axis=1), k=1, axes=(1, 2)).copy()

    return aug_spatial, aug_crop_target, scalar, market_fractions


def compute_awil_sample_weights(
    steps: np.ndarray,
    returns: np.ndarray,
    baseline_values: np.ndarray,
    tau: float = 1.0,
    clip_min: float = 0.2,
    clip_max: float = 5.0
) -> np.ndarray:
    """
    Computes phase-bucketed, normalized, and clipped AWIL sample loss weights.
    w_t = clip(exp((A_t - mu_b) / (tau * sigma_b)), 0.2, 5.0)
    """
    advantages = returns - baseline_values
    num_samples = len(steps)
    weights = np.ones(num_samples, dtype=np.float32)

    buckets = np.array([get_phase_bucket(int(s)) for s in steps])

    for b in range(5):
        mask = (buckets == b)
        if not np.any(mask):
            continue
        b_advs = advantages[mask]
        mu_b = float(np.mean(b_advs))
        sigma_b = float(np.std(b_advs))
        if sigma_b < 1e-4:
            sigma_b = 1.0

        norm_adv = (b_advs - mu_b) / (tau * sigma_b)
        b_weights = np.exp(np.clip(norm_adv, -5.0, 5.0))
        weights[mask] = np.clip(b_weights, clip_min, clip_max)

    return weights


def parse_replay_transitions(
    replay_dict: Dict[str, Any],
    min_cash: float = 50000.0,
    max_transitions_per_match: int = 720
) -> List[Dict[str, Any]]:
    """
    Parses full match replays into transition dictionaries with quality filtering ($50k minimum return).
    """
    steps = replay_dict.get("steps", [])
    if not steps or len(steps) < 10:
        return []

    # Check terminal cash scores
    last_step = steps[-1]
    rewards = [s.get("reward", 0.0) or 0.0 for s in last_step]

    qualifying_players = [p for p, r in enumerate(rewards) if r >= min_cash]
    if not qualifying_players:
        return []

    transitions: List[Dict[str, Any]] = []

    for player in qualifying_players:
        p_return = float(rewards[player])
        prev_farmer_act = None

        for t_idx, step_data in enumerate(steps[:-1]):
            agent_step = step_data[player]
            obs = agent_step.get("observation", {})
            if not obs:
                continue

            action = agent_step.get("action", {}) or {}
            farmer_act = action.get("farmer", ["PASS"])

            # Deduplicate consecutive identical PASS actions
            if farmer_act == ["PASS"] and prev_farmer_act == ["PASS"]:
                continue
            prev_farmer_act = farmer_act

            # Encode state
            obs["player"] = player
            x_spatial, x_scalar = encode_observation(obs)

            # Targets
            crop_target = np.zeros((5, 10, 10), dtype=np.float32)
            farm = obs.get("farms", [{}, {}])[player]
            tiles = farm.get("tiles", [])
            for r in range(min(10, len(tiles))):
                for c in range(min(10, len(tiles[r]))):
                    t = tiles[r][c]
                    if isinstance(t, dict) and t.get("kind") == "PLANT":
                        cr = t.get("crop")
                        if cr in CROPS:
                            crop_target[CROPS.index(cr), r, c] = 1.0

            workforce_target = len(farm.get("hands", []))
            unlocked_quads = farm.get("unlocked_quadrants", ["NW"])
            land_expand_target = 1.0 if len(unlocked_quads) > 1 else 0.0

            market_orders = action.get("market", [])
            market_fractions = np.zeros(9, dtype=np.float32)
            shed = (obs.get("private", {}) or {}).get("shed", {})
            for order in market_orders:
                if len(order) >= 3 and order[0] == "SELL":
                    prod = order[1]
                    if prod in PRODUCTS:
                        p_idx = PRODUCTS.index(prod)
                        qty = order[2]
                        in_shed = shed.get(prod, 0)
                        if in_shed > 0:
                            market_fractions[p_idx] = min(float(qty) / float(in_shed), 1.0)

            transitions.append({
                "spatial": x_spatial,
                "scalar": x_scalar,
                "crop_heatmaps": crop_target,
                "workforce": workforce_target,
                "land_expand": land_expand_target,
                "seed_replenish": np.zeros(5, dtype=np.float32),
                "market_fractions": market_fractions,
                "terminal_return": p_return,
                "step": int(obs.get("step", t_idx)),
                "baseline_value": p_return * 0.9,  # Default baseline estimate
            })

            if len(transitions) >= max_transitions_per_match:
                break

    return transitions


class AWILDataset(Dataset):
    """
    PyTorch Dataset for Advantage-Weighted Imitation Learning (AWIL).
    Provides synchronized D4 spatial data augmentation and sample weighting.
    """
    def __init__(self, transitions: List[Dict[str, Any]], augment: bool = True, tau: float = 1.0):
        self.transitions = transitions
        self.augment = augment
        self.tau = tau

        # Compute sample weights across all transitions
        if self.transitions:
            steps = np.array([t["step"] for t in self.transitions], dtype=np.int32)
            returns = np.array([t["terminal_return"] for t in self.transitions], dtype=np.float32)
            baseline_values = np.array([t.get("baseline_value", t["terminal_return"]) for t in self.transitions], dtype=np.float32)
            self.weights = compute_awil_sample_weights(steps, returns, baseline_values, tau=self.tau)
        else:
            self.weights = np.array([], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.transitions[idx]
        spatial = t["spatial"]
        crop_target = t["crop_heatmaps"]
        scalar = t["scalar"]
        market_fractions = t["market_fractions"]

        if self.augment:
            spatial, crop_target, scalar, market_fractions = apply_d4_augmentation(
                spatial, crop_target, scalar, market_fractions
            )

        return {
            "x_spatial": torch.from_numpy(spatial).float(),
            "x_scalar": torch.from_numpy(scalar).float(),
            "target_crop_heatmaps": torch.from_numpy(crop_target).float(),
            "target_workforce": torch.tensor(t["workforce"], dtype=torch.long),
            "target_land_expand": torch.tensor(t["land_expand"], dtype=torch.float32),
            "target_seed_replenish": torch.from_numpy(t["seed_replenish"]).float(),
            "target_market_fractions": torch.from_numpy(market_fractions).float(),
            "target_value": torch.tensor(t["terminal_return"], dtype=torch.float32),
            "sample_weight": torch.tensor(self.weights[idx], dtype=torch.float32),
        }
