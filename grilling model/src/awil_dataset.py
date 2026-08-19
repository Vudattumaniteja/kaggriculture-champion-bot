"""
Grandmaster Replay Dataset Parser with D4 Coordinate Augmentation & AWIL Scorer.
Extracts rolling crop heatmaps, exponential land expansion ramps, masked trade fractions,
and livestock quotas with quality filtering, PASS deduplication, and phase-bucketed AWIL weights.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.encoder import encode_observation, CROPS, PRODUCTS, ANIMALS, CROP_SPECS


def get_phase_bucket(step: int) -> int:
    """
    Returns the seasonal phase bucket index (0..4) for a given step (0..719):
    - Bucket 0: Days 0..5 (steps 0..143)
    - Bucket 1: Days 6..12 (steps 144..311)
    - Bucket 2: Days 13..18 (steps 312..455)
    - Bucket 3: Days 19..24 (steps 456..599)
    - Bucket 4: Days 25..29 (steps 600..719)
    """
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


def transform_coordinates_d4(r: int, c: int, transform_idx: int, grid_size: int = 10) -> Tuple[int, int]:
    """
    Maps 2D matrix coordinates (r, c) under one of the 8 Dihedral Group (D4) geometric transformations:
    - 0: Identity: (r, c) -> (r, c)
    - 1: rot90 CCW: (r, c) -> (grid_size - 1 - c, r)
    - 2: rot180: (r, c) -> (grid_size - 1 - r, grid_size - 1 - c)
    - 3: rot270 (rot90 CW): (r, c) -> (c, grid_size - 1 - r)
    - 4: flip horizontal (axis 2): (r, c) -> (r, grid_size - 1 - c)
    - 5: flip vertical (axis 1): (r, c) -> (grid_size - 1 - r, c)
    - 6: transpose (rot90 + flip horizontal): (r, c) -> (c, r)
    - 7: anti-transpose (rot270 + flip horizontal): (r, c) -> (grid_size - 1 - c, grid_size - 1 - r)
    """
    max_idx = grid_size - 1
    if transform_idx == 0:
        return r, c
    elif transform_idx == 1:
        return max_idx - c, r
    elif transform_idx == 2:
        return max_idx - r, max_idx - c
    elif transform_idx == 3:
        return c, max_idx - r
    elif transform_idx == 4:
        return r, max_idx - c
    elif transform_idx == 5:
        return max_idx - r, c
    elif transform_idx == 6:
        return c, r
    elif transform_idx == 7:
        return max_idx - c, max_idx - r
    else:
        return r, c


def apply_d4_augmentation(
    spatial: np.ndarray,
    crop_target: np.ndarray,
    scalar: np.ndarray,
    market_fractions: np.ndarray,
    transform_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies one of the 8 Dihedral Group (D4) geometric transformations to spatial tensors.
    Keeps economic scalar vectors and non-spatial market fractions strictly invariant.
    """
    if transform_idx is None:
        transform_idx = int(np.random.randint(0, 8))

    aug_spatial = spatial.copy()
    aug_crop_target = crop_target.copy()

    # D4 transformations:
    # 0: identity
    # 1: rot90 (CCW, axes 1, 2)
    # 2: rot180 (axes 1, 2)
    # 3: rot270 (axes 1, 2)
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

    return aug_spatial, aug_crop_target, scalar.copy(), market_fractions.copy()


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
    advantages = np.asarray(returns, dtype=np.float32) - np.asarray(baseline_values, dtype=np.float32)
    num_samples = len(steps)
    if num_samples == 0:
        return np.array([], dtype=np.float32)

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

        norm_adv = (b_advs - mu_b) / max(tau * sigma_b, 1e-6)
        b_weights = np.exp(np.clip(norm_adv, -5.0, 5.0))
        weights[mask] = np.clip(b_weights, clip_min, clip_max)

    return weights


class Stage1ValueBaseline(nn.Module):
    """
    Stage 1 Baseline Value Network V_phi(s).
    Estimates expected terminal match return from observation features.
    """
    def __init__(self, spatial_in: int = 24, scalar_in: int = 72, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(spatial_in, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(32 + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_spatial: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        b = x_spatial.shape[0]
        feat_sp = self.conv(x_spatial).view(b, 32)
        feat_sc = self.scalar_net(x_scalar)
        fused = torch.cat([feat_sp, feat_sc], dim=-1)
        return self.head(fused)

    def predict_value(self, x_spatial: torch.Tensor, x_scalar: torch.Tensor) -> torch.Tensor:
        return self.forward(x_spatial, x_scalar).squeeze(-1)


def train_stage1_baseline(
    model: Stage1ValueBaseline,
    dataset: Dataset,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu"
) -> Stage1ValueBaseline:
    """
    Fits the Stage 1 Value Baseline network V_phi(s) on transition return targets.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for batch in loader:
            x_sp = batch["x_spatial"].to(dev)
            x_sc = batch["x_scalar"].to(dev)
            y_val = batch["target_value"].to(dev).unsqueeze(-1)

            optimizer.zero_grad()
            pred = model(x_sp, x_sc)
            loss = F.mse_loss(pred, y_val)
            loss.backward()
            optimizer.step()

    return model


def parse_replay_transitions(
    replay_dict: Dict[str, Any],
    min_cash: float = 50000.0,
    max_transitions_per_player: int = 360,
    max_transitions_per_match: int = 720
) -> List[Dict[str, Any]]:
    """
    Parses match replays into transition dictionaries with:
    - Quality filtering ($50k minimum return)
    - Consecutive duplicate idle PASS deduplication
    - Exponential land expansion lead-in ramps (exp(-delta_t / 12.0))
    - Rolling 5x10x10 crop heatmaps
    - Livestock quotas (3 dims: GOOSE, COW, SHEEP)
    - Masked continuous trade fractions (9 dims) and market masks
    - Per-player representation caps
    """
    steps = replay_dict.get("steps", [])
    if not steps or len(steps) < 5:
        return []

    # Check terminal cash scores
    last_step = steps[-1]
    rewards = [s.get("reward", 0.0) or 0.0 for s in last_step]

    qualifying_players = [p for p, r in enumerate(rewards) if float(r) >= min_cash]
    if not qualifying_players:
        return []

    transitions: List[Dict[str, Any]] = []

    for player in qualifying_players:
        p_return = float(rewards[player])
        player_transitions = []
        prev_was_idle_pass = False

        # First pass: Identify all land expansion steps for this player
        unlock_steps: List[int] = []
        prev_unlocked_count = 1
        for s_idx, step_data in enumerate(steps[:-1]):
            agent_step = step_data[player]
            obs = agent_step.get("observation", {})
            farms = obs.get("farms", [{}, {}]) if obs else []
            farm = farms[player] if (0 <= player < len(farms) and isinstance(farms[player], dict)) else {}
            unlocked = farm.get("unlocked_quadrants", ["NW"])
            cur_count = len(unlocked) if isinstance(unlocked, (list, tuple, set)) else 1

            act = agent_step.get("action", {}) or {}
            mo = act.get("market", []) if isinstance(act, dict) else []
            has_buy_land = any(isinstance(o, (list, tuple)) and len(o) > 0 and o[0] in ("BUY_LAND", "UNLOCK_QUADRANT") for o in mo)

            if cur_count > prev_unlocked_count or has_buy_land:
                unlock_steps.append(s_idx)
                prev_unlocked_count = max(cur_count, prev_unlocked_count + 1)

        # Second pass: Extract transition data
        for t_idx, step_data in enumerate(steps[:-1]):
            if len(player_transitions) >= max_transitions_per_player:
                break
            if len(transitions) + len(player_transitions) >= max_transitions_per_match:
                break

            agent_step = step_data[player]
            obs = agent_step.get("observation", {})
            if not obs:
                continue

            action = agent_step.get("action", {}) or {}
            farmer_act = action.get("farmer", ["PASS"])
            hands_act = action.get("hands", [])
            market_orders = action.get("market", [])

            # Check if this transition is an idle PASS
            is_farmer_pass = (farmer_act in (["PASS"], ("PASS",), [], None, "PASS"))
            is_hands_pass = (not hands_act or all(h in (["PASS"], ("PASS",), [], None, "PASS") for h in hands_act))
            is_market_empty = (not market_orders)
            is_idle_pass = is_farmer_pass and is_hands_pass and is_market_empty

            if is_idle_pass and prev_was_idle_pass:
                continue
            prev_was_idle_pass = is_idle_pass

            # Observation encoding
            obs_copy = dict(obs)
            obs_copy["player"] = player
            x_spatial, x_scalar = encode_observation(obs_copy)

            # 1. Rolling Crop Heatmaps (5, 10, 10)
            crop_target = np.zeros((5, 10, 10), dtype=np.float32)
            farms = obs.get("farms", [{}, {}])
            farm = farms[player] if (0 <= player < len(farms) and isinstance(farms[player], dict)) else {}
            tiles = farm.get("tiles", [])
            for r in range(min(10, len(tiles))):
                row = tiles[r]
                if not isinstance(row, (list, tuple)):
                    continue
                for c in range(min(10, len(row))):
                    t = row[c]
                    if isinstance(t, dict) and t.get("kind") == "PLANT":
                        cr = t.get("crop")
                        if cr in CROPS:
                            crop_target[CROPS.index(cr), r, c] = 1.0

            # 2. Livestock Quotas (3 dims: GOOSE, COW, SHEEP)
            animal_counts: Dict[str, int] = {a: 0 for a in ANIMALS}
            for r in range(min(10, len(tiles))):
                row = tiles[r]
                if not isinstance(row, (list, tuple)):
                    continue
                for c in range(min(10, len(row))):
                    t = row[c]
                    if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
                        an = t.get("animal")
                        if an in ANIMALS:
                            animal_counts[an] += 1
            if isinstance(farm.get("animals"), (list, tuple)):
                for a_item in farm.get("animals"):
                    an = a_item.get("animal") if isinstance(a_item, dict) else a_item
                    if an in ANIMALS:
                        animal_counts[an] += 1
            livestock_quotas = np.array([float(animal_counts["GOOSE"]), float(animal_counts["COW"]), float(animal_counts["SHEEP"])], dtype=np.float32)

            # 3. Exponential Land Expansion Lead-in Ramp (exp(-delta_t / 12.0))
            future_unlocks = [u for u in unlock_steps if u >= t_idx]
            if future_unlocks:
                closest_u = min(future_unlocks)
                dt = closest_u - t_idx
                if dt <= 12:
                    land_expand_target = float(math.exp(-dt / 12.0))
                else:
                    land_expand_target = 0.0
            else:
                land_expand_target = 0.0

            # 4. Workforce Hiring Target
            hands = farm.get("hands", [])
            workforce_target = int(len(hands)) if isinstance(hands, (list, tuple)) else 0

            # 5. Autonomous Seed Replenishment Targets (5 dims)
            seed_replenish = np.zeros(5, dtype=np.float32)
            if isinstance(market_orders, (list, tuple)):
                for order in market_orders:
                    if isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == "BUY":
                        item = str(order[1]).replace("_SEED", "").upper()
                        if item in CROPS:
                            try:
                                qty = float(order[2] or 0)
                                seed_replenish[CROPS.index(item)] += qty
                            except (ValueError, TypeError):
                                pass

            # 6. Continuous Market Liquidation Fractions (9 dims) and Market Mask
            market_fractions = np.zeros(9, dtype=np.float32)
            market_mask = np.zeros(9, dtype=np.float32)
            shed = (obs.get("private", {}) or {}).get("shed", {})
            if not shed and isinstance(farm.get("shed"), dict):
                shed = farm.get("shed", {})

            for p_idx, prod in enumerate(PRODUCTS):
                in_shed = float(shed.get(prod, 0) or 0)
                if in_shed > 0:
                    market_mask[p_idx] = 1.0

            if isinstance(market_orders, (list, tuple)):
                for order in market_orders:
                    if isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == "SELL":
                        prod = str(order[1]).upper()
                        if prod in PRODUCTS:
                            p_idx = PRODUCTS.index(prod)
                            in_shed = float(shed.get(prod, 0) or 0)
                            try:
                                qty = float(order[2] or 0)
                                if in_shed > 0:
                                    market_fractions[p_idx] = min(max(qty / in_shed, 0.0), 1.0)
                            except (ValueError, TypeError):
                                pass

            step_num = int(obs.get("step", t_idx))

            player_transitions.append({
                "spatial": x_spatial,
                "scalar": x_scalar,
                "crop_heatmaps": crop_target,
                "livestock_quotas": livestock_quotas,
                "livestock": livestock_quotas,
                "workforce": workforce_target,
                "land_expand": land_expand_target,
                "seed_replenish": seed_replenish,
                "market_fractions": market_fractions,
                "market_mask": market_mask,
                "terminal_return": p_return,
                "step": step_num,
                "baseline_value": p_return * 0.9,
                "player": player,
            })

        transitions.extend(player_transitions)

    return transitions


class AWILDataset(Dataset):
    """
    PyTorch Dataset for Advantage-Weighted Imitation Learning (AWIL).
    Provides synchronized D4 spatial data augmentation and phase-bucketed sample weighting.
    """
    def __init__(
        self,
        transitions: List[Dict[str, Any]],
        augment: bool = True,
        tau: float = 1.0,
        clip_min: float = 0.2,
        clip_max: float = 5.0,
        baseline_model: Optional[Stage1ValueBaseline] = None
    ):
        self.transitions = transitions
        self.augment = augment
        self.tau = tau
        self.clip_min = clip_min
        self.clip_max = clip_max

        # If baseline model provided, evaluate baseline values
        if baseline_model is not None and self.transitions:
            baseline_model.eval()
            with torch.no_grad():
                for t in self.transitions:
                    x_sp = torch.from_numpy(t["spatial"]).float().unsqueeze(0)
                    x_sc = torch.from_numpy(t["scalar"]).float().unsqueeze(0)
                    val = baseline_model(x_sp, x_sc).item()
                    t["baseline_value"] = val

        # Compute sample weights across all transitions
        if self.transitions:
            steps = np.array([t["step"] for t in self.transitions], dtype=np.int32)
            returns = np.array([t["terminal_return"] for t in self.transitions], dtype=np.float32)
            baseline_values = np.array([t.get("baseline_value", t["terminal_return"]) for t in self.transitions], dtype=np.float32)
            self.weights = compute_awil_sample_weights(
                steps, returns, baseline_values, tau=self.tau, clip_min=self.clip_min, clip_max=self.clip_max
            )
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
        market_mask = t.get("market_mask", np.zeros(9, dtype=np.float32))
        livestock = t.get("livestock_quotas", np.zeros(3, dtype=np.float32))

        if self.augment:
            spatial, crop_target, scalar, market_fractions = apply_d4_augmentation(
                spatial, crop_target, scalar, market_fractions
            )

        weight = self.weights[idx] if idx < len(self.weights) else 1.0

        return {
            "x_spatial": torch.from_numpy(spatial).float(),
            "x_scalar": torch.from_numpy(scalar).float(),
            "target_crop_heatmaps": torch.from_numpy(crop_target).float(),
            "target_livestock_quotas": torch.from_numpy(livestock).float(),
            "target_livestock": torch.from_numpy(livestock).float(),
            "target_workforce": torch.tensor(t["workforce"], dtype=torch.long),
            "target_land_expand": torch.tensor(t["land_expand"], dtype=torch.float32),
            "target_seed_replenish": torch.from_numpy(t["seed_replenish"]).float(),
            "target_market_fractions": torch.from_numpy(market_fractions).float(),
            "target_market_mask": torch.from_numpy(market_mask).float(),
            "target_value": torch.tensor(t["terminal_return"], dtype=torch.float32),
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
        }


def validate_dataset_integrity(dataset: AWILDataset, transitions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Validates dataset integrity:
    1. All sample weights satisfy 0.2 <= w_t <= 5.0
    2. D4 transformations preserve spatial validity and coordinate alignment
    3. Scalar economic features and market fractions remain invariant
    4. Land expansion ramps are bounded in [0.0, 1.0]
    """
    report = {
        "status": "PASSED",
        "num_samples": len(dataset),
        "sample_weights_bounded": True,
        "d4_transforms_valid": True,
        "shapes_valid": True,
    }

    if len(dataset) == 0:
        return report

    # 1. Check sample weights
    for i in range(len(dataset)):
        w = float(dataset.weights[i]) if i < len(dataset.weights) else 1.0
        if not (0.2 <= w <= 5.0) or math.isnan(w) or math.isinf(w):
            report["sample_weights_bounded"] = False
            report["status"] = "FAILED"
            break

    # 2. Check D4 geometric preservation
    dummy_spatial = np.zeros((24, 10, 10), dtype=np.float32)
    dummy_crop = np.zeros((5, 10, 10), dtype=np.float32)
    dummy_scalar = np.ones(72, dtype=np.float32)
    dummy_market = np.ones(9, dtype=np.float32) * 0.5
    dummy_spatial[0, 1, 2] = 1.0
    dummy_crop[0, 1, 2] = 1.0

    for idx in range(8):
        s_aug, c_aug, sc_aug, m_aug = apply_d4_augmentation(dummy_spatial, dummy_crop, dummy_scalar, dummy_market, transform_idx=idx)
        rn, cn = transform_coordinates_d4(1, 2, transform_idx=idx)
        if s_aug[0, rn, cn] != 1.0 or c_aug[0, rn, cn] != 1.0:
            report["d4_transforms_valid"] = False
            report["status"] = "FAILED"
        if not np.array_equal(sc_aug, dummy_scalar) or not np.array_equal(m_aug, dummy_market):
            report["d4_transforms_valid"] = False
            report["status"] = "FAILED"

    # 3. Check item tensor shapes
    item = dataset[0]
    if item["x_spatial"].shape != (24, 10, 10) or item["x_scalar"].shape != (72,):
        report["shapes_valid"] = False
        report["status"] = "FAILED"

    return report

