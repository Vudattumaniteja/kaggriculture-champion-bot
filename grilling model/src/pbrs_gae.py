"""
Maturity-Aware Potential-Based Reward Shaping (PBRS) and Trajectory GAE Processor.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from src.encoder import CROPS, PRODUCTS, ANIMALS, CROP_SPECS, BASE_PRICES


def compute_state_potential(obs: Dict[str, Any], player: int = 0, total_turns: int = 720) -> float:
    """
    Computes the state potential Phi(s, t) with asset maturity decay schedules.
    Phi(s, t) = Cash_t + sum_k omega_k(t) * Value_k(s_t) - Deadweight_Penalty(t)
    """
    step = int(obs.get("step", 0))
    rem_turns = max(0, total_turns - step)

    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}

    cash = float(my_farm.get("money", 0.0))
    tiles = my_farm.get("tiles", [])

    # 1. Crops Potential
    crops_val = 0.0
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                crop = t.get("crop", "CARROT")
                cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                base_val = float(BASE_PRICES.get(crop, 35.0)) * 2.0
                maturation_turns = cspec["max_yield_day"] * 24

                if rem_turns >= maturation_turns:
                    w = 1.0
                elif rem_turns < 24:
                    w = 0.0
                else:
                    w = float(rem_turns - 24) / max(1.0, float(maturation_turns - 24))
                crops_val += w * base_val

    # 2. Livestock Potential (10-day break-even = 240 turns)
    animals_val = 0.0
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            if isinstance(t, dict) and t.get("kind") in ["COOP", "PASTURE"]:
                an = t.get("animal")
                if an:
                    base_an_val = 300.0 if an == "GOOSE" else 400.0
                    if rem_turns >= 240:
                        w = 1.0
                    elif rem_turns < 48:
                        w = 0.0
                    else:
                        w = float(rem_turns - 48) / (240.0 - 48.0)
                    animals_val += w * base_an_val

    # 3. Seeds Potential
    seeds_val = 0.0
    for crop in CROPS:
        count = float(seeds.get(crop, 0))
        if count <= 0:
            continue
        cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
        grow_turns = cspec["max_yield_day"] * 24
        seed_cost = float(cspec["seed"])
        w = 1.0 if rem_turns >= (grow_turns + 24) else 0.0
        seeds_val += w * seed_cost * count

    # 4. Shed Inventory Potential & Deadweight Penalty
    shed_val = 0.0
    deadweight_penalty = 0.0
    for prod in PRODUCTS:
        count = float(shed.get(prod, 0))
        if count <= 0:
            continue
        unit_price = float(BASE_PRICES.get(prod, 25.0))
        total_prod_val = unit_price * count

        if rem_turns >= 24:
            w = 1.0
        elif rem_turns >= 2:
            w = 0.5
        else:
            w = 0.0
            deadweight_penalty += 2.0 * total_prod_val  # 2.0x deadweight penalty at t >= 718

        shed_val += w * total_prod_val

    return cash + crops_val + animals_val + seeds_val + shed_val - deadweight_penalty


def compute_pbrs_step_reward(
    s_t: Dict[str, Any],
    s_next: Dict[str, Any],
    raw_reward: float = 0.0,
    gamma: float = 0.995,
    player: int = 0
) -> float:
    """
    Computes shaped step reward: r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t).
    """
    phi_t = compute_state_potential(s_t, player=player)
    phi_next = compute_state_potential(s_next, player=player)
    return float(raw_reward + gamma * phi_next - phi_t)


class TrajectoryGAEProcessor:
    """
    Computes Generalized Advantage Estimation (GAE) and target values along match trajectories.
    Parameters: gamma = 0.995, lambda = 0.95.
    """
    def __init__(self, gamma: float = 0.995, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def process_trajectory(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes:
        - rewards: Shape (T,)
        - values: Shape (T+1,) with terminal value values[-1] = 0
        Returns:
        - shaped_rewards: Shape (T,)
        - advantages: Shape (T,)
        - value_targets: Shape (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        value_targets = advantages + values[:-1]
        return rewards, advantages, value_targets
