"""
Maturity-Aware Potential-Based Reward Shaping (PBRS) and Trajectory GAE Processor.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import torch

from src.encoder import CROPS, PRODUCTS, ANIMALS, CROP_SPECS, BASE_PRICES
from src.network import value_to_two_hot, NUM_BINS, V_MIN, V_MAX


def compute_state_potential(
    obs: Dict[str, Any],
    player: int = 0,
    total_turns: int = 720,
    relative: bool = False,
    deadweight_penalty_coeff: float = 2.0
) -> float:
    """
    Computes the state potential Phi(s, t) with asset maturity decay schedules.
    Phi(s, t) = Cash_t + sum_k omega_k(t) * Value_k(s_t) - Deadweight_Penalty(t)
    
    If relative=True, returns Phi(s, t, player) - Phi(s, t, 1 - player).
    """
    if relative:
        opp = 1 - player
        phi_player = compute_state_potential(
            obs=obs,
            player=player,
            total_turns=total_turns,
            relative=False,
            deadweight_penalty_coeff=deadweight_penalty_coeff
        )
        phi_opp = compute_state_potential(
            obs=obs,
            player=opp,
            total_turns=total_turns,
            relative=False,
            deadweight_penalty_coeff=deadweight_penalty_coeff
        )
        return phi_player - phi_opp

    step = int(obs.get("step", 0))
    rem_turns = max(0, total_turns - step)

    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if (0 <= player < len(farms) and isinstance(farms[player], dict)) else {}
    
    # Extract private shed / seeds (check obs.private or farm.private)
    private = obs.get("private", {}) or {}
    if not isinstance(private, dict):
        private = {}
    if not private and "private" in my_farm and isinstance(my_farm["private"], dict):
        private = my_farm["private"]

    shed = private.get("shed", {}) or {}
    seeds = private.get("seeds", {}) or {}

    cash = float(my_farm.get("money", 0.0))
    tiles = my_farm.get("tiles", [])

    deadweight_crops = 0.0
    deadweight_animals = 0.0
    deadweight_seeds = 0.0
    deadweight_shed = 0.0

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
                    deadweight_crops += base_val
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
                        deadweight_animals += base_an_val
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
        if rem_turns >= (grow_turns + 24):
            w = 1.0
        else:
            w = 0.0
            deadweight_seeds += seed_cost * count
        seeds_val += w * seed_cost * count

    # 4. Shed Inventory Potential & Deadweight Penalty
    shed_val = 0.0
    for prod in PRODUCTS:
        count = float(shed.get(prod, 0))
        if count <= 0:
            continue
        unit_price = float(BASE_PRICES.get(prod, 25.0))
        total_prod_val = unit_price * count

        if step >= 718 or rem_turns <= 2:
            w = 0.0
            deadweight_shed += total_prod_val
        elif rem_turns >= 24:
            w = 1.0
        else:
            w = 0.5

        shed_val += w * total_prod_val

    # Deadweight Penalty enforced at Turn 718 (t >= 718 / rem_turns <= 2)
    deadweight_penalty = 0.0
    if step >= 718 or rem_turns <= 2:
        total_deadweight = deadweight_shed + deadweight_seeds
        deadweight_penalty = deadweight_penalty_coeff * total_deadweight

    return cash + crops_val + animals_val + seeds_val + shed_val - deadweight_penalty


def compute_pbrs_step_reward(
    s_t: Dict[str, Any],
    s_next: Dict[str, Any],
    raw_reward: float = 0.0,
    gamma: float = 0.995,
    player: int = 0,
    total_turns: int = 720,
    relative: bool = False,
    deadweight_penalty_coeff: float = 2.0
) -> float:
    """
    Computes shaped step reward: r'_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t).
    """
    phi_t = compute_state_potential(
        obs=s_t,
        player=player,
        total_turns=total_turns,
        relative=relative,
        deadweight_penalty_coeff=deadweight_penalty_coeff
    )
    phi_next = compute_state_potential(
        obs=s_next,
        player=player,
        total_turns=total_turns,
        relative=relative,
        deadweight_penalty_coeff=deadweight_penalty_coeff
    )
    return float(raw_reward + gamma * phi_next - phi_t)


class MaturityAwarePBRS:
    """
    Maturity-Aware Potential-Based Reward Shaping (PBRS) with biological asset discounting schedules.
    Guarantees mathematical reward telescoping and enforces 2.0x deadweight penalty at Turn 718.
    """
    def __init__(
        self,
        gamma: float = 0.995,
        total_turns: int = 720,
        relative: bool = False,
        deadweight_penalty_coeff: float = 2.0
    ):
        self.gamma = gamma
        self.total_turns = total_turns
        self.relative = relative
        self.deadweight_penalty_coeff = deadweight_penalty_coeff

    def compute_potential(self, obs: Dict[str, Any], player: int = 0) -> float:
        return compute_state_potential(
            obs=obs,
            player=player,
            total_turns=self.total_turns,
            relative=self.relative,
            deadweight_penalty_coeff=self.deadweight_penalty_coeff
        )

    def compute_step_reward(
        self,
        s_t: Dict[str, Any],
        s_next: Dict[str, Any],
        raw_reward: float = 0.0,
        player: int = 0
    ) -> float:
        phi_t = self.compute_potential(s_t, player=player)
        phi_next = self.compute_potential(s_next, player=player)
        return float(raw_reward + self.gamma * phi_next - phi_t)

    def shape_trajectory(
        self,
        observations: List[Dict[str, Any]],
        raw_rewards: Optional[np.ndarray] = None,
        player: int = 0
    ) -> np.ndarray:
        T = len(observations) - 1
        if T <= 0:
            return np.zeros(0, dtype=np.float32)

        if raw_rewards is None:
            raw_rewards = np.zeros(T, dtype=np.float32)

        shaped = np.zeros(T, dtype=np.float32)
        potentials = [self.compute_potential(obs, player=player) for obs in observations]

        for t in range(T):
            shaped[t] = raw_rewards[t] + self.gamma * potentials[t + 1] - potentials[t]

        return shaped


class TrajectoryGAEProcessor:
    """
    Computes Generalized Advantage Estimation (GAE) and target values along match trajectories.
    Parameters: gamma = 0.995, lambda = 0.95.
    Generates two-hot symlog categorical distribution targets over [-15.0, +15.0].
    """
    def __init__(
        self,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        num_bins: int = NUM_BINS,
        v_min: float = V_MIN,
        v_max: float = V_MAX
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max

    def process_trajectory(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        bootstrap_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes:
        - rewards: Shape (T,)
        - values: Shape (T+1,) or Shape (T,)
        Returns:
        - shaped_rewards: Shape (T,)
        - advantages: Shape (T,)
        - value_targets: Shape (T,)
        """
        T = len(rewards)
        if len(values) == T:
            vals = np.empty(T + 1, dtype=np.float32)
            vals[:T] = values
            vals[T] = bootstrap_value
        else:
            vals = np.asarray(values, dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * vals[t + 1] - vals[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        value_targets = advantages + vals[:T]
        return np.asarray(rewards, dtype=np.float32), advantages, value_targets

    def compute_two_hot_targets(self, value_targets: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Converts scalar dollar cash targets into two-hot categorical distributions over symlog bins.
        """
        if isinstance(value_targets, np.ndarray):
            v_tensor = torch.from_numpy(value_targets).float()
        elif isinstance(value_targets, torch.Tensor):
            v_tensor = value_targets.float()
        else:
            v_tensor = torch.tensor(value_targets, dtype=torch.float32)

        if v_tensor.ndim == 0:
            v_tensor = v_tensor.unsqueeze(0)

        return value_to_two_hot(
            v_tensor,
            num_bins=self.num_bins,
            v_min=self.v_min,
            v_max=self.v_max
        )

    def process_match(
        self,
        observations: List[Dict[str, Any]],
        value_estimates: Union[np.ndarray, torch.Tensor],
        raw_rewards: Optional[np.ndarray] = None,
        pbrs: Optional[MaturityAwarePBRS] = None,
        player: int = 0
    ) -> Dict[str, Any]:
        """
        Processes full or truncated match observation trajectory end-to-end:
        1. Computes PBRS potential and shaped rewards.
        2. Computes GAE advantages and value targets.
        3. Computes two-hot symlog categorical targets.
        """
        if pbrs is None:
            pbrs = MaturityAwarePBRS(gamma=self.gamma)

        potentials = np.array(
            [pbrs.compute_potential(obs, player=player) for obs in observations],
            dtype=np.float32
        )

        shaped_rewards = pbrs.shape_trajectory(
            observations=observations,
            raw_rewards=raw_rewards,
            player=player
        )

        if isinstance(value_estimates, torch.Tensor):
            vals = value_estimates.detach().cpu().numpy()
        else:
            vals = np.asarray(value_estimates, dtype=np.float32)

        shaped_r, advantages, value_targets = self.process_trajectory(
            rewards=shaped_rewards,
            values=vals
        )

        two_hot_targets = self.compute_two_hot_targets(value_targets)

        return {
            "shaped_rewards": shaped_r,
            "advantages": advantages,
            "value_targets": value_targets,
            "two_hot_targets": two_hot_targets,
            "potentials": potentials,
        }

