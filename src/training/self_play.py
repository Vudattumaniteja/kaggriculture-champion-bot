"""
Flexible AlphaZero Reinforcement Learning (RL) Self-Play Engine for Kaggriculture.
Uses pure reward shaping & heavy deadweight penalties (penalizing trapped seeds, unsold goods, and idle animals)
giving the neural network full strategic freedom while autonomously learning optimal liquidation timing.
"""

import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from kaggle_environments import make

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.ssl_network import SSLPolicyValueNet
from src.models.encoder import NUM_MACRO_ACTIONS, MACRO_ACTIONS, encode_observation
from src.models.mcts import MCTSNode
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

# Cost table for calculating deadweight trapped assets at turn 719
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


def calculate_trapped_deadweight(obs: Dict[str, Any], player_idx: int) -> float:
    """
    Calculates the dollar value of non-liquidated capital trapped at the end of the match.
    Gives the RL model learning signals to autonomously prune deadweight.
    """
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_idx] if player_idx < len(farms) else {}
    private = obs.get("private", {}) or {}

    deadweight = 0.0

    # 1. Unused seeds in pocket
    seeds = private.get("seeds", {})
    for crop, count in seeds.items():
        if count > 0:
            seed_key = f"{crop}_SEED"
            deadweight += count * ITEM_COSTS.get(seed_key, 20)

    # 2. Animals or goods left sitting in the shed unsold
    shed = private.get("shed", {})
    for item, count in shed.items():
        if count > 0:
            deadweight += count * ITEM_COSTS.get(item, 50)

    # 3. Unharvested crops left in the ground
    tiles = my_farm.get("tiles", [])
    for row in tiles:
        for t in row:
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    deadweight += ITEM_COSTS.get(crop, 35)

    return deadweight


def calculate_4pillar_reward(
    cash: float, opp_cash: float, deadweight: float
) -> Tuple[float, float, Dict[str, float]]:
    """
    Computes the locked 4-pillar reward function for Kaggriculture AlphaZero RL:
      1. Win Margin: (cash - opp_cash)
      2. Explosive Growth Bonus: +1.5 * ((max(0, cash - 3000) / 5000) ** 1.5) * 1000
      3. Heavy Capital Loss Penalty: -2.5 * (max(0, 3000 - cash) / 1000) * 1000
      4. Heavy Deadweight Trapped Asset Penalty: -2.0 * (deadweight / 1000) * 1000
      Value Target: z = tanh(Raw / 4000.0)
    """
    win_margin = cash - opp_cash
    growth_bonus = 1.5 * ((max(0.0, cash - 3000.0) / 5000.0) ** 1.5) * 1000.0
    cap_loss_penalty = 2.5 * (max(0.0, 3000.0 - cash) / 1000.0) * 1000.0
    deadweight_penalty = 2.0 * (deadweight / 1000.0) * 1000.0

    raw = win_margin + growth_bonus - cap_loss_penalty - deadweight_penalty
    z = float(np.tanh(raw / 4000.0))

    breakdown = {
        "cash": cash,
        "opp_cash": opp_cash,
        "win_margin": win_margin,
        "growth_bonus": growth_bonus,
        "cap_loss_penalty": cap_loss_penalty,
        "deadweight_penalty": deadweight_penalty,
        "raw": raw,
        "z": z,
    }
    return raw, z, breakdown


class FlexibleRLAgent:
    """MCTS Agent with Dirichlet exploration for RL self-play."""

    def __init__(
        self,
        model: SSLPolicyValueNet,
        num_simulations: int = 20,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.device = device

        self.executors = {
            0: crop_farmer_carrot,
            1: crop_farmer_wheat,
            2: crop_farmer_portfolio,
            3: hybrid_expert_agent,
            4: hybrid_expert_agent,
            5: hybrid_expert_agent,
            6: livestock_agent,
            7: None,  # Special liquidation
            8: arbitrage_agent,
            9: livestock_agent,
        }

    def search_action_distribution(
        self, obs: Dict[str, Any], add_exploration_noise: bool = True
    ) -> Tuple[int, np.ndarray, float]:
        """Runs MCTS and returns (selected_macro_action, policy_target_vector, value_estimate)."""
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_probs, value_t = self.model.predict(grid_t, scalars_t)

        probs = action_probs.squeeze(0).cpu().numpy()
        val_est = float(value_t.item())

        # Add Dirichlet exploration noise to root prior during self-play
        if add_exploration_noise and self.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * NUM_MACRO_ACTIONS)
            probs = (1 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise
            probs = probs / probs.sum()

        root = MCTSNode(prior=1.0)
        root.expand(probs)

        # MCTS rollouts
        for _ in range(self.num_simulations):
            node = root
            best_action = -1
            best_score = -float("inf")
            for a_idx, child in node.children.items():
                score = child.get_puct_score(parent_visits=node.visit_count, c_puct=self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action = a_idx

            if best_action != -1:
                selected_child = node.children[best_action]
                selected_child.update(val_est)
                root.update(val_est)

        # Compute search policy distribution from visit counts
        visit_counts = np.array([root.children[a].visit_count for a in range(NUM_MACRO_ACTIONS)], dtype=np.float32)
        total_visits = visit_counts.sum()

        if total_visits == 0:
            pi_target = probs
        else:
            pi_target = visit_counts / total_visits

        if self.temperature == 0.0 or total_visits == 0:
            selected_action = int(np.argmax(pi_target))
        else:
            exp_counts = visit_counts ** (1.0 / max(1e-3, self.temperature))
            dist = exp_counts / exp_counts.sum()
            selected_action = int(np.random.choice(NUM_MACRO_ACTIONS, p=dist))

        return selected_action, pi_target, val_est

    def get_action(self, obs: Dict[str, Any], add_noise: bool = True) -> Tuple[Dict[str, Any], np.ndarray, float]:
        selected_macro, pi_target, val_est = self.search_action_distribution(obs, add_exploration_noise=add_noise)

        # Translate macro-action into concrete turn execution
        if selected_macro == 7:
            # Flexible Harvest & Liquidation
            res = hybrid_expert_agent(obs)
            shed = obs.get("private", {}).get("shed", {})
            orders = []
            for item, count in shed.items():
                if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                    orders.append(["SELL", item, count])
            res["market"] = orders[:10]
            return res, pi_target, val_est

        executor = self.executors.get(selected_macro, hybrid_expert_agent)
        res = executor(obs)
        return res, pi_target, val_est


class SelfPlayDataset(Dataset):
    def __init__(self, buffer: List[Dict[str, Any]]):
        self.grids = torch.tensor(np.array([b["grid"] for b in buffer]), dtype=torch.float32)
        self.scalars = torch.tensor(np.array([b["scalars"] for b in buffer]), dtype=torch.float32)
        self.pi_targets = torch.tensor(np.array([b["pi"] for b in buffer]), dtype=torch.float32)
        self.values = torch.tensor(np.array([b["value"] for b in buffer]), dtype=torch.float32).unsqueeze(1)
        self.future_scalars = torch.tensor(np.array([b["future_scalars"] for b in buffer]), dtype=torch.float32)
        self.actions = torch.tensor(np.array([b["action"] for b in buffer]), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.grids[idx],
            self.scalars[idx],
            self.pi_targets[idx],
            self.values[idx],
            self.future_scalars[idx],
            self.actions[idx],
        )


def run_self_play_iteration(
    model: SSLPolicyValueNet,
    num_matches: int = 10,
    num_simulations: int = 20,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs self-play matches and collects experience with 4-pillar reward shaping."""
    agent_p0 = FlexibleRLAgent(
        model,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        temperature=1.0,
        device=device,
    )
    agent_p1 = FlexibleRLAgent(
        model,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        temperature=1.0,
        device=device,
    )

    experience_buffer: List[Dict[str, Any]] = []
    p0_banks: List[float] = []
    p1_banks: List[float] = []
    p0_deadweights: List[float] = []
    p1_deadweights: List[float] = []
    p0_z_values: List[float] = []
    p1_z_values: List[float] = []

    print(f"--- Running {num_matches} Self-Play Matches (MCTS sims={num_simulations}, Dir alpha={dirichlet_alpha}, eps={dirichlet_epsilon}) ---")
    t0 = time.time()

    for m_idx in range(1, num_matches + 1):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 7000 + m_idx})
        env.reset()

        p0_history: List[Dict[str, Any]] = []
        p1_history: List[Dict[str, Any]] = []

        while not env.done:
            obs0 = env.state[0].observation
            obs1 = env.state[1].observation

            g0, s0 = encode_observation(obs0)
            g1, s1 = encode_observation(obs1)

            act0, pi0, val0 = agent_p0.get_action(obs0, add_noise=True)
            act1, pi1, val1 = agent_p1.get_action(obs1, add_noise=True)

            macro0 = int(np.argmax(pi0))
            macro1 = int(np.argmax(pi1))

            p0_history.append({"grid": g0, "scalars": s0, "pi": pi0, "action": macro0})
            p1_history.append({"grid": g1, "scalars": s1, "pi": pi1, "action": macro1})

            env.step([act0, act1])

        # Terminal rewards
        raw_r0 = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
        raw_r1 = float(env.state[1].reward if env.state[1].reward is not None else 0.0)

        # Calculate trapped non-liquidated capital at turn 719
        final_obs0 = env.state[0].observation
        final_obs1 = env.state[1].observation

        dw0 = calculate_trapped_deadweight(final_obs0, player_idx=0)
        dw1 = calculate_trapped_deadweight(final_obs1, player_idx=1)

        # Compute locked 4-pillar reward for both players
        raw0, z0, b0 = calculate_4pillar_reward(cash=raw_r0, opp_cash=raw_r1, deadweight=dw0)
        raw1, z1, b1 = calculate_4pillar_reward(cash=raw_r1, opp_cash=raw_r0, deadweight=dw1)

        p0_banks.append(raw_r0)
        p1_banks.append(raw_r1)
        p0_deadweights.append(dw0)
        p1_deadweights.append(dw1)
        p0_z_values.append(z0)
        p1_z_values.append(z1)

        # Assign future scalars (horizon = 4) and outcome values
        n_steps = len(p0_history)
        for i in range(n_steps):
            f_idx = min(n_steps - 1, i + 4)
            p0_history[i]["future_scalars"] = p0_history[f_idx]["scalars"]
            p0_history[i]["value"] = z0

            p1_history[i]["future_scalars"] = p1_history[f_idx]["scalars"]
            p1_history[i]["value"] = z1

        experience_buffer.extend(p0_history)
        experience_buffer.extend(p1_history)

        print(
            f"  [Match {m_idx:2d}/{num_matches}] P0: ${raw_r0:,.0f} (DW: ${dw0:.0f}, z: {z0:+.3f}) | "
            f"P1: ${raw_r1:,.0f} (DW: ${dw1:.0f}, z: {z1:+.3f}) | Buffer Size: {len(experience_buffer):,}"
        )

    elapsed = time.time() - t0
    all_banks = p0_banks + p1_banks
    all_dw = p0_deadweights + p1_deadweights
    all_z = p0_z_values + p1_z_values

    stats = {
        "num_matches": num_matches,
        "total_transitions": len(experience_buffer),
        "mean_p0_bank": float(np.mean(p0_banks)),
        "mean_p1_bank": float(np.mean(p1_banks)),
        "mean_bank": float(np.mean(all_banks)),
        "max_bank": float(np.max(all_banks)),
        "min_bank": float(np.min(all_banks)),
        "mean_deadweight": float(np.mean(all_dw)),
        "mean_z_value": float(np.mean(all_z)),
        "duration_sec": elapsed,
    }
    print(
        f"Self-Play Complete in {elapsed:.1f}s | Mean Bank: ${stats['mean_bank']:,.0f} | "
        f"Mean Deadweight: ${stats['mean_deadweight']:,.0f} | Buffer: {stats['total_transitions']:,} steps"
    )
    return experience_buffer, stats


def train_rl_iteration(
    model: SSLPolicyValueNet,
    buffer: List[Dict[str, Any]],
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Trains the model on self-play transitions using cross-entropy on MCTS search policies."""
    dataset = SelfPlayDataset(buffer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    mse_fn = nn.MSELoss()

    total_loss_accum = 0.0
    pol_loss_accum = 0.0
    val_loss_accum = 0.0
    ssl_loss_accum = 0.0
    count = 0

    for ep in range(1, epochs + 1):
        for grids, scalars, pi_targets, values, future_scalars, actions in loader:
            grids = grids.to(device)
            scalars = scalars.to(device)
            pi_targets = pi_targets.to(device)
            values = values.to(device)
            future_scalars = future_scalars.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()

            logits, pred_val, pred_future = model(grids, scalars, action_indices=actions)

            # Policy Loss: Cross-Entropy with MCTS Soft Target Distribution
            log_probs = F.log_softmax(logits, dim=-1)
            loss_p = -(pi_targets * log_probs).sum(dim=-1).mean()

            # Value Loss: MSE with 4-pillar z target
            loss_v = mse_fn(pred_val, values)

            # SSL Dynamics Loss (Preserve World Model Dynamics)
            loss_ssl = mse_fn(pred_future, future_scalars)

            total_loss = loss_p + 1.0 * loss_v + 0.3 * loss_ssl

            total_loss.backward()
            optimizer.step()

            bs = grids.size(0)
            total_loss_accum += total_loss.item() * bs
            pol_loss_accum += loss_p.item() * bs
            val_loss_accum += loss_v.item() * bs
            ssl_loss_accum += loss_ssl.item() * bs
            count += bs

    return {
        "avg_loss": total_loss_accum / count,
        "pol_loss": pol_loss_accum / count,
        "val_loss": val_loss_accum / count,
        "ssl_loss": ssl_loss_accum / count,
    }


def run_rl_pipeline(
    initial_weights_path: str = "weights/grandmaster_ssl.pt",
    champion_weights_path: str = "weights/rl_alphazero_champion.pt",
    num_iterations: int = 3,
    matches_per_iteration: int = 10,
    num_simulations: int = 20,
    epochs_per_iter: int = 5,
    lr: float = 3e-4,
) -> List[Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f" Starting AlphaZero RL Self-Play Pipeline ({num_iterations} Iterations x {matches_per_iteration} Matches)")
    print(f" Device: {device} | Initial Base Weights: {initial_weights_path}")
    print(f" Champion Weights Destination: {champion_weights_path}")
    print(f" Reward Function: Locked 4-Pillar (Net Margin + Growth - Cap Loss - Deadweight)")
    print("=" * 80)

    model = SSLPolicyValueNet().to(device)
    if os.path.exists(initial_weights_path):
        model.load_state_dict(torch.load(initial_weights_path, map_location=device, weights_only=True))
        print(f"Successfully loaded initial weights from {initial_weights_path}")
    else:
        print(f"Warning: Initial weights not found at {initial_weights_path}, initializing from scratch.")

    history: List[Dict[str, Any]] = []

    for iter_idx in range(1, num_iterations + 1):
        print(f"\n{'='*30} RL ITERATION {iter_idx}/{num_iterations} {'='*30}")

        # 1. Generate self-play games with 4-pillar reward function
        buffer, sp_stats = run_self_play_iteration(
            model,
            num_matches=matches_per_iteration,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            device=device,
        )

        # 2. Train neural network on self-play experience
        train_metrics = train_rl_iteration(
            model,
            buffer,
            epochs=epochs_per_iter,
            batch_size=128,
            lr=lr,
            device=device,
        )
        print(
            f"Iteration {iter_idx} Training Metrics -> Loss: {train_metrics['avg_loss']:.4f} "
            f"(Policy Loss: {train_metrics['pol_loss']:.4f}, Value MSE: {train_metrics['val_loss']:.4f}, SSL MSE: {train_metrics['ssl_loss']:.5f})"
        )

        # 3. Save checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(champion_weights_path)), exist_ok=True)
        torch.save(model.state_dict(), champion_weights_path)
        print(f"Saved Champion Checkpoint for Iteration {iter_idx} to: {champion_weights_path}")

        iter_summary = {
            "iteration": iter_idx,
            "sp_stats": sp_stats,
            "train_metrics": train_metrics,
        }
        history.append(iter_summary)

    print("\n" + "=" * 80)
    print(" AlphaZero RL Self-Play Pipeline Completed Successfully!")
    print(f" Final Champion Model Checkpoint: {champion_weights_path}")
    print("=" * 80)
    return history


if __name__ == "__main__":
    run_rl_pipeline(num_iterations=3, matches_per_iteration=10, num_simulations=20)
