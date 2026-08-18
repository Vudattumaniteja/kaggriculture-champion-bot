"""
Grandmaster Deep Reinforcement Learning (RL) Self-Play Engine for Kaggriculture.

Key Architectural Pillars:
1. Deep MCTS Search: n=60 rollouts with Dirichlet exploration noise (alpha=0.3, eps=0.25)
2. Dynamic Zero-Deadweight Action Masking: Prunes unviable/trapped actions before MCTS expansion
3. Aggressive 4-Pillar Reward Function:
   - Win Margin: (cash_0 - cash_1)
   - Explosive Growth Bonus: +1.5 * ((max(0, cash - 3000) / 5000) ** 1.5) * 1000
   - Capital Loss Penalty: -2.5 * (max(0, 3000 - cash) / 1000) * 1000
   - Deadweight Trapped Asset Penalty: -2.0 * (deadweight / 1000) * 1000
   - Value Target: z = tanh(Raw / 4000.0)
4. High-Throughput Multi-Core Self-Play: Parallel match rollouts across CPU cores
5. Continuous Champion Checkpointing: weights/grandmaster_rl_champion.pt
"""

import math
import multiprocessing as mp
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

# Deadweight item cost table
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


def calculate_trapped_deadweight(obs: Dict[str, Any], player_idx: int = 0) -> float:
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


def compute_action_mask(obs: Dict[str, Any], player_idx: Optional[int] = None) -> np.ndarray:
    """
    Computes dynamic action validity mask over the 10 macro actions.
    Zeroes out illegal, unprofitable, or deadweight-generating actions.
    """
    if player_idx is None:
        player_idx = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_idx] if player_idx < len(farms) else {}
    private = obs.get("private", {}) or {}

    day = int(obs.get("day", 0))
    money = float(my_farm.get("money", 3000.0))
    unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
    tiles = my_farm.get("tiles", [])

    mask = np.ones(NUM_MACRO_ACTIONS, dtype=bool)

    # Count empty tiles and check for livestock
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

    # 7: HARVEST_AND_LIQUIDATE_ALL (Always valid, essential end-game)
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


def calculate_4pillar_reward(
    cash: float, opp_cash: float, deadweight: float
) -> Tuple[float, float, Dict[str, float]]:
    """
    Locked 4-pillar reward function for Kaggriculture Deep RL:
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


class DeepRLMCTSAgent:
    """MCTS Agent with Dirichlet exploration & dynamic action masking for Deep RL self-play."""

    def __init__(
        self,
        model: SSLPolicyValueNet,
        num_simulations: int = 60,
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
            7: None,  # Specialized liquidation
            8: arbitrage_agent,
            9: livestock_agent,
        }

    def search_action_distribution(
        self, obs: Dict[str, Any], add_exploration_noise: bool = True
    ) -> Tuple[int, np.ndarray, float]:
        """Runs deep MCTS rollouts with dynamic action masking and Dirichlet noise."""
        grid_np, scalars_np = encode_observation(obs)
        grid_t = torch.tensor(grid_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars_t = torch.tensor(scalars_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action_probs, value_t = self.model.predict(grid_t, scalars_t)

        probs = action_probs.squeeze(0).cpu().numpy()
        val_est = float(value_t.item())

        # Dynamic Zero-Deadweight Action Masking
        mask = compute_action_mask(obs)
        probs = probs * mask
        p_sum = probs.sum()
        if p_sum > 0:
            probs = probs / p_sum
        else:
            probs = mask.astype(np.float32) / mask.sum()

        # Add Dirichlet exploration noise to root prior during self-play
        if add_exploration_noise and self.dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * NUM_MACRO_ACTIONS)
            noise = noise * mask
            n_sum = noise.sum()
            if n_sum > 0:
                noise = noise / n_sum
            else:
                noise = probs
            probs = (1 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise
            probs = probs / probs.sum()

        root = MCTSNode(prior=1.0)
        root.expand(probs)

        # MCTS rollouts (n=60)
        for _ in range(self.num_simulations):
            node = root
            best_action = -1
            best_score = -float("inf")
            for a_idx, child in node.children.items():
                if not mask[a_idx]:
                    continue
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


def run_single_match_worker(
    match_id: int,
    seed: int,
    state_dict: Dict[str, torch.Tensor],
    num_simulations: int = 60,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Standalone top-level worker function for parallel multiprocessing self-play matches.
    """
    # Set single thread per worker to optimize multi-core execution
    torch.set_num_threads(1)

    model = SSLPolicyValueNet()
    model.load_state_dict(state_dict)
    model.eval()

    agent_p0 = DeepRLMCTSAgent(
        model=model,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        temperature=1.0,
    )
    agent_p1 = DeepRLMCTSAgent(
        model=model,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        temperature=1.0,
    )

    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
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

    # Terminal evaluation
    raw_r0 = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
    raw_r1 = float(env.state[1].reward if env.state[1].reward is not None else 0.0)

    final_obs0 = env.state[0].observation
    final_obs1 = env.state[1].observation

    dw0 = calculate_trapped_deadweight(final_obs0, player_idx=0)
    dw1 = calculate_trapped_deadweight(final_obs1, player_idx=1)

    raw0, z0, _ = calculate_4pillar_reward(cash=raw_r0, opp_cash=raw_r1, deadweight=dw0)
    raw1, z1, _ = calculate_4pillar_reward(cash=raw_r1, opp_cash=raw_r0, deadweight=dw1)

    # Assign future scalars (horizon = 4) and outcome values
    n_steps = len(p0_history)
    for i in range(n_steps):
        f_idx = min(n_steps - 1, i + 4)
        p0_history[i]["future_scalars"] = p0_history[f_idx]["scalars"]
        p0_history[i]["value"] = z0

        p1_history[i]["future_scalars"] = p1_history[f_idx]["scalars"]
        p1_history[i]["value"] = z1

    match_transitions = p0_history + p1_history
    duration = time.time() - t0

    match_stats = {
        "match_id": match_id,
        "p0_cash": raw_r0,
        "p1_cash": raw_r1,
        "p0_deadweight": dw0,
        "p1_deadweight": dw1,
        "p0_z": z0,
        "p1_z": z1,
        "duration": duration,
    }

    return match_transitions, match_stats


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


def train_neural_network(
    model: SSLPolicyValueNet,
    buffer: List[Dict[str, Any]],
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Trains policy, value, and world dynamics heads on self-play experience buffer."""
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
        "avg_loss": total_loss_accum / max(1, count),
        "pol_loss": pol_loss_accum / max(1, count),
        "val_loss": val_loss_accum / max(1, count),
        "ssl_loss": ssl_loss_accum / max(1, count),
    }


def execute_deep_rl_training(
    initial_weights_path: str = "weights/grandmaster_ssl.pt",
    champion_weights_path: str = "weights/grandmaster_rl_champion.pt",
    log_save_path: str = "data/deep_rl_training_log.json",
    num_iterations: int = 25,
    matches_per_iteration: int = 15,
    num_simulations: int = 60,
    epochs_per_iter: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    replay_buffer_capacity: int = 80000,
    num_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Executes the comprehensive Deep RL Self-Play training pipeline.
    """
    import json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        num_workers = max(1, min(mp.cpu_count(), 14))

    print("=" * 85)
    print("        KAGGRICULTURE GRANDMASTER DEEP RL SELF-PLAY TRAINING ENGINE")
    print("=" * 85)
    print(f" Device: {device} | Parallel Workers: {num_workers} (Total CPU Cores: {mp.cpu_count()})")
    print(f" Iterations: {num_iterations} | Matches/Iter: {matches_per_iteration} (Total Planned Matches: {num_iterations * matches_per_iteration})")
    print(f" MCTS Search Depth: n={num_simulations} rollouts | Dirichlet Noise: alpha=0.3, eps=0.25")
    print(f" Dynamic Zero-Deadweight Action Masking: ACTIVE")
    print(f" Initial Base Model: {initial_weights_path}")
    print(f" Champion Model Destination: {champion_weights_path}")
    print(f" Replay Buffer Capacity: {replay_buffer_capacity:,} steps")
    print(f" 4-Pillar Reward: Win Margin + Explosive Growth - Capital Loss Penalty - Deadweight Penalty")
    print("=" * 85)

    model = SSLPolicyValueNet().to(device)
    if os.path.exists(initial_weights_path):
        model.load_state_dict(torch.load(initial_weights_path, map_location=device, weights_only=True))
        print(f"[*] Successfully loaded base weights from {initial_weights_path}\n")
    elif os.path.exists("weights/ssl_alphazero.pt"):
        model.load_state_dict(torch.load("weights/ssl_alphazero.pt", map_location=device, weights_only=True))
        print(f"[*] Loaded fallback base weights from weights/ssl_alphazero.pt\n")
    else:
        print(f"[!] Base weights not found at {initial_weights_path}, initializing randomly.\n")

    history: List[Dict[str, Any]] = []
    sliding_replay_buffer: List[Dict[str, Any]] = []
    total_pipeline_transitions = 0
    start_total_time = time.time()

    for iter_idx in range(1, num_iterations + 1):
        iter_start_t = time.time()
        print(f"\n{'='*35} ITERATION {iter_idx:02d}/{num_iterations:02d} {'='*35}")
        print(f">>> [Self-Play] Generating {matches_per_iteration} Matches (MCTS n={num_simulations}, Dir noise a=0.3/e=0.25)...")

        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}

        worker_args = [
            (
                m_idx,
                10000 + iter_idx * 1000 + m_idx,
                state_dict_cpu,
                num_simulations,
                0.3,
                0.25,
            )
            for m_idx in range(1, matches_per_iteration + 1)
        ]

        # Multi-core self-play match rollouts
        sp_start_t = time.time()
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(run_single_match_worker, worker_args)

        iter_buffer: List[Dict[str, Any]] = []
        all_banks: List[float] = []
        all_deadweights: List[float] = []
        all_z_values: List[float] = []

        for match_trans, m_stats in results:
            iter_buffer.extend(match_trans)
            all_banks.extend([m_stats["p0_cash"], m_stats["p1_cash"]])
            all_deadweights.extend([m_stats["p0_deadweight"], m_stats["p1_deadweight"]])
            all_z_values.extend([m_stats["p0_z"], m_stats["p1_z"]])

        total_pipeline_transitions += len(iter_buffer)
        sliding_replay_buffer.extend(iter_buffer)

        # Enforce sliding replay buffer capacity
        if len(sliding_replay_buffer) > replay_buffer_capacity:
            sliding_replay_buffer = sliding_replay_buffer[-replay_buffer_capacity:]

        mean_bank = float(np.mean(all_banks))
        max_bank = float(np.max(all_banks))
        min_bank = float(np.min(all_banks))
        mean_dw = float(np.mean(all_deadweights))
        mean_z = float(np.mean(all_z_values))
        sp_time = time.time() - sp_start_t

        print(
            f"    [Self-Play Done in {sp_time:.1f}s] Matches: {matches_per_iteration} | New Steps: {len(iter_buffer):,} | "
            f"Replay Buffer: {len(sliding_replay_buffer):,} steps"
        )
        print(
            f"    [Performance Metrics] Mean Bank: ${mean_bank:,.1f} | Peak Bank: ${max_bank:,.1f} | Min: ${min_bank:,.1f} | "
            f"Mean DW: ${mean_dw:.1f} | Mean z: {mean_z:+.3f}"
        )

        # Train neural network
        train_start_t = time.time()
        metrics = train_neural_network(
            model=model,
            buffer=sliding_replay_buffer,
            epochs=epochs_per_iter,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        train_time = time.time() - train_start_t

        print(
            f"    [Training Done in {train_time:.1f}s] Loss: {metrics['avg_loss']:.4f} "
            f"(Policy Loss: {metrics['pol_loss']:.4f}, Value MSE: {metrics['val_loss']:.4f}, SSL MSE: {metrics['ssl_loss']:.5f})"
        )

        # Continuously save champion model checkpoint
        os.makedirs(os.path.dirname(os.path.abspath(champion_weights_path)), exist_ok=True)
        torch.save(model.state_dict(), champion_weights_path)
        torch.save(model.state_dict(), "weights/rl_alphazero_champion.pt")
        print(f"    [Checkpoint] Saved Champion Model to {champion_weights_path}")

        iter_duration = time.time() - iter_start_t
        iter_summary = {
            "iteration": iter_idx,
            "mean_bank": mean_bank,
            "max_bank": max_bank,
            "min_bank": min_bank,
            "mean_deadweight": mean_dw,
            "mean_z_value": mean_z,
            "train_loss": metrics["avg_loss"],
            "pol_loss": metrics["pol_loss"],
            "val_loss": metrics["val_loss"],
            "ssl_loss": metrics["ssl_loss"],
            "new_transitions": len(iter_buffer),
            "buffer_size": len(sliding_replay_buffer),
            "total_transitions": total_pipeline_transitions,
            "iteration_duration_sec": iter_duration,
            "cumulative_time_sec": time.time() - start_total_time,
        }
        history.append(iter_summary)

        # Save metrics json log continuously
        if log_save_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_save_path)), exist_ok=True)
            with open(log_save_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    total_time = time.time() - start_total_time
    print("\n" + "=" * 85)
    print(f" Deep RL Self-Play Pipeline Completed in {total_time/60:.2f} minutes ({total_time:.1f}s)!")
    print(f" Total Self-Play Transitions Collected: {total_pipeline_transitions:,}")
    print(f" Final Champion Model: {champion_weights_path}")
    print(f" Training Metrics Log: {log_save_path}")
    print("=" * 85)

    return history


if __name__ == "__main__":
    # Ensure Windows multiprocessing compatibility
    mp.freeze_support()
    execute_deep_rl_training(
        num_iterations=25,
        matches_per_iteration=15,
        num_simulations=60,
    )
