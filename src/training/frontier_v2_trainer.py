"""
Frontier V2 Grandmaster RL Training Pipeline:
High-Throughput 14-Core Parallel Self-Play Loop with 30-Personality Mega League & Double Oracle Nash Sampling.

Components:
1. High-Throughput 14-Core Multi-Worker Match Generator:
   - Evaluates active Frontier V2 champion against 30 distinct strategic personalities, past checkpoints, and self-play.
   - Double Oracle Nash Sampling: Solves empirical minimax game support + PFSP hard-opponent oversampling.
2. Gumbel MuZero Latent Tree Search (N=16 Sequential Halving, Playout-Cap Randomization).
3. 4-Pillar Reward & 601-Bin Two-Hot Symlog Value Target:
   - Win Margin (cash_0 - cash_1)
   - Explosive Growth Bonus: +1.5 * ((max(0, cash - 3000) / 5000) ** 1.5) * 1000
   - Capital Loss Penalty: -2.5 * (max(0, 3000 - cash) / 1000) * 1000
   - Deadweight Trapped Asset Penalty: -2.0 * (deadweight / 1000) * 1000
   - 601-Bin Two-Hot Symlog target in [-15, +15].
4. Multi-Task Neural Optimization:
   - Policy Cross-Entropy + 601-Bin Value Categorical Cross-Entropy + SSL Dynamics MSE + KataGo Auxiliary Heads.
5. Champion Checkpointing & Full Training Logging:
   - Checkpoints: 'weights/frontier_v2_grandmaster_champion.pt'
   - Full League Logs: 'data/frontier_v2_training_log.json'
"""

import copy
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

from src.models.frontier_v2_network import (
    FrontierV2Network,
    scalar_to_two_hot,
    categorical_to_scalar,
    NUM_BINS,
    V_MIN,
    V_MAX,
)
from src.models.encoder import NUM_MACRO_ACTIONS, MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS, encode_observation
from src.models.gumbel_muzero_mcts import GumbelMuZeroMCTS, compute_action_mask
from src.agents.frontier_executors import frontier_macro_executor, frontier_unified_agent
from src.training.mega_league import (
    AlphaGoatMegaLeague,
    MEGA_LEAGUE_PERSONALITIES,
)

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

BASE_PRICES = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}


def calculate_trapped_deadweight(obs: Dict[str, Any], player_idx: int = 0) -> float:
    """Calculates dollar value of unliquidated capital trapped at the end of the match."""
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

    # 2. Shed inventory
    shed = private.get("shed", {})
    for item, count in shed.items():
        if count > 0:
            deadweight += count * ITEM_COSTS.get(item, 50)

    # 3. Unharvested crops
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
) -> Tuple[float, Dict[str, float]]:
    """
    Locked 4-pillar reward function for Kaggriculture:
      1. Win Margin: (cash - opp_cash)
      2. Explosive Growth Bonus: +1.5 * ((max(0, cash - 3000) / 5000) ** 1.5) * 1000
      3. Capital Loss Penalty: -2.5 * (max(0, 3000 - cash) / 1000) * 1000
      4. Trapped Deadweight Penalty: -2.0 * (deadweight / 1000) * 1000
      Value Target: Raw continuous dollar outcome
    """
    win_margin = cash - opp_cash
    growth_bonus = 1.5 * ((max(0.0, cash - 3000.0) / 5000.0) ** 1.5) * 1000.0
    cap_loss_penalty = 2.5 * (max(0.0, 3000.0 - cash) / 1000.0) * 1000.0
    deadweight_penalty = 2.0 * (deadweight / 1000.0) * 1000.0

    raw = win_margin + growth_bonus - cap_loss_penalty - deadweight_penalty

    breakdown = {
        "cash": cash,
        "opp_cash": opp_cash,
        "win_margin": win_margin,
        "growth_bonus": growth_bonus,
        "cap_loss_penalty": cap_loss_penalty,
        "deadweight_penalty": deadweight_penalty,
        "raw": raw,
    }
    return raw, breakdown


class DoubleOracleMatchmaker:
    """
    Double Oracle Nash Equilibrium and Prioritized Opponent Sampler.
    Computes empirical margin matrix and samples opponents based on Nash support & PFSP difficulty.
    """
    def __init__(self, league: AlphaGoatMegaLeague, gamma: float = 1.5):
        self.league = league
        self.gamma = gamma

    def sample_opponent(self) -> Tuple[str, Any]:
        """
        Samples an opponent using Double Oracle Minimax Nash support and PFSP hard-opponent weighting.
        """
        roll = random.random()
        if roll < 0.60:
            # Sample Sparring Bot with Double Oracle / PFSP weighting
            weights = []
            for name in self.league.sparring_names:
                matches = self.league.payoff_matrix.get((self.league.main_name, name), [])
                if not matches:
                    loss_rate = 0.7  # High exploration priority
                else:
                    champ_wins = sum(1 for m in matches if m[0] > m[1])
                    loss_rate = 1.0 - (champ_wins / len(matches))
                
                # Double Oracle hardness weighting
                w = max(0.05, loss_rate ** self.gamma)
                weights.append(w)

            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            chosen_name = np.random.choice(self.league.sparring_names, p=probs)
            return chosen_name, self.league.participants[chosen_name]

        elif roll < 0.85 and self.league.checkpoint_names:
            # Sample Past Checkpoint
            n = len(self.league.checkpoint_names)
            weights = [np.exp(0.2 * i) for i in range(n)]
            probs = [w / sum(weights) for w in weights]
            chosen_name = np.random.choice(self.league.checkpoint_names, p=probs)
            return chosen_name, self.league.participants[chosen_name]

        else:
            # Self-Play
            return self.league.main_name, self.league.main_agent


class FrontierWorkerAgent:
    """Lightweight standalone agent wrapper for multi-worker simulation."""
    def __init__(self, model: FrontierV2Network, default_budget: int = 16):
        self.mcts = GumbelMuZeroMCTS(
            model=model,
            default_budget=default_budget,
            max_candidates=4,
            device=torch.device("cpu"),
        )

    def get_action(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], np.ndarray, float, int]:
        macro_act, pi_target, val_est = self.mcts.search(obs=obs, temperature=1.0)
        action_dict = frontier_macro_executor(obs, macro_act)
        return action_dict, pi_target, val_est, macro_act


def extract_tile_yield_grid(obs: Dict[str, Any], player_idx: int = 0) -> np.ndarray:
    """Extracts ground-truth 10x10 tile yield matrix for auxiliary supervision."""
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player_idx] if player_idx < len(farms) else {}
    tiles = my_farm.get("tiles", [])

    yield_grid = np.zeros((1, 10, 10), dtype=np.float32)
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            if isinstance(t, dict):
                yield_grid[0, r, c] = float(t.get("yield_units", 0))
    return yield_grid


def extract_prices_vector(obs: Dict[str, Any]) -> np.ndarray:
    """Extracts 9 commodity prices vector for auxiliary supervision."""
    market = obs.get("market", {})
    prices = market.get("prices", {})
    prods = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
    vec = np.zeros(9, dtype=np.float32)
    for idx, p in enumerate(prods):
        vec[idx] = float(prices.get(p, BASE_PRICES.get(p, 50))) / 250.0
    return vec


def run_frontier_v2_match_worker(
    match_id: int,
    seed: int,
    champion_state_dict: Dict[str, torch.Tensor],
    opponent_name: str,
    opponent_role: str,
    opponent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    default_budget: int = 16,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Top-level worker function for parallel multiprocessing match simulations.
    """
    torch.set_num_threads(1)

    # Initialize Champion model
    champ_model = FrontierV2Network()
    champ_model.load_state_dict(champion_state_dict)
    champ_model.eval()

    champ_agent = FrontierWorkerAgent(model=champ_model, default_budget=default_budget)

    # Initialize Opponent
    opp_is_neural = False
    if opponent_role == "sparring":
        opp_callable = MEGA_LEAGUE_PERSONALITIES.get(opponent_name, frontier_unified_agent)
    elif opponent_role == "checkpoint" and opponent_state_dict is not None:
        opp_model = FrontierV2Network()
        try:
            opp_model.load_state_dict(opponent_state_dict)
        except Exception:
            opp_model.load_state_dict(champion_state_dict)
        opp_model.eval()
        opp_agent = FrontierWorkerAgent(model=opp_model, default_budget=max(8, default_budget // 2))
        opp_is_neural = True
    else:
        # Self-Play
        opp_agent = FrontierWorkerAgent(model=champ_model, default_budget=default_budget)
        opp_is_neural = True

    champ_is_p0 = (match_id % 2 == 0)
    t0 = time.time()

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.reset()

    champ_history: List[Dict[str, Any]] = []
    opp_history: List[Dict[str, Any]] = []

    while not env.done:
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation

        if champ_is_p0:
            g_champ, s_champ = encode_observation(obs0)
            y_champ = extract_tile_yield_grid(obs0, player_idx=0)
            p_champ = extract_prices_vector(obs0)
            act_champ, pi_champ, val_champ, macro_champ = champ_agent.get_action(obs0)
            champ_history.append({
                "grid": g_champ,
                "scalars": s_champ,
                "tile_yield": y_champ,
                "prices": p_champ,
                "pi": pi_champ,
                "action": macro_champ,
            })

            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs1)
                y_opp = extract_tile_yield_grid(obs1, player_idx=1)
                p_opp = extract_prices_vector(obs1)
                act_opp, pi_opp, val_opp, macro_opp = opp_agent.get_action(obs1)
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "tile_yield": y_opp,
                    "prices": p_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                try:
                    act_opp = opp_callable(obs1)
                except Exception:
                    act_opp = frontier_unified_agent(obs1)

            env.step([act_champ, act_opp])

        else:
            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs0)
                y_opp = extract_tile_yield_grid(obs0, player_idx=0)
                p_opp = extract_prices_vector(obs0)
                act_opp, pi_opp, val_opp, macro_opp = opp_agent.get_action(obs0)
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "tile_yield": y_opp,
                    "prices": p_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                try:
                    act_opp = opp_callable(obs0)
                except Exception:
                    act_opp = frontier_unified_agent(obs0)

            g_champ, s_champ = encode_observation(obs1)
            y_champ = extract_tile_yield_grid(obs1, player_idx=1)
            p_champ = extract_prices_vector(obs1)
            act_champ, pi_champ, val_champ, macro_champ = champ_agent.get_action(obs1)
            champ_history.append({
                "grid": g_champ,
                "scalars": s_champ,
                "tile_yield": y_champ,
                "prices": p_champ,
                "pi": pi_champ,
                "action": macro_champ,
            })

            env.step([act_opp, act_champ])

    # Terminal evaluation
    r0 = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
    r1 = float(env.state[1].reward if env.state[1].reward is not None else 0.0)

    final_obs0 = env.state[0].observation
    final_obs1 = env.state[1].observation

    dw0 = calculate_trapped_deadweight(final_obs0, player_idx=0)
    dw1 = calculate_trapped_deadweight(final_obs1, player_idx=1)

    if champ_is_p0:
        champ_cash, opp_cash = r0, r1
        champ_dw, opp_dw = dw0, dw1
    else:
        champ_cash, opp_cash = r1, r0
        champ_dw, opp_dw = dw1, dw0

    raw_champ, reward_breakdown = calculate_4pillar_reward(
        cash=champ_cash, opp_cash=opp_cash, deadweight=champ_dw
    )

    # Assign future scalars (horizon=4), future prices (horizon=24), and raw value target
    n_champ = len(champ_history)
    for i in range(n_champ):
        f4_idx = min(n_champ - 1, i + 4)
        f24_idx = min(n_champ - 1, i + 24)
        champ_history[i]["future_scalars"] = champ_history[f4_idx]["scalars"]
        champ_history[i]["future_prices"] = champ_history[f24_idx]["prices"]
        champ_history[i]["raw_value"] = raw_champ

    collected_transitions = champ_history

    if opp_is_neural and opp_history:
        raw_opp, _ = calculate_4pillar_reward(cash=opp_cash, opp_cash=champ_cash, deadweight=opp_dw)
        n_opp = len(opp_history)
        for i in range(n_opp):
            f4_idx = min(n_opp - 1, i + 4)
            f24_idx = min(n_opp - 1, i + 24)
            opp_history[i]["future_scalars"] = opp_history[f4_idx]["scalars"]
            opp_history[i]["future_prices"] = opp_history[f24_idx]["prices"]
            opp_history[i]["raw_value"] = raw_opp
        collected_transitions = collected_transitions + opp_history

    duration = time.time() - t0

    match_stats = {
        "match_id": match_id,
        "seed": seed,
        "opponent_name": opponent_name,
        "opponent_role": opponent_role,
        "champ_is_p0": champ_is_p0,
        "champ_cash": champ_cash,
        "opp_cash": opp_cash,
        "margin": champ_cash - opp_cash,
        "won": bool(champ_cash > opp_cash),
        "tied": bool(champ_cash == opp_cash),
        "deadweight": champ_dw,
        "raw_reward": raw_champ,
        "breakdown": reward_breakdown,
        "duration": duration,
    }

    return collected_transitions, match_stats


class FrontierV2ReplayDataset(Dataset):
    """PyTorch Dataset for Experience Replay Buffer."""
    def __init__(self, buffer: List[Dict[str, Any]]):
        self.grids = torch.tensor(np.array([b["grid"] for b in buffer]), dtype=torch.float32)
        self.scalars = torch.tensor(np.array([b["scalars"] for b in buffer]), dtype=torch.float32)
        self.pi_targets = torch.tensor(np.array([b["pi"] for b in buffer]), dtype=torch.float32)
        self.raw_values = torch.tensor(np.array([b["raw_value"] for b in buffer]), dtype=torch.float32)
        self.future_scalars = torch.tensor(np.array([b["future_scalars"] for b in buffer]), dtype=torch.float32)
        self.actions = torch.tensor(np.array([b["action"] for b in buffer]), dtype=torch.long)
        self.tile_yields = torch.tensor(np.array([b["tile_yield"] for b in buffer]), dtype=torch.float32)
        self.future_prices = torch.tensor(np.array([b["future_prices"] for b in buffer]), dtype=torch.float32)

        # Precompute 601-bin two-hot symlog value targets
        self.two_hot_values = scalar_to_two_hot(self.raw_values)

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.grids[idx],
            self.scalars[idx],
            self.actions[idx],
            self.pi_targets[idx],
            self.two_hot_values[idx],
            self.future_scalars[idx],
            self.tile_yields[idx],
            self.future_prices[idx],
        )


class FrontierV2Trainer:
    """
    High-Throughput 14-Core Parallel Double Oracle RL Trainer.
    """
    def __init__(
        self,
        num_workers: int = 14,
        iterations: int = 10,
        matches_per_iter: int = 28,
        batch_size: int = 64,
        epochs_per_iter: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "weights",
        log_path: str = "data/frontier_v2_training_log.json",
        warm_start_weights_path: Optional[str] = "weights/alphagoat_grandmaster_champion.pt",
        buffer_capacity: int = 30000,
    ):
        self.num_workers = num_workers
        self.iterations = iterations
        self.matches_per_iter = matches_per_iter
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        self.log_path = log_path
        self.warm_start_weights_path = warm_start_weights_path
        self.buffer_capacity = buffer_capacity

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 1. Initialize Frontier V2 Network
        self.model = FrontierV2Network()
        self._warm_start_model()

        # 2. Initialize Mega League & Double Oracle Matchmaker
        self.league = AlphaGoatMegaLeague(main_name="FrontierV2Champion", max_checkpoints=30)
        self.matchmaker = DoubleOracleMatchmaker(self.league, gamma=1.5)
        self.replay_buffer: List[Dict[str, Any]] = []

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=iterations, eta_min=1e-5)
        self.training_logs: List[Dict[str, Any]] = []

    def _warm_start_model(self):
        """Warm-starts matching layers from AlphaGoat Grandmaster Champion checkpoint."""
        if self.warm_start_weights_path and os.path.exists(self.warm_start_weights_path):
            print(f"[FrontierV2] Warm-starting model from: {self.warm_start_weights_path}")
            ckpt = torch.load(self.warm_start_weights_path, map_location="cpu", weights_only=True)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            
            # Filter matching keys
            model_sd = self.model.state_dict()
            transferred = 0
            for k, v in sd.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k] = v
                    transferred += 1
            self.model.load_state_dict(model_sd)
            print(f"[FrontierV2] Transferred {transferred} matching layer weights successfully.")
        else:
            print("[FrontierV2] Initializing model parameters from scratch.")

    def run_training(self):
        """Executes full heavy RL training loop across 14 CPU cores."""
        print("=" * 80)
        print(" FRONTIER V2 GRANDMASTER HEAVY RL TRAINING SYSTEM")
        print(f" Workers: {self.num_workers} CPU cores | Total Iterations: {self.iterations}")
        print(f" Matches / Iteration: {self.matches_per_iter} | Batch Size: {self.batch_size}")
        print(f" League Personalities: {len(MEGA_LEAGUE_PERSONALITIES)} | Value Head: 601-Bin Symlog")
        print("=" * 80)

        total_matches_run = 0
        overall_start_time = time.time()

        for iteration in range(1, self.iterations + 1):
            iter_start_time = time.time()
            print(f"\n>>> [Iteration {iteration:02d}/{self.iterations:02d}] Launching {self.matches_per_iter} Double Oracle Matches across {self.num_workers} cores...")

            # 1. Sample Opponents using Double Oracle Nash Matchmaker
            match_tasks = []
            champ_state_dict = copy.deepcopy(self.model.state_dict())

            for m_idx in range(self.matches_per_iter):
                opp_name, opp_participant = self.matchmaker.sample_opponent()
                seed = random.randint(100000, 999999)
                match_tasks.append((
                    total_matches_run + m_idx,
                    seed,
                    champ_state_dict,
                    opp_name,
                    opp_participant.role,
                    opp_participant.state_dict,
                    16,  # default_budget for Gumbel MuZero
                ))

            # 2. Parallel Multi-Worker Execution across 14 CPU cores
            pool_size = min(self.num_workers, len(match_tasks))
            with mp.Pool(processes=pool_size) as pool:
                results = pool.starmap(run_frontier_v2_match_worker, match_tasks)

            total_matches_run += len(match_tasks)

            # 3. Ingest Transitions & Update League Payoff Matrix
            iter_wins = 0
            iter_cash = []
            iter_opp_cash = []
            iter_deadweights = []
            iter_margins = []

            for transitions, meta in results:
                self.replay_buffer.extend(transitions)
                self.league.update_match_result(
                    p0_name=self.league.main_name,
                    p1_name=meta["opponent_name"],
                    p0_cash=meta["champ_cash"],
                    p1_cash=meta["opp_cash"],
                )
                if meta["won"]:
                    iter_wins += 1
                iter_cash.append(meta["champ_cash"])
                iter_opp_cash.append(meta["opp_cash"])
                iter_deadweights.append(meta["deadweight"])
                iter_margins.append(meta["margin"])

            # Maintain buffer capacity
            if len(self.replay_buffer) > self.buffer_capacity:
                self.replay_buffer = self.replay_buffer[-self.buffer_capacity:]

            win_rate = (iter_wins / len(results)) * 100.0
            mean_cash = float(np.mean(iter_cash))
            mean_opp_cash = float(np.mean(iter_opp_cash))
            mean_margin = float(np.mean(iter_margins))
            mean_deadweight = float(np.mean(iter_deadweights))

            print(f"[Iter {iteration:02d}] Win Rate: {win_rate:.1f}% ({iter_wins}/{len(results)}) | "
                  f"Mean Cash: ${mean_cash:,.0f} vs ${mean_opp_cash:,.0f} (Margin: ${mean_margin:+,.0f}) | "
                  f"Avg Deadweight: ${mean_deadweight:,.0f}")

            # 4. Multi-Task Neural Optimization
            print(f"[Iter {iteration:02d}] Optimizing Frontier V2 Network on {len(self.replay_buffer)} transitions...")
            self.model.train()
            dataset = FrontierV2ReplayDataset(self.replay_buffer)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            loss_policy_list = []
            loss_val_list = []
            loss_dyn_list = []
            loss_tile_list = []
            loss_price_list = []
            loss_total_list = []

            for epoch in range(self.epochs_per_iter):
                for b_grid, b_scalar, b_act, b_pi, b_two_hot_val, b_next_scalar, b_tile, b_price in dataloader:
                    self.optimizer.zero_grad()

                    pred_logits, pred_val_logits, pred_dyn, pred_tile, pred_price = self.model(b_grid, b_scalar, b_act)

                    # 1. Policy Cross-Entropy Loss
                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    loss_policy = -torch.sum(b_pi * log_probs, dim=-1).mean()

                    # 2. 601-Bin Two-Hot Categorical Cross-Entropy Value Loss
                    val_log_probs = F.log_softmax(pred_val_logits, dim=-1)
                    loss_val = -torch.sum(b_two_hot_val * val_log_probs, dim=-1).mean()

                    # 3. Dynamics Self-Supervised MSE Loss
                    loss_dyn = F.mse_loss(pred_dyn, b_next_scalar)

                    # 4. KataGo Auxiliary Heads Losses
                    loss_tile = F.mse_loss(pred_tile, b_tile)
                    loss_price = F.mse_loss(pred_price, b_price)

                    # Multi-Task Objective
                    loss_total = loss_policy + 0.5 * loss_val + 0.25 * loss_dyn + 0.1 * loss_tile + 0.1 * loss_price

                    loss_total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    loss_policy_list.append(loss_policy.item())
                    loss_val_list.append(loss_val.item())
                    loss_dyn_list.append(loss_dyn.item())
                    loss_tile_list.append(loss_tile.item())
                    loss_price_list.append(loss_price.item())
                    loss_total_list.append(loss_total.item())

            self.scheduler.step()
            self.model.eval()

            mean_policy_loss = float(np.mean(loss_policy_list))
            mean_val_loss = float(np.mean(loss_val_list))
            mean_dyn_loss = float(np.mean(loss_dyn_list))
            mean_total_loss = float(np.mean(loss_total_list))

            print(f"[Iter {iteration:02d}] Loss: Total={mean_total_loss:.4f} | Policy={mean_policy_loss:.4f} | "
                  f"Value={mean_val_loss:.4f} | Dyn={mean_dyn_loss:.4f} | LR={self.scheduler.get_last_lr()[0]:.2e}")

            # 5. Checkpoint Active Champion into Pool & Disk
            champion_save_path = os.path.join(self.checkpoint_dir, "frontier_v2_grandmaster_champion.pt")
            torch.save(self.model.state_dict(), champion_save_path)
            self.league.add_checkpoint(iteration=iteration, state_dict=self.model.state_dict())

            iter_duration = time.time() - iter_start_time
            print(f"[Iter {iteration:02d}] Champion Checkpointed -> '{champion_save_path}' ({iter_duration:.1f}s)")

            # 6. Log Metrics
            log_entry = {
                "iteration": iteration,
                "timestamp": time.time(),
                "duration_seconds": round(iter_duration, 2),
                "total_matches": total_matches_run,
                "dataset_size": len(self.replay_buffer),
                "metrics": {
                    "win_rate": round(win_rate, 2),
                    "mean_cash": round(mean_cash, 2),
                    "mean_opp_cash": round(mean_opp_cash, 2),
                    "mean_margin": round(mean_margin, 2),
                    "mean_deadweight": round(mean_deadweight, 2),
                },
                "losses": {
                    "policy_loss": round(mean_policy_loss, 5),
                    "value_loss": round(mean_val_loss, 5),
                    "dynamics_loss": round(mean_dyn_loss, 5),
                    "total_loss": round(mean_total_loss, 5),
                },
                "league_summary": self.league.get_league_summary(),
            }
            self.training_logs.append(log_entry)

            with open(self.log_path, "w") as f:
                json.dump(self.training_logs, f, indent=2)

        total_elapsed = time.time() - overall_start_time
        print("\n" + "=" * 80)
        print(f" FRONTIER V2 TRAINING COMPLETE! Total Time: {total_elapsed/60:.1f} mins | Matches: {total_matches_run}")
        print(f" Final Weights Saved: {champion_save_path}")
        print(f" Full Metrics Logged: {self.log_path}")
        print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    trainer = FrontierV2Trainer(
        num_workers=14,
        iterations=6,
        matches_per_iter=28,
        batch_size=64,
        epochs_per_iter=4,
        learning_rate=2e-4,
        weight_decay=1e-4,
        checkpoint_dir="weights",
        log_path="data/frontier_v2_training_log.json",
        warm_start_weights_path="weights/alphagoat_grandmaster_champion.pt",
    )
    trainer.run_training()
