"""
AlphaGoat Heavy RL Training Pipeline: 30-Personality Mega League + MuZero Latent Tree Search.

Orchestration Architecture:
1. High-Throughput 14-Core Multi-Worker Match Generator:
   - Evaluates active AlphaGoat champion against 30 distinct strategic personalities, past checkpoints, and self-play.
   - Prioritized Fictitious Self-Play (PFSP) with dynamic hard-opponent oversampling.
2. MuZero Latent Tree Search (SSL World Dynamics imagination):
   - Vectorized imagined rollouts in 128-dim latent space.
   - Dirichlet noise exploration (alpha=0.3, eps=0.25) + Dynamic Zero-Deadweight Action Masking.
3. 4-Pillar Reward Optimization:
   - Win Margin (cash_0 - cash_1)
   - Explosive Growth Bonus: +1.5 * ((max(0, cash - 3000) / 5000) ** 1.5) * 1000
   - Capital Loss Penalty: -2.5 * (max(0, 3000 - cash) / 1000) * 1000
   - Deadweight Trapped Asset Penalty: -2.0 * (deadweight / 1000) * 1000
   - Value Target: z = tanh(Raw / 4000.0)
4. Multi-Task Neural Optimization:
   - Policy Cross-Entropy Loss + Value MSE Loss + World Dynamics Self-Supervised MSE Loss.
5. Continuous Champion Checkpointing & League Logging:
   - Checkpoints: 'weights/alphagoat_grandmaster_champion.pt'
   - Full League Logs: 'data/alphagoat_training_log.json'
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

from src.models.ssl_network import SSLPolicyValueNet
from src.models.encoder import NUM_MACRO_ACTIONS, MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS, encode_observation
from src.models.muzero_mcts import MuZeroLatentMCTS, compute_action_mask
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
from src.training.mega_league import (
    AlphaGoatMegaLeague,
    MEGA_LEAGUE_PERSONALITIES,
    MelonRusherAgent,
    TomatoMonopolistAgent,
    StrawberryAristocratAgent,
    CarrotSprinterAgent,
    WheatIndustrialistAgent,
    PortfolioHedgerAgent,
    DairyBaronAgent,
    GooseEggSwarmAgent,
    WoolSpecialistAgent,
    OrganicFertilizerTycoonAgent,
    PizzaShopSniperAgent,
    BakeryMonopolistAgent,
    SmoothieExploiterAgent,
    MarketPriceCrasherAgent,
    CommoditySpeculatorAgent,
    FourQuadrantOverlordAgent,
    NWMinimalistAgent,
    FiveWorkerSwarmAgent,
    LeanSoloOperatorAgent,
    SerpentineChorerAgent,
    AntiCompetitorShadowAgent,
    GreedySnowballerAgent,
    SafePreserverAgent,
    StochasticPerturbationAdversary,
    KaggleStarterAgent,
    HistoricalCheckpointAgent,
)

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
) -> Tuple[float, float, Dict[str, float]]:
    """
    Locked 4-pillar reward function for Kaggriculture:
      1. Win Margin: (cash - opp_cash)
      2. Explosive Growth Bonus: +1.5 * ((max(0, cash - 3000) / 5000) ** 1.5) * 1000
      3. Capital Loss Penalty: -2.5 * (max(0, 3000 - cash) / 1000) * 1000
      4. Trapped Deadweight Penalty: -2.0 * (deadweight / 1000) * 1000
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


class WorkerMuZeroAgent:
    """Lightweight standalone agent wrapper for multi-worker simulation."""

    def __init__(
        self,
        model: SSLPolicyValueNet,
        num_simulations: int = 40,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
    ):
        self.mcts = MuZeroLatentMCTS(
            model=model,
            num_simulations=num_simulations,
            c_puct=1.5,
            max_latent_depth=3,
            device=torch.device("cpu"),
        )
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature

        self.executors = {
            0: crop_farmer_carrot,
            1: crop_farmer_wheat,
            2: crop_farmer_portfolio,
            3: hybrid_expert_agent,
            4: hybrid_expert_agent,
            5: hybrid_expert_agent,
            6: livestock_agent,
            7: None,  # Liquidation
            8: arbitrage_agent,
            9: livestock_agent,
        }

    def get_action(self, obs: Dict[str, Any], add_noise: bool = True) -> Tuple[Dict[str, Any], np.ndarray, float, int]:
        best_macro, pi_target, val_est = self.mcts.search(
            obs=obs,
            temperature=self.temperature,
            add_dirichlet_noise=add_noise,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
        )

        if best_macro == 7:
            res = hybrid_expert_agent(obs)
            shed = obs.get("private", {}).get("shed", {})
            orders = []
            for item, count in shed.items():
                if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                    orders.append(["SELL", item, count])
            res["market"] = orders[:10]
            return res, pi_target, val_est, best_macro

        executor = self.executors.get(best_macro, hybrid_expert_agent)
        return executor(obs), pi_target, val_est, best_macro


def get_personality_callable(name: str) -> Callable:
    """Returns executable callable for a given personality name."""
    if name in MEGA_LEAGUE_PERSONALITIES:
        return MEGA_LEAGUE_PERSONALITIES[name]
    return hybrid_expert_agent


def run_alphagoat_match_worker(
    match_id: int,
    seed: int,
    champion_state_dict: Dict[str, torch.Tensor],
    opponent_name: str,
    opponent_role: str,  # 'sparring', 'checkpoint', 'main'
    opponent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    num_simulations: int = 40,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Top-level worker function for parallel multiprocessing league match simulations.
    """
    torch.set_num_threads(1)

    # Initialize Champion model
    champ_model = SSLPolicyValueNet()
    champ_model.load_state_dict(champion_state_dict)
    champ_model.eval()

    champ_agent = WorkerMuZeroAgent(
        model=champ_model,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
    )

    # Initialize Opponent
    opp_is_neural = False
    if opponent_role == "sparring":
        opp_callable = get_personality_callable(opponent_name)
    elif opponent_role == "checkpoint" and opponent_state_dict is not None:
        opp_model = SSLPolicyValueNet()
        opp_model.load_state_dict(opponent_state_dict)
        opp_model.eval()
        opp_agent = WorkerMuZeroAgent(
            model=opp_model,
            num_simulations=max(20, num_simulations // 2),
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            temperature=0.8,
        )
        opp_is_neural = True
    else:
        # Self-Play
        opp_agent = WorkerMuZeroAgent(
            model=champ_model,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            temperature=1.0,
        )
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
            act_champ, pi_champ, val_champ, macro_champ = champ_agent.get_action(obs0, add_noise=True)
            champ_history.append({
                "grid": g_champ,
                "scalars": s_champ,
                "pi": pi_champ,
                "action": macro_champ,
            })

            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs1)
                act_opp, pi_opp, val_opp, macro_opp = opp_agent.get_action(obs1, add_noise=True)
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                try:
                    act_opp = opp_callable(obs1)
                except Exception:
                    act_opp = hybrid_expert_agent(obs1)

            env.step([act_champ, act_opp])

        else:
            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs0)
                act_opp, pi_opp, val_opp, macro_opp = opp_agent.get_action(obs0, add_noise=True)
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                try:
                    act_opp = opp_callable(obs0)
                except Exception:
                    act_opp = hybrid_expert_agent(obs0)

            g_champ, s_champ = encode_observation(obs1)
            act_champ, pi_champ, val_champ, macro_champ = champ_agent.get_action(obs1, add_noise=True)
            champ_history.append({
                "grid": g_champ,
                "scalars": s_champ,
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

    raw_champ, z_champ, reward_breakdown = calculate_4pillar_reward(
        cash=champ_cash, opp_cash=opp_cash, deadweight=champ_dw
    )

    # Assign future scalars (horizon=4) and terminal value target
    n_champ = len(champ_history)
    for i in range(n_champ):
        f_idx = min(n_champ - 1, i + 4)
        champ_history[i]["future_scalars"] = champ_history[f_idx]["scalars"]
        champ_history[i]["value"] = z_champ

    collected_transitions = champ_history

    if opp_is_neural and opp_history:
        _, z_opp, _ = calculate_4pillar_reward(cash=opp_cash, opp_cash=champ_cash, deadweight=opp_dw)
        n_opp = len(opp_history)
        for i in range(n_opp):
            f_idx = min(n_opp - 1, i + 4)
            opp_history[i]["future_scalars"] = opp_history[f_idx]["scalars"]
            opp_history[i]["value"] = z_opp
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
        "z_target": z_champ,
        "breakdown": reward_breakdown,
        "duration": duration,
    }

    return collected_transitions, match_stats


# ==============================================================================
# REPLAY DATASET & MULTI-TASK LOSS
# ==============================================================================

class AlphaGoatReplayDataset(Dataset):
    """PyTorch Dataset for Experience Replay Buffer."""

    def __init__(self, buffer: List[Dict[str, Any]]):
        self.grids = torch.tensor(np.array([b["grid"] for b in buffer]), dtype=torch.float32)
        self.scalars = torch.tensor(np.array([b["scalars"] for b in buffer]), dtype=torch.float32)
        self.pi_targets = torch.tensor(np.array([b["pi"] for b in buffer]), dtype=torch.float32)
        self.values = torch.tensor(np.array([b["value"] for b in buffer]), dtype=torch.float32).unsqueeze(1)
        self.future_scalars = torch.tensor(np.array([b["future_scalars"] for b in buffer]), dtype=torch.float32)
        self.actions = torch.tensor(np.array([b["action"] for b in buffer]), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.grids[idx],
            self.scalars[idx],
            self.actions[idx],
            self.pi_targets[idx],
            self.values[idx],
            self.future_scalars[idx],
        )


# ==============================================================================
# ALPHAGOAT HEAVY TRAINING ORCHESTRATOR
# ==============================================================================

class AlphaGoatHeavyTrainer:
    """
    AlphaGoat Heavy Multi-Worker RL Self-Play Training Orchestrator.
    Runs 14-core parallel league simulations, optimizes multi-task loss,
    and maintains dynamic PFSP Elo tracking across all 30 personalities.
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
        log_path: str = "data/alphagoat_training_log.json",
        initial_weights_path: Optional[str] = "weights/muzero_league_champion.pt",
        buffer_capacity: int = 25000,
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
        self.initial_weights_path = initial_weights_path
        self.buffer_capacity = buffer_capacity

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # 1. Initialize Neural Network Champion
        self.model = SSLPolicyValueNet()
        self._load_initial_weights()

        # 2. Initialize Mega League & PFSP Matchmaker
        self.league = AlphaGoatMegaLeague(main_name="AlphaGoatChampion", max_checkpoints=30)
        self.replay_buffer: List[Dict[str, Any]] = []

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=iterations, eta_min=1e-5)

        self.training_logs: List[Dict[str, Any]] = []

    def _load_initial_weights(self):
        if self.initial_weights_path and os.path.exists(self.initial_weights_path):
            print(f"[AlphaGoat] Loading initial weights from: {self.initial_weights_path}")
            ckpt = torch.load(self.initial_weights_path, map_location="cpu")
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd)
            print("[AlphaGoat] Initial weights loaded successfully.")
        elif os.path.exists("weights/grandmaster_rl_champion.pt"):
            print("[AlphaGoat] Loading fallback weights from: weights/grandmaster_rl_champion.pt")
            ckpt = torch.load("weights/grandmaster_rl_champion.pt", map_location="cpu")
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd)
        else:
            print("[AlphaGoat] Initializing model from scratch.")

    def run_training(self):
        """Executes full heavy RL training loop across 14 CPU cores."""
        print("=" * 80)
        print(" ALPHAGOAT 30-PERSONALITY MEGA LEAGUE HEAVY TRAINING SYSTEM")
        print(f" Workers: {self.num_workers} CPU cores | Total Iterations: {self.iterations}")
        print(f" Matches / Iteration: {self.matches_per_iter} | Batch Size: {self.batch_size}")
        print(f" Personalities in League: {len(MEGA_LEAGUE_PERSONALITIES)}")
        print("=" * 80)

        total_matches_run = 0
        overall_start_time = time.time()

        for iteration in range(1, self.iterations + 1):
            iter_start_time = time.time()
            print(f"\n>>> [Iteration {iteration:02d}/{self.iterations:02d}] Launching {self.matches_per_iter} League Matches across {self.num_workers} cores...")

            # 1. Sample Opponents for this iteration using Prioritized Fictitious Self-Play (PFSP)
            match_tasks = []
            champ_state_dict = copy.deepcopy(self.model.state_dict())

            for m_idx in range(self.matches_per_iter):
                opp_name, opp_participant = self.league.sample_opponent(
                    sparring_prob=0.55,
                    checkpoint_prob=0.30,
                    self_play_prob=0.15,
                    gamma=1.5,
                )
                seed = random.randint(100000, 999999)
                match_tasks.append((
                    total_matches_run + m_idx,
                    seed,
                    champ_state_dict,
                    opp_name,
                    opp_participant.role,
                    opp_participant.state_dict,
                    40,  # num_simulations per step
                ))

            # 2. Parallel Multi-Worker Execution across 14 CPU cores
            pool_size = min(self.num_workers, len(match_tasks))
            with mp.Pool(processes=pool_size) as pool:
                results = pool.starmap(run_alphagoat_match_worker, match_tasks)

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
            print(f"[Iter {iteration:02d}] Optimizing Neural Network on {len(self.replay_buffer)} transitions...")
            self.model.train()
            dataset = AlphaGoatReplayDataset(self.replay_buffer)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

            loss_policy_list = []
            loss_val_list = []
            loss_dyn_list = []
            loss_total_list = []

            for epoch in range(self.epochs_per_iter):
                for b_grid, b_scalar, b_act, b_pi, b_val, b_next_scalar in dataloader:
                    self.optimizer.zero_grad()

                    pred_logits, pred_val, pred_dyn = self.model(b_grid, b_scalar, b_act)

                    # Policy Cross-Entropy Loss
                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    loss_policy = -torch.sum(b_pi * log_probs, dim=-1).mean()

                    # Value MSE Loss
                    loss_val = F.mse_loss(pred_val, b_val)

                    # Dynamics Self-Supervised MSE Loss
                    loss_dyn = F.mse_loss(pred_dyn, b_next_scalar)

                    # Total Multi-Task Objective
                    loss_total = loss_policy + 0.5 * loss_val + 0.25 * loss_dyn

                    loss_total.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    loss_policy_list.append(loss_policy.item())
                    loss_val_list.append(loss_val.item())
                    loss_dyn_list.append(loss_dyn.item())
                    loss_total_list.append(loss_total.item())

            self.scheduler.step()
            self.model.eval()

            mean_policy_loss = float(np.mean(loss_policy_list))
            mean_val_loss = float(np.mean(loss_val_list))
            mean_dyn_loss = float(np.mean(loss_dyn_list))
            mean_total_loss = float(np.mean(loss_total_list))

            print(f"[Iter {iteration:02d}] Loss: Total={mean_total_loss:.4f} | Policy={mean_policy_loss:.4f} | "
                  f"Value={mean_val_loss:.4f} | Dyn={mean_dyn_loss:.4f} | LR={self.scheduler.get_last_lr()[0]:.2e}")

            # 5. Checkpoint Active Champion into Mega League Pool & Disk
            champion_save_path = os.path.join(self.checkpoint_dir, "alphagoat_grandmaster_champion.pt")
            torch.save(self.model.state_dict(), champion_save_path)
            self.league.add_checkpoint(iteration=iteration, state_dict=self.model.state_dict())

            iter_duration = time.time() - iter_start_time
            print(f"[Iter {iteration:02d}] Champion Checkpointed -> '{champion_save_path}' ({iter_duration:.1f}s)")

            # 6. Log Comprehensive Metrics
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
        print(f" ALPHAGOAT TRAINING COMPLETE! Total Time: {total_elapsed/60:.1f} mins | Matches: {total_matches_run}")
        print(f" Final Weights Saved: {champion_save_path}")
        print(f" Full Metrics Logged: {self.log_path}")
        print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    trainer = AlphaGoatHeavyTrainer(
        num_workers=14,
        iterations=8,
        matches_per_iter=28,
        batch_size=64,
        epochs_per_iter=4,
        learning_rate=2e-4,
        weight_decay=1e-4,
        checkpoint_dir="weights",
        log_path="data/alphagoat_training_log.json",
        initial_weights_path="weights/muzero_league_champion.pt",
    )
    trainer.run_training()
