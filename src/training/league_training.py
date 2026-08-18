"""
AlphaStar Multi-Agent League & MuZero Self-Play Training Pipeline for Kaggriculture.

Orchestration Architecture:
1. Multi-Worker League Match Generator:
   - Evaluates active MuZero champion against 4 distinct sparring bots, past checkpoints, and self-play.
   - FSP (Fictitious Self-Play) sampling with dynamic difficulty weighting.
2. High-Throughput Latent Tree Search:
   - Vectorized imagined rollouts with SSL World Dynamics Head.
   - Dirichlet noise exploration + dynamic zero-deadweight action masking.
3. 4-Pillar Reward Function:
   - Win Margin (cash_0 - cash_1)
   - Explosive Growth Bonus: +1.5 * ((max(0, cash - 3000) / 5000) ** 1.5) * 1000
   - Capital Loss Penalty: -2.5 * (max(0, 3000 - cash) / 1000) * 1000
   - Deadweight Trapped Asset Penalty: -2.0 * (deadweight / 1000) * 1000
   - Value Target: z = tanh(Raw / 4000.0)
4. Multi-Task Neural Optimization:
   - Policy Cross-Entropy Loss + Value MSE Loss + World Dynamics Self-Supervised MSE Loss.
5. Continuous Champion Checkpointing & League Replay Logging:
   - Weights: 'weights/muzero_league_champion.pt'
   - Logs: 'data/league_training_log.json'
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
from src.training.league import (
    MultiAgentLeague,
    melon_jackpot_rusher_agent,
    town_shop_monopolizer_agent,
    livestock_tycoon_agent,
    carrot_clockwork_engine_agent,
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
        num_simulations: int = 50,
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

    def get_action(self, obs: Dict[str, Any], add_noise: bool = True) -> Tuple[Dict[str, Any], np.ndarray, float]:
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
            return res, pi_target, val_est

        executor = self.executors.get(best_macro, hybrid_expert_agent)
        return executor(obs), pi_target, val_est


def run_league_match_worker(
    match_id: int,
    seed: int,
    champion_state_dict: Dict[str, torch.Tensor],
    opponent_name: str,
    opponent_role: str,  # 'sparring', 'checkpoint', 'main'
    opponent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    num_simulations: int = 50,
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
        if opponent_name == "MelonJackpotRusher":
            opp_callable = melon_jackpot_rusher_agent
        elif opponent_name == "TownShopMonopolizer":
            opp_callable = town_shop_monopolizer_agent
        elif opponent_name == "LivestockTycoon":
            opp_callable = livestock_tycoon_agent
        elif opponent_name == "CarrotClockworkEngine":
            opp_callable = carrot_clockwork_engine_agent
        else:
            opp_callable = hybrid_expert_agent
    elif opponent_role == "checkpoint" and opponent_state_dict is not None:
        opp_model = SSLPolicyValueNet()
        opp_model.load_state_dict(opponent_state_dict)
        opp_model.eval()
        opp_agent = WorkerMuZeroAgent(
            model=opp_model,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            temperature=1.0,
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

    # 50% random side swap to eliminate first-player asymmetry
    champ_is_p0 = (match_id % 2 == 0)

    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.reset()

    champ_history: List[Dict[str, Any]] = []
    opp_history: List[Dict[str, Any]] = []

    while not env.done:
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation

        if champ_is_p0:
            g_champ, s_champ = encode_observation(obs0)
            act_champ, pi_champ, val_champ = champ_agent.get_action(obs0, add_noise=True)
            macro_champ = int(np.argmax(pi_champ))
            champ_history.append({
                "grid": g_champ,
                "scalars": s_champ,
                "pi": pi_champ,
                "action": macro_champ,
            })

            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs1)
                act_opp, pi_opp, val_opp = opp_agent.get_action(obs1, add_noise=True)
                macro_opp = int(np.argmax(pi_opp))
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                act_opp = opp_callable(obs1)

            env.step([act_champ, act_opp])

        else:
            if opp_is_neural:
                g_opp, s_opp = encode_observation(obs0)
                act_opp, pi_opp, val_opp = opp_agent.get_action(obs0, add_noise=True)
                macro_opp = int(np.argmax(pi_opp))
                opp_history.append({
                    "grid": g_opp,
                    "scalars": s_opp,
                    "pi": pi_opp,
                    "action": macro_opp,
                })
            else:
                act_opp = opp_callable(obs0)

            g_champ, s_champ = encode_observation(obs1)
            act_champ, pi_champ, val_champ = champ_agent.get_action(obs1, add_noise=True)
            macro_champ = int(np.argmax(pi_champ))
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

    raw_champ, z_champ, _ = calculate_4pillar_reward(cash=champ_cash, opp_cash=opp_cash, deadweight=champ_dw)

    # Assign future scalars (horizon=4) and terminal value target
    n_champ = len(champ_history)
    for i in range(n_champ):
        f_idx = min(n_champ - 1, i + 4)
        champ_history[i]["future_scalars"] = champ_history[f_idx]["scalars"]
        champ_history[i]["value"] = z_champ

    collected_transitions = champ_history

    if opp_is_neural and opp_history:
        raw_opp, z_opp, _ = calculate_4pillar_reward(cash=opp_cash, opp_cash=champ_cash, deadweight=opp_dw)
        n_opp = len(opp_history)
        for i in range(n_opp):
            f_idx = min(n_opp - 1, i + 4)
            opp_history[i]["future_scalars"] = opp_history[f_idx]["scalars"]
            opp_history[i]["value"] = z_opp
        collected_transitions = collected_transitions + opp_history

    duration = time.time() - t0

    match_stats = {
        "match_id": match_id,
        "opponent_name": opponent_name,
        "opponent_role": opponent_role,
        "champ_is_p0": champ_is_p0,
        "champ_cash": champ_cash,
        "opp_cash": opp_cash,
        "champ_dw": champ_dw,
        "opp_dw": opp_dw,
        "champ_won": bool(champ_cash > opp_cash),
        "champ_z": z_champ,
        "duration": duration,
    }

    return collected_transitions, match_stats


class LeagueReplayDataset(Dataset):
    """PyTorch Dataset for Experience Replay Buffer."""

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


def train_league_champion(
    model: SSLPolicyValueNet,
    buffer: List[Dict[str, Any]],
    epochs: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Trains policy head, value head, and SSL world dynamics head on replay experience."""
    dataset = LeagueReplayDataset(buffer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    mse_fn = nn.MSELoss()

    total_loss_accum = 0.0
    pol_loss_accum = 0.0
    val_loss_accum = 0.0
    ssl_loss_accum = 0.0
    count = 0

    for _ in range(epochs):
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

            # SSL Dynamics Loss: World Model Forward Prediction
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


def execute_league_training(
    initial_weights_path: str = "weights/grandmaster_rl_champion.pt",
    champion_weights_path: str = "weights/muzero_league_champion.pt",
    log_save_path: str = "data/league_training_log.json",
    num_iterations: int = 15,
    matches_per_iteration: int = 14,
    num_simulations: int = 50,
    epochs_per_iter: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    replay_capacity: int = 80000,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Executes the complete AlphaStar Multi-Agent League Self-Play Training loop.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if num_workers is None:
        num_workers = max(1, min(mp.cpu_count(), 14))

    os.makedirs("weights", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("=" * 85)
    print("      ALPHASTAR MULTI-AGENT LEAGUE & MUZERO SELF-PLAY TRAINING ENGINE")
    print("=" * 85)
    print(f" Device: {device} | Parallel Workers: {num_workers} (Total CPU Cores: {mp.cpu_count()})")
    print(f" Iterations: {num_iterations} | Matches/Iter: {matches_per_iteration} (Total Planned: {num_iterations * matches_per_iteration})")
    print(f" MCTS Search: MuZero Latent Tree Search (n={num_simulations}, max_depth=3)")
    print(f" Action Masking: Dynamic Zero-Deadweight Masking Active")
    print(f" Base Model: {initial_weights_path}")
    print(f" League Champion Destination: {champion_weights_path}")
    print(f" Multi-Agent League: 4 Sparring Bots + Historical Checkpoints + FSP Sampling")
    print("=" * 85)

    # Initialize Champion Neural Network
    champion_model = SSLPolicyValueNet().to(device)
    if os.path.exists(initial_weights_path):
        champion_model.load_state_dict(torch.load(initial_weights_path, map_location=device, weights_only=True))
        print(f"[*] Loaded initial base weights from {initial_weights_path}\n")
    elif os.path.exists("weights/grandmaster_ssl.pt"):
        champion_model.load_state_dict(torch.load("weights/grandmaster_ssl.pt", map_location=device, weights_only=True))
        print(f"[*] Loaded fallback base weights from weights/grandmaster_ssl.pt\n")
    else:
        print(f"[!] Initial weights not found, training from random initialization.\n")

    # Initialize AlphaStar Multi-Agent League
    league = MultiAgentLeague(main_name="MuZeroChampion", max_checkpoints=20)
    # Register initial base model as checkpoint 0
    state_dict_cpu = {k: v.cpu().clone() for k, v in champion_model.state_dict().items()}
    league.add_checkpoint(iteration=0, state_dict=state_dict_cpu, name="BaseGrandmasterRL")

    history: List[Dict[str, Any]] = []
    sliding_replay_buffer: List[Dict[str, Any]] = []
    total_transitions = 0
    start_time = time.time()

    for iter_idx in range(1, num_iterations + 1):
        iter_t0 = time.time()
        print(f"\n{'='*32} LEAGUE ITERATION {iter_idx:02d}/{num_iterations:02d} {'='*32}")

        state_dict_cpu = {k: v.cpu().clone() for k, v in champion_model.state_dict().items()}

        # 1. SAMPLE MATCH OPPONENTS FROM LEAGUE VIA FSP
        worker_args = []
        opp_names_in_iter = []
        for m_idx in range(matches_per_iteration):
            seed = 20000 + iter_idx * 1000 + m_idx
            opp_name, opp_participant = league.sample_opponent(
                strategy="fsp",
                sparring_prob=0.50,
                checkpoint_prob=0.35,
                self_play_prob=0.15,
            )
            opp_names_in_iter.append(opp_name)
            worker_args.append((
                m_idx,
                seed,
                state_dict_cpu,
                opp_name,
                opp_participant.role,
                opp_participant.state_dict,
                num_simulations,
            ))

        print(f">>> [League Rollouts] Running {matches_per_iteration} matches across {num_workers} parallel workers...")
        opp_counts = {}
        for oname in opp_names_in_iter:
            opp_counts[oname] = opp_counts.get(oname, 0) + 1
        print(f"    Opponent Breakdown: {opp_counts}")

        # Execute parallel match rollouts
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(run_league_match_worker, worker_args)

        iter_transitions = []
        iter_champ_cash = []
        iter_opp_cash = []
        iter_champ_dw = []
        iter_wins = 0

        for transitions, stats in results:
            iter_transitions.extend(transitions)
            iter_champ_cash.append(stats["champ_cash"])
            iter_opp_cash.append(stats["opp_cash"])
            iter_champ_dw.append(stats["champ_dw"])
            if stats["champ_won"]:
                iter_wins += 1

            # Update AlphaStar League stats & Elo
            league.update_match_result(
                p0_name="MuZeroChampion",
                p1_name=stats["opponent_name"],
                p0_cash=stats["champ_cash"],
                p1_cash=stats["opp_cash"],
            )

        # Update sliding replay buffer
        sliding_replay_buffer.extend(iter_transitions)
        if len(sliding_replay_buffer) > replay_capacity:
            excess = len(sliding_replay_buffer) - replay_capacity
            sliding_replay_buffer = sliding_replay_buffer[excess:]

        total_transitions += len(iter_transitions)
        win_rate = (iter_wins / matches_per_iteration) * 100.0
        mean_champ_cash = float(np.mean(iter_champ_cash))
        mean_opp_cash = float(np.mean(iter_opp_cash))
        mean_champ_dw = float(np.mean(iter_champ_dw))
        peak_cash = float(np.max(iter_champ_cash))

        # 2. NEURAL NETWORK TRAINING ON REPLAY BUFFER
        print(f">>> [Optimization] Training on {len(sliding_replay_buffer):,} buffer steps ({epochs_per_iter} epochs, bs={batch_size})...")
        train_stats = train_league_champion(
            model=champion_model,
            buffer=sliding_replay_buffer,
            epochs=epochs_per_iter,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )

        # 3. LEAGUE CHECKPOINT & RECENT CHAMPION UPDATE
        state_dict_trained = {k: v.cpu().clone() for k, v in champion_model.state_dict().items()}
        league.add_checkpoint(
            iteration=iter_idx,
            state_dict=state_dict_trained,
            name=f"Champion_Iter{iter_idx:02d}",
        )

        # Save latest champion weights
        torch.save(state_dict_trained, champion_weights_path)

        iter_duration = time.time() - iter_t0
        elapsed_total = time.time() - start_time

        iter_summary = {
            "iteration": iter_idx,
            "win_rate": round(win_rate, 1),
            "wins": iter_wins,
            "matches": matches_per_iteration,
            "mean_champ_cash": round(mean_champ_cash, 1),
            "mean_opp_cash": round(mean_opp_cash, 1),
            "peak_single_game_cash": round(peak_cash, 1),
            "mean_deadweight": round(mean_champ_dw, 1),
            "loss_total": round(train_stats["avg_loss"], 4),
            "loss_policy": round(train_stats["pol_loss"], 4),
            "loss_value": round(train_stats["val_loss"], 4),
            "loss_dynamics": round(train_stats["ssl_loss"], 4),
            "buffer_size": len(sliding_replay_buffer),
            "iter_duration_seconds": round(iter_duration, 1),
            "elapsed_seconds": round(elapsed_total, 1),
        }
        history.append(iter_summary)

        print(
            f"[+] Iter {iter_idx:02d} Finished ({iter_duration:.1f}s) | "
            f"Win Rate: {win_rate:.1f}% ({iter_wins}/{matches_per_iteration}) | "
            f"Mean Cash: ${mean_champ_cash:,.0f} vs ${mean_opp_cash:,.0f} | "
            f"Peak Cash: ${peak_cash:,.0f} | "
            f"DW: ${mean_champ_dw:.0f} | "
            f"Loss: {train_stats['avg_loss']:.4f} (Pol: {train_stats['pol_loss']:.4f}, Val: {train_stats['val_loss']:.4f}, Dyn: {train_stats['ssl_loss']:.4f})"
        )

        # Save training logs to disk
        league_summary = league.get_league_summary()
        log_payload = {
            "training_history": history,
            "league_leaderboard": league_summary["leaderboard"],
            "league_head_to_head": league_summary["head_to_head"],
            "total_iterations": iter_idx,
            "total_transitions_generated": total_transitions,
            "total_elapsed_seconds": round(elapsed_total, 1),
            "champion_weights_path": champion_weights_path,
        }
        with open(log_save_path, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)

    print("\n" + "=" * 85)
    print("          ALPHASTAR LEAGUE TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 85)
    print(f" Trained Model Saved: {champion_weights_path}")
    print(f" Full Training Log Saved: {log_save_path}")
    print(f" Total Elapsed Time: {(time.time() - start_time) / 60:.2f} minutes")
    print("=" * 85)

    return log_payload


if __name__ == "__main__":
    execute_league_training(
        initial_weights_path="weights/grandmaster_rl_champion.pt",
        champion_weights_path="weights/muzero_league_champion.pt",
        log_save_path="data/league_training_log.json",
        num_iterations=15,
        matches_per_iteration=14,
        num_simulations=50,
        epochs_per_iter=4,
        batch_size=128,
        lr=3e-4,
    )
