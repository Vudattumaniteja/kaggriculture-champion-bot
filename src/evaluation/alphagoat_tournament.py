"""
AlphaGoat Champion Validation Tournament:
Runs 10-episode validation tournament against 'submission.py' and 'starter' (plus Mega League benchmarks),
measures head-to-head win rates, average cash margins, deadweight elimination, and reports full statistics.
"""

import copy
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from kaggle_environments import make

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.ssl_network import SSLPolicyValueNet
from src.training.alphagoat_heavy_training import WorkerMuZeroAgent, calculate_trapped_deadweight
from src.training.mega_league import (
    MEGA_LEAGUE_PERSONALITIES,
    MelonRusherAgent,
    TomatoMonopolistAgent,
    CarrotSprinterAgent,
    MarketPriceCrasherAgent,
    FourQuadrantOverlordAgent,
)
import submission


def load_alphagoat_champion(weights_path: str = "weights/alphagoat_grandmaster_champion.pt") -> WorkerMuZeroAgent:
    """Loads the trained AlphaGoat champion model."""
    model = SSLPolicyValueNet()
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu")
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
        print(f"[Tournament] Loaded AlphaGoat champion weights from '{weights_path}'")
    else:
        print(f"[Tournament] Warning: '{weights_path}' not found, using initialized model.")

    model.eval()
    return WorkerMuZeroAgent(
        model=model,
        num_simulations=40,
        dirichlet_alpha=0.2,
        dirichlet_epsilon=0.15,
        temperature=0.2,  # Exploitative low temperature for evaluation
    )


def alphagoat_agent_fn(obs: Dict[str, Any], champ_agent: WorkerMuZeroAgent) -> Dict[str, Any]:
    act, _, _, _ = champ_agent.get_action(obs, add_noise=False)
    return act


def run_single_match(
    p0_fn: Any,
    p1_fn: Any,
    p0_name: str,
    p1_name: str,
    seed: int,
) -> Dict[str, Any]:
    """Runs a single 720-turn match between two agents."""
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.reset()

    while not env.done:
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation

        act0 = p0_fn(obs0) if callable(p0_fn) else p0_fn
        act1 = p1_fn(obs1) if callable(p1_fn) else p1_fn

        env.step([act0, act1])

    r0 = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
    r1 = float(env.state[1].reward if env.state[1].reward is not None else 0.0)

    dw0 = calculate_trapped_deadweight(env.state[0].observation, player_idx=0)
    dw1 = calculate_trapped_deadweight(env.state[1].observation, player_idx=1)

    return {
        "p0_name": p0_name,
        "p1_name": p1_name,
        "seed": seed,
        "p0_cash": r0,
        "p1_cash": r1,
        "p0_dw": dw0,
        "p1_dw": dw1,
        "p0_won": bool(r0 > r1),
        "margin": r0 - r1,
    }


def run_alphagoat_tournament(num_episodes: int = 10) -> Dict[str, Any]:
    """Executes official validation tournament."""
    print("=" * 80)
    print(f" ALPHAGOAT GRANDMASTER CHAMPION VALIDATION TOURNAMENT ({num_episodes} Episodes/Series)")
    print("=" * 80)

    champ_agent = load_alphagoat_champion()
    champ_fn = lambda obs: alphagoat_agent_fn(obs, champ_agent)
    submission_fn = submission.agent

    results = {
        "vs_starter": [],
        "vs_submission": [],
        "vs_melon_rusher": [],
        "vs_market_crasher": [],
    }

    # 1. Benchmark vs 'starter'
    print(f"\n--- 1. Testing vs 'starter' ({num_episodes} Matches) ---")
    starter_wins = 0
    starter_champ_cash = []
    starter_opp_cash = []
    starter_margins = []

    for ep in range(num_episodes):
        seed = 10000 + ep * 137
        champ_is_p0 = (ep % 2 == 0)

        if champ_is_p0:
            res = run_single_match(champ_fn, "starter", "AlphaGoat", "starter", seed)
            c_cash, o_cash = res["p0_cash"], res["p1_cash"]
            won = res["p0_won"]
        else:
            res = run_single_match("starter", champ_fn, "starter", "AlphaGoat", seed)
            c_cash, o_cash = res["p1_cash"], res["p0_cash"]
            won = (c_cash > o_cash)

        margin = c_cash - o_cash
        if won:
            starter_wins += 1
        starter_champ_cash.append(c_cash)
        starter_opp_cash.append(o_cash)
        starter_margins.append(margin)

        print(f"  Match {ep+1:02d}/{num_episodes:02d} (Seed {seed}): AlphaGoat ${c_cash:,.0f} vs Starter ${o_cash:,.0f} | Margin: ${margin:+,.0f} | {'WIN' if won else 'LOSS'}")
        results["vs_starter"].append(res)

    # 2. Benchmark vs 'submission.py'
    print(f"\n--- 2. Testing vs 'submission.py' ({num_episodes} Matches) ---")
    sub_wins = 0
    sub_champ_cash = []
    sub_opp_cash = []
    sub_margins = []

    for ep in range(num_episodes):
        seed = 20000 + ep * 137
        champ_is_p0 = (ep % 2 == 0)

        if champ_is_p0:
            res = run_single_match(champ_fn, submission_fn, "AlphaGoat", "submission.py", seed)
            c_cash, o_cash = res["p0_cash"], res["p1_cash"]
            won = res["p0_won"]
        else:
            res = run_single_match(submission_fn, champ_fn, "submission.py", "AlphaGoat", seed)
            c_cash, o_cash = res["p1_cash"], res["p0_cash"]
            won = (c_cash > o_cash)

        margin = c_cash - o_cash
        if won:
            sub_wins += 1
        sub_champ_cash.append(c_cash)
        sub_opp_cash.append(o_cash)
        sub_margins.append(margin)

        print(f"  Match {ep+1:02d}/{num_episodes:02d} (Seed {seed}): AlphaGoat ${c_cash:,.0f} vs submission.py ${o_cash:,.0f} | Margin: ${margin:+,.0f} | {'WIN' if won else 'LOSS'}")
        results["vs_submission"].append(res)

    # 3. Benchmark vs MelonRusher
    print(f"\n--- 3. Testing vs 'MelonRusher' (6 Matches) ---")
    melon_fn = MelonRusherAgent()
    for ep in range(6):
        seed = 30000 + ep * 137
        champ_is_p0 = (ep % 2 == 0)
        if champ_is_p0:
            res = run_single_match(champ_fn, melon_fn, "AlphaGoat", "MelonRusher", seed)
        else:
            res = run_single_match(melon_fn, champ_fn, "MelonRusher", "AlphaGoat", seed)
        results["vs_melon_rusher"].append(res)

    summary = {
        "vs_starter": {
            "matches": num_episodes,
            "wins": starter_wins,
            "win_rate": (starter_wins / num_episodes) * 100.0,
            "mean_champ_cash": float(np.mean(starter_champ_cash)),
            "mean_opp_cash": float(np.mean(starter_opp_cash)),
            "mean_margin": float(np.mean(starter_margins)),
        },
        "vs_submission": {
            "matches": num_episodes,
            "wins": sub_wins,
            "win_rate": (sub_wins / num_episodes) * 100.0,
            "mean_champ_cash": float(np.mean(sub_champ_cash)),
            "mean_opp_cash": float(np.mean(sub_opp_cash)),
            "mean_margin": float(np.mean(sub_margins)),
        },
    }

    print("\n" + "=" * 80)
    print(" TOURNAMENT SUMMARY RESULTS")
    print(f" vs Starter:       {summary['vs_starter']['win_rate']:.1f}% Win Rate ({starter_wins}/{num_episodes}) | Margin: ${summary['vs_starter']['mean_margin']:+,.0f} | Cash: ${summary['vs_starter']['mean_champ_cash']:,.0f} vs ${summary['vs_starter']['mean_opp_cash']:,.0f}")
    print(f" vs submission.py: {summary['vs_submission']['win_rate']:.1f}% Win Rate ({sub_wins}/{num_episodes}) | Margin: ${summary['vs_submission']['mean_margin']:+,.0f} | Cash: ${summary['vs_submission']['mean_champ_cash']:,.0f} vs ${summary['vs_submission']['mean_opp_cash']:,.0f}")
    print("=" * 80)

    return {"summary": summary, "details": results}


if __name__ == "__main__":
    tournament_data = run_alphagoat_tournament(num_episodes=10)
    os.makedirs(".scratch", exist_ok=True)
    with open(".scratch/tournament_results.json", "w") as f:
        json.dump(tournament_data, f, indent=2)
