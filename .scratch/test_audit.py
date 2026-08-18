"""
Detailed Replay and Move Auditor for ssl_bot vs starter.
Captures full turn-by-turn moves, capital allocation, trapped assets at turn 719,
and compares high-scoring matches ($6,250 and $8,288) with low-scoring matches.
"""

import json
import os
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import torch
from kaggle_environments import make

from src.models.mcts import SSLMCTSAgent
from src.evaluation.runner import resolve_agent

def audit_match(model_path: str, seed: int, episode_num: int, is_swapped: bool = False):
    print(f"\n==========================================")
    print(f"Auditing Episode #{episode_num} (Seed: {seed}, Model: {os.path.basename(model_path)}, Swapped: {is_swapped})")
    print(f"==========================================")

    agent_instance = SSLMCTSAgent(
        model_weights_path=model_path,
        num_simulations=20,
        c_puct=1.5,
        temperature=0.0,
    )
    
    # We will wrap the agent to log chosen macro-action, farmer move, market moves, etc.
    decision_log = []

    def wrapped_ssl(obs, config=None):
        # Record observation details before action
        step = obs.get("step", 0)
        day = step // 24
        hour = step % 24
        money = obs.get("players", [{}])[obs.get("player", 0)].get("bank", 0) if "players" in obs else obs.get("money", 0)
        
        # MCTS search
        macro_act, probs, val = agent_instance.search_best_action(obs)
        action = agent_instance(obs, config)

        decision_log.append({
            "step": step,
            "day": day,
            "hour": hour,
            "money": money,
            "macro_act": macro_act,
            "macro_probs": probs.tolist(),
            "mcts_val": val,
            "farmer": action.get("farmer"),
            "hands": action.get("hands", []),
            "market": action.get("market", []),
        })
        return action

    config = {"episodeSteps": 720, "seed": seed}
    env = make("kaggriculture", configuration=config, debug=False)

    if not is_swapped:
        env.run([wrapped_ssl, "starter"])
        ssl_idx = 0
        starter_idx = 1
    else:
        env.run(["starter", wrapped_ssl])
        ssl_idx = 1
        starter_idx = 0

    final_ssl_reward = env.steps[-1][ssl_idx].reward
    final_starter_reward = env.steps[-1][starter_idx].reward
    print(f"Final Score -> ssl_bot: ${final_ssl_reward} | starter: ${final_starter_reward}")

    # Trace telemetry turn-by-turn
    # Track spending categories
    spending = {
        "coops_built": 0,      # Coops cost time/materials or $0 build? (Check cost)
        "pastures_built": 0,
        "coop_cost": 0,
        "pasture_cost": 0,
        "animals_bought": {"GOOSE": 0, "COW": 0, "SHEEP": 0},
        "animal_cash_spent": 0,
        "seeds_bought": {"WHEAT": 0, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0},
        "seed_cash_spent": 0,
        "labor_hires": 0,
        "labor_cash_spent": 0,
        "land_bought": 0,
        "land_cash_spent": 0,
        "market_revenue": 0,
        "sales_by_item": {},
    }

    # Inspect all steps from env
    # In kaggriculture, env.steps[t][ssl_idx].observation contains game state
    # Let's inspect turn 719 state
    final_obs = env.steps[-1][ssl_idx].observation
    player_obs = final_obs.get("players", [{}])[ssl_idx] if "players" in final_obs else final_obs
    private_obs = final_obs.get("private", {})
    
    # Trapped assets analysis at turn 719
    grid = final_obs.get("grid", []) or final_obs.get("board", [])
    shed = private_obs.get("shed", {})
    seeds_in_hand = private_obs.get("seeds", {})
    money_end = final_ssl_reward

    # Parse spending & market orders from decision_log & env
    day_periods = {
        "Days 0-10 (Turns 0-239)": {"seeds": {}, "animals": {}, "labor": 0, "land": 0, "sales": {}, "actions": []},
        "Days 10-20 (Turns 240-479)": {"seeds": {}, "animals": {}, "labor": 0, "land": 0, "sales": {}, "actions": []},
        "Days 20-30 (Turns 480-719)": {"seeds": {}, "animals": {}, "labor": 0, "land": 0, "sales": {}, "actions": []},
    }

    macro_action_counts = {
        "Days 0-10": {},
        "Days 10-20": {},
        "Days 20-30": {},
    }

    for log in decision_log:
        d = log["day"]
        period_key = "Days 0-10" if d < 10 else ("Days 10-20" if d < 20 else "Days 20-30")
        m_act = log["macro_act"]
        macro_action_counts[period_key][m_act] = macro_action_counts[period_key].get(m_act, 0) + 1

        market_orders = log["market"]
        full_period_key = "Days 0-10 (Turns 0-239)" if d < 10 else ("Days 10-20 (Turns 240-479)" if d < 20 else "Days 20-30 (Turns 480-719)")

        for order in market_orders:
            op = order[0]
            if op == "BUY_SEED":
                crop = order[1]
                qty = order[2]
                day_periods[full_period_key]["seeds"][crop] = day_periods[full_period_key]["seeds"].get(crop, 0) + qty
            elif op == "BUY_ANIMAL":
                anim = order[1]
                qty = order[2]
                day_periods[full_period_key]["animals"][anim] = day_periods[full_period_key]["animals"].get(anim, 0) + qty
            elif op == "HIRE":
                day_periods[full_period_key]["labor"] += 1
            elif op == "BUY_LAND":
                day_periods[full_period_key]["land"] += 1
            elif op == "SELL":
                item = order[1]
                qty = order[2]
                day_periods[full_period_key]["sales"][item] = day_periods[full_period_key]["sales"].get(item, 0) + qty

    return {
        "episode": episode_num,
        "seed": seed,
        "model": os.path.basename(model_path),
        "ssl_reward": final_ssl_reward,
        "starter_reward": final_starter_reward,
        "decision_log": decision_log,
        "day_periods": day_periods,
        "macro_action_counts": macro_action_counts,
        "final_obs": final_obs,
    }

if __name__ == "__main__":
    # Test match 2 and match 8 with both grandmaster_ssl.pt and ssl_alphazero.pt
    seeds_to_test = [
        (43, 1),
        (44, 2),
        (45, 3),
        (46, 4),
        (47, 5),
        (48, 6),
        (49, 7),
        (50, 8),
        (51, 9),
        (52, 10),
    ]

    print("Running audit...")
    res8 = audit_match("weights/grandmaster_ssl.pt", 50, 8)
    res2 = audit_match("weights/grandmaster_ssl.pt", 44, 2)
