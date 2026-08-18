import kaggle_environments
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

from src.agents.hrl_12worker_dispatcher import HRL12WorkerHungarianDispatcher
import submission

print("================================================================================")
print(" FORENSIC COMPARISON: $99.4k BOT (Dispatcher) vs $50k BOT (Submission)")
print("================================================================================")

def run_match_and_profile(agent_fn, label):
    env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    starter_fn = kaggle_environments.environments.get("kaggriculture")["agents"]["starter"]
    
    state = env.reset()
    
    sales_by_prod = {}
    crops_planted = {}
    animals_bought = {}
    max_cows = 0
    max_sheep = 0
    
    for step in range(720):
        obs0 = state[0].observation
        obs1 = state[1].observation
        
        act0 = agent_fn(obs0)
        act1 = starter_fn(obs1)
        
        # Track market sales from act0
        for order in act0.get("market", []):
            if len(order) >= 3 and order[0] == "SELL":
                prod = order[1]
                qty = order[2]
                sales_by_prod[prod] = sales_by_prod.get(prod, 0) + qty
            elif len(order) >= 3 and order[0] == "BUY_ANIMAL":
                a = order[1]
                qty = order[2]
                animals_bought[a] = animals_bought.get(a, 0) + qty
            elif len(order) >= 3 and order[0] == "BUY_SEED":
                c = order[1]
                qty = order[2]
                crops_planted[c] = crops_planted.get(c, 0) + qty
                
        state = env.step([act0, act1])
        
        # Track living animals
        f0 = state[0].observation["farms"][0]
        cows = 0
        sheep = 0
        for row in f0["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP"):
                    a = tile.get("animal")
                    if a == "COW": cows += 1
                    elif a == "SHEEP": sheep += 1
        max_cows = max(max_cows, cows)
        max_sheep = max(max_sheep, sheep)
        
    final_reward = state[0].reward
    final_shed = state[0].observation["private"]["shed"]
    final_money = state[0].observation["farms"][0]["money"]
    
    print(f"\n[{label}] Final Cash: ${final_money:,.1f} | Final Reward: {final_reward}")
    print(f"  Peak Living Herd: {max_cows} Cows, {max_sheep} Sheep (Total: {max_cows + max_sheep})")
    print(f"  Animals Purchased: {animals_bought}")
    print(f"  Seeds Purchased: {crops_planted}")
    print(f"  Market Sales Volume: {sales_by_prod}")
    print(f"  Endgame Shed Deadweight: {final_shed}")
    return {
        "final_money": final_money,
        "max_cows": max_cows,
        "max_sheep": max_sheep,
        "sales": sales_by_prod,
        "shed": final_shed,
    }

old_bot = HRL12WorkerHungarianDispatcher()
res_99k = run_match_and_profile(old_bot, "ORIGINAL $99.4k BOT (Pastoral-Fertilizer Flywheel)")

new_bot = submission.HybridMarketAgent()
res_50k = run_match_and_profile(new_bot, "CURRENT $50k BOT (Hybrid Market Agent)")
