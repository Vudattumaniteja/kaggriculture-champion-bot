"""
Test and Benchmark Script for HRL 12-Worker Hungarian Chore Dispatcher.
Simulates a full 720-step match against baseline 'starter' and logs detailed financial and operational statistics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
from collections import defaultdict
from kaggle_environments import make

from src.agents.hrl_12worker_dispatcher import agent as hrl_agent

def run_hrl_benchmark(episodes: int = 1, seed: int = 42):
    print(f"=== Running HRL 12-Worker Dispatcher Benchmark ({episodes} episodes) ===")
    
    for ep in range(1, episodes + 1):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed + ep}, debug=True)
        start_time = time.time()
        env.run([hrl_agent, "starter"])
        duration = time.time() - start_time
        
        steps = env.steps
        final_step = steps[-1]
        
        r0 = final_step[0].reward
        r1 = final_step[1].reward
        
        # Analyze financial and operational metrics
        total_fertilizer_sold = 0
        total_fertilizer_revenue = 0.0
        total_wool_sold = 0
        total_milk_sold = 0
        total_care_actions = 0
        total_feed_actions = 0
        total_fert_collected = 0
        total_hires = 0
        
        daily_cash = {}
        
        for t, step_data in enumerate(steps):
            s0 = step_data[0]
            obs0 = s0.get("observation", {})
            act0 = s0.get("action", {})
            day = obs0.get("day", t // 24)
            hour = obs0.get("hour", t % 24)
            
            my_farm = obs0.get("farms", [{}, {}])[0] if len(obs0.get("farms", [])) > 0 else {}
            money = my_farm.get("money", 0.0)
            if hour == 23 or t == len(steps) - 1:
                daily_cash[day] = money
                
            if isinstance(act0, dict):
                for order in act0.get("market", []):
                    if isinstance(order, list) and len(order) > 0:
                        if order[0] == "SELL":
                            item = order[1]
                            qty = order[2] if len(order) > 2 else 1
                            if item == "FERTILIZER":
                                total_fertilizer_sold += qty
                            elif item == "WOOL":
                                total_wool_sold += qty
                            elif item == "MILK":
                                total_milk_sold += qty
                        elif order[0] == "HIRE":
                            total_hires += (order[1] if len(order) > 1 else 1)
                            
                for u_act in [act0.get("farmer", [])] + act0.get("hands", []):
                    if isinstance(u_act, list) and len(u_act) > 0:
                        op = u_act[0]
                        if op == "CARE":
                            total_care_actions += 1
                        elif op == "FEED":
                            total_feed_actions += 1
                        elif op == "COLLECT_FERTILIZER":
                            total_fert_collected += 1
                            
        print(f"\n--- Episode {ep} Results ---")
        print(f"HRL Dispatcher Final Cash: ${r0:,.1f}")
        print(f"Starter Final Cash:        ${r1:,.1f}")
        print(f"Profit Margin:             ${(r0 - r1):+,.1f}")
        print(f"Simulation Duration:       {duration:.2f}s (Avg {(duration*1000/720):.2f} ms/step)")
        print(f"Total Fertilizer Sold:     {total_fertilizer_sold} units")
        print(f"Total Wool Sold:           {total_wool_sold} units")
        print(f"Total Milk Sold:           {total_milk_sold} units")
        print(f"Total Fertilizer Collected:{total_fert_collected} units")
        print(f"Total Care Actions:        {total_care_actions}")
        print(f"Total Feed Actions:        {total_feed_actions}")
        print(f"Total Labor Hires:         {total_hires}")
        print("\nDaily Cash Milestones:")
        for d in sorted(daily_cash.keys()):
            if d % 3 == 0 or d == 29:
                print(f"  Day {d:02d}: ${daily_cash[d]:,.0f}")
                
if __name__ == "__main__":
    run_hrl_benchmark(episodes=5, seed=42)
