import json
import os
import sys
from collections import defaultdict, Counter

REPLAY_PATH = r"C:\Users\Manit\Downloads\93943624.json"

with open(REPLAY_PATH, "r") as f:
    replay = json.load(f)

steps = replay["steps"]
num_steps = len(steps)

print(f"Total steps: {num_steps}")

# Track player stats
p0_money = []
p1_money = []
p0_actions = []
p1_actions = []
p0_market_orders = []
p1_market_orders = []

p0_quadrants = []
p1_quadrants = []
p0_hires_by_day = defaultdict(int)
p1_hires_by_day = defaultdict(int)

p0_plantings = defaultdict(int)
p1_plantings = defaultdict(int)
p0_plant_events = []
p1_plant_events = []

p0_structures = []
p1_structures = []

p0_sales = defaultdict(int)
p1_sales = defaultdict(int)
p0_buys = defaultdict(int)
p1_buys = defaultdict(int)

prices_history = defaultdict(list)

# Let's inspect step 0 to step 719
for step_idx, step in enumerate(steps):
    s0 = step[0]
    s1 = step[1]
    
    obs0 = s0.get("observation", {})
    obs1 = s1.get("observation", {})
    
    act0 = s0.get("action", {})
    act1 = s1.get("action", {})
    
    day = obs0.get("day", step_idx // 24)
    hour = obs0.get("hour", step_idx % 24)
    
    farms0 = obs0.get("farms", [])
    farms1 = obs1.get("farms", [])
    
    m0 = farms0[0]["money"] if len(farms0) > 0 else 0
    m1 = farms0[1]["money"] if len(farms0) > 1 else 0
    p0_money.append((step_idx, day, hour, m0))
    p1_money.append((step_idx, day, hour, m1))
    
    quad0 = farms0[0].get("unlocked_quadrants", []) if len(farms0) > 0 else []
    quad1 = farms0[1].get("unlocked_quadrants", []) if len(farms0) > 1 else []
    
    if len(quad0) > len(p0_quadrants):
        new_q = [q for q in quad0 if q not in p0_quadrants]
        p0_quadrants = list(quad0)
        print(f"[P0 Unlock] Step {step_idx} (Day {day}, Hr {hour}): Unlocked {new_q} (Current money: ${m0:.1f})")
        
    if len(quad1) > len(p1_quadrants):
        new_q = [q for q in quad1 if q not in p1_quadrants]
        p1_quadrants = list(quad1)
        print(f"[P1 Unlock] Step {step_idx} (Day {day}, Hr {hour}): Unlocked {new_q} (Current money: ${m1:.1f})")
        
    mo0 = act0.get("market", []) if isinstance(act0, dict) else []
    mo1 = act1.get("market", []) if isinstance(act1, dict) else []
    
    if mo0:
        p0_market_orders.append((step_idx, day, hour, mo0))
        for order in mo0:
            if isinstance(order, list) and len(order) > 0:
                op = order[0]
                if op == "BUY_LAND":
                    pass
                elif op == "HIRE":
                    p0_hires_by_day[day] += (order[1] if len(order) > 1 else 1)
                elif op == "SELL":
                    item = order[1] if len(order) > 1 else "UNKNOWN"
                    qty = order[2] if len(order) > 2 else 1
                    p0_sales[item] += qty
                elif op == "BUY":
                    item = order[1] if len(order) > 1 else "UNKNOWN"
                    qty = order[2] if len(order) > 2 else 1
                    p0_buys[item] += qty
                    
    if mo1:
        p1_market_orders.append((step_idx, day, hour, mo1))
        for order in mo1:
            if isinstance(order, list) and len(order) > 0:
                op = order[0]
                if op == "BUY_LAND":
                    pass
                elif op == "HIRE":
                    p1_hires_by_day[day] += (order[1] if len(order) > 1 else 1)
                elif op == "SELL":
                    item = order[1] if len(order) > 1 else "UNKNOWN"
                    qty = order[2] if len(order) > 2 else 1
                    p1_sales[item] += qty
                elif op == "BUY":
                    item = order[1] if len(order) > 1 else "UNKNOWN"
                    qty = order[2] if len(order) > 2 else 1
                    p1_buys[item] += qty

    f_act0 = act0.get("farmer", []) if isinstance(act0, dict) else []
    h_act0 = act0.get("hands", []) if isinstance(act0, dict) else []
    
    f_act1 = act1.get("farmer", []) if isinstance(act1, dict) else []
    h_act1 = act1.get("hands", []) if isinstance(act1, dict) else []

    all_acts_0 = [f_act0] + h_act0
    for a in all_acts_0:
        if a and isinstance(a, list) and len(a) > 0:
            if a[0] == "PLANT":
                crop = a[1] if len(a) > 1 else "UNKNOWN"
                p0_plantings[crop] += 1
                p0_plant_events.append((step_idx, day, hour, crop))
            elif a[0] == "BUILD":
                struct = a[1] if len(a) > 1 else "UNKNOWN"
                p0_structures.append((step_idx, day, hour, struct))
                
    all_acts_1 = [f_act1] + h_act1
    for a in all_acts_1:
        if a and isinstance(a, list) and len(a) > 0:
            if a[0] == "PLANT":
                crop = a[1] if len(a) > 1 else "UNKNOWN"
                p1_plantings[crop] += 1
                p1_plant_events.append((step_idx, day, hour, crop))
            elif a[0] == "BUILD":
                struct = a[1] if len(a) > 1 else "UNKNOWN"
                p1_structures.append((step_idx, day, hour, struct))

    market_info = obs0.get("market", {})
    if "prices" in market_info:
        for k, v in market_info["prices"].items():
            prices_history[k].append((step_idx, v))

print("\n=== SUMMARY OVERVIEW ===")
print(f"Final Rewards: P0 = {s0.get('reward')}, P1 = {s1.get('reward')}")
print(f"Final Money (Step 719): P0 = ${p0_money[-1][3]:.2f}, P1 = ${p1_money[-1][3]:.2f}")

print("\n--- Quadrants Unlocked ---")
print(f"P0 Quadrants: {p0_quadrants}")
print(f"P1 Quadrants: {p1_quadrants}")

print("\n--- Hires By Day ---")
print(f"P0 Hires: {dict(p0_hires_by_day)}")
print(f"P1 Hires: {dict(p1_hires_by_day)}")

print("\n--- Plantings ---")
print(f"P0 Total Plantings: {dict(p0_plantings)}")
print(f"P1 Total Plantings: {dict(p1_plantings)}")

print("\n--- Structures Built ---")
print(f"P0 Structures: {p0_structures}")
print(f"P1 Structures: {p1_structures}")

print("\n--- Market Purchases ---")
print(f"P0 Buys: {dict(p0_buys)}")
print(f"P1 Buys: {dict(p1_buys)}")

print("\n--- Market Sales ---")
print(f"P0 Sales: {dict(p0_sales)}")
print(f"P1 Sales: {dict(p1_sales)}")
