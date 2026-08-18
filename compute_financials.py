import json
import os
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

def compute_financials(p_idx):
    revenue_by_product = defaultdict(float)
    qty_sold_by_product = defaultdict(int)
    cost_by_category = defaultdict(float)
    seed_bought_qty = defaultdict(int)
    
    prev_money = 3000.0
    for t in range(len(steps)):
        s = steps[t][p_idx]
        obs = s.get("observation", {})
        act = s.get("action", {})
        farms = obs.get("farms", [])
        my_farm = farms[p_idx] if p_idx < len(farms) else {}
        curr_money = my_farm.get("money", 0.0)
        
        # Check market prices at this turn
        mkt_prices = obs.get("market", {}).get("prices", {})
        
        # Parse market actions
        if isinstance(act, dict):
            for o in act.get("market", []):
                if not isinstance(o, list) or not o:
                    continue
                op = o[0]
                if op == "SELL":
                    item = o[1]
                    qty = o[2] if len(o) > 2 else 1
                    price = mkt_prices.get(item, 0)
                    # Note: engine calculates exact revenue, let's track estimated and actual
                    qty_sold_by_product[item] += qty
                elif op == "BUY_SEED":
                    item = o[1]
                    qty = o[2] if len(o) > 2 else 1
                    seed_bought_qty[item] += qty
                elif op == "BUY_LAND":
                    cost_by_category["LAND"] += 1000.0 if len(cost_by_category["LAND"]) == 0 else 2000.0 # approximate
                elif op == "BUY_ANIMAL":
                    an = o[1]
                    qty = o[2] if len(o) > 2 else 1
                    # approximate costs: goose 100, cow 1000, sheep 400
                    cost_by_category["ANIMALS"] += (1000 if an == "COW" else (400 if an == "SHEEP" else 100)) * qty
                    
        # Check actual money diffs to get precise sales revenue
        # When money increases, it's sales revenue
        # When money decreases, it's hires/seeds/land/structures/animals
        
    return {
        "qty_sold": dict(qty_sold_by_product),
        "seeds_bought": dict(seed_bought_qty),
    }

p0_fin = compute_financials(0)
p1_fin = compute_financials(1)

print("P0 Fin:", p0_fin)
print("P1 Fin:", p1_fin)
