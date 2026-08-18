import json
import os
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

def parse_detailed_transactions(p_idx):
    product_revenue = defaultdict(float)
    product_qty_sold = defaultdict(int)
    
    labor_cost = 0.0
    land_cost = 0.0
    seed_costs = defaultdict(float)
    animal_costs = defaultdict(float)
    structure_costs = defaultdict(float)
    
    # Prices table per turn
    for t in range(len(steps) - 1):
        s_curr = steps[t][p_idx]
        obs = s_curr.get("observation", {})
        act = s_curr.get("action", {})
        mkt_prices = obs.get("market", {}).get("prices", {})
        
        # Check sales
        if isinstance(act, dict):
            for o in act.get("market", []):
                if not isinstance(o, list) or not o:
                    continue
                op = o[0]
                if op == "SELL":
                    item = o[1]
                    qty = o[2] if len(o) > 2 else 1
                    # In market sell, price received is mkt_prices[item]
                    p = mkt_prices.get(item, 0)
                    product_revenue[item] += p * qty
                    product_qty_sold[item] += qty
                elif op == "BUY_SEED":
                    item = o[1]
                    qty = o[2] if len(o) > 2 else 1
                    # Seed costs: MELON=80, CARROT=10, TOMATO=50, WHEAT=20, STRAWBERRY=80
                    # Let's verify standard seed costs
                    seed_price_map = {"CARROT": 10, "WHEAT": 20, "TOMATO": 50, "STRAWBERRY": 80, "MELON": 80}
                    seed_costs[item] += seed_price_map.get(item, 0) * qty
                elif op == "BUY_LAND":
                    # land cost
                    pass
                elif op == "BUY_ANIMAL":
                    an = o[1]
                    qty = o[2] if len(o) > 2 else 1
                    anim_map = {"GOOSE": 100, "SHEEP": 400, "COW": 1000}
                    animal_costs[an] += anim_map.get(an, 0) * qty
                elif op == "HIRE":
                    # cost sequence
                    pass

    return {
        "product_revenue": dict(product_revenue),
        "product_qty_sold": dict(product_qty_sold),
        "seed_costs": dict(seed_costs),
        "animal_costs": dict(animal_costs),
    }

p0_detail = parse_detailed_transactions(0)
p1_detail = parse_detailed_transactions(1)

print("P0 Detail:")
print("  Revenue by product:", p0_detail["product_revenue"])
print("  Qty sold by product:", p0_detail["product_qty_sold"])
print("  Seed costs:", p0_detail["seed_costs"])
print("  Animal costs:", p0_detail["animal_costs"])

print("\nP1 Detail:")
print("  Revenue by product:", p1_detail["product_revenue"])
print("  Qty sold by product:", p1_detail["product_qty_sold"])
print("  Seed costs:", p1_detail["seed_costs"])
print("  Animal costs:", p1_detail["animal_costs"])
