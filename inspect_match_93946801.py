import json
import os
from collections import defaultdict, Counter

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

def analyze_p(p_idx):
    cash_traj = []
    quads = []
    hires = defaultdict(int)
    crops_planted = defaultdict(int)
    crops_harvested = defaultdict(int)
    crop_water = defaultdict(int)
    crop_fertilize = 0
    structures_built = defaultdict(int)
    animals_bought = defaultdict(int)
    animals_placed = defaultdict(int)
    animal_care = 0
    animal_feed = 0
    animal_harvest = 0
    fertilizer_collected = 0
    digs = 0
    
    market_orders_summary = defaultdict(lambda: defaultdict(int))
    sales_breakdown = defaultdict(lambda: {"qty": 0, "revenue": 0.0})
    seed_buys = defaultdict(lambda: {"qty": 0, "cost": 0.0})
    product_buys = defaultdict(lambda: {"qty": 0, "cost": 0.0})
    
    # Track inventory transactions
    for t, step in enumerate(steps):
        s = step[p_idx]
        obs = s.get("observation", {})
        act = s.get("action", {})
        day = obs.get("day", t // 24)
        hour = obs.get("hour", t % 24)
        
        farms = obs.get("farms", [])
        my_farm = farms[p_idx] if p_idx < len(farms) else {}
        money = my_farm.get("money", 0.0)
        cash_traj.append((t, day, hour, money))
        
        uq = my_farm.get("unlocked_quadrants", [])
        if len(uq) > len(quads):
            for q in uq:
                if q not in [x[0] for x in quads]:
                    quads.append((q, t, day, hour, money))
                    
        # actions
        if isinstance(act, dict):
            mkt = act.get("market", [])
            for o in mkt:
                if not isinstance(o, list) or not o:
                    continue
                op = o[0]
                market_orders_summary[op][day] += 1
                if op == "HIRE":
                    hires[day] += (o[1] if len(o) > 1 else 1)
                elif op == "BUY_ANIMAL":
                    animals_bought[o[1]] += (o[2] if len(o) > 2 else 1)
                elif op == "BUY_SEED":
                    seed_buys[o[1]]["qty"] += (o[2] if len(o) > 2 else 1)
                elif op == "BUY_PRODUCT":
                    product_buys[o[1]]["qty"] += (o[2] if len(o) > 2 else 1)
                elif op == "SELL":
                    sales_breakdown[o[1]]["qty"] += (o[2] if len(o) > 2 else 1)
            
            farmer_act = act.get("farmer", [])
            hands_act = act.get("hands", [])
            for a in [farmer_act] + hands_act:
                if not isinstance(a, list) or not a:
                    continue
                op = a[0]
                if op == "BUILD_COOP":
                    structures_built["COOP"] += 1
                elif op == "BUILD_PASTURE":
                    structures_built["PASTURE"] += 1
                elif op == "PLACE_ANIMAL":
                    animals_placed[a[1] if len(a) > 1 else "UNKNOWN"] += 1
                elif op == "PLANT":
                    crops_planted[a[1] if len(a) > 1 else "UNKNOWN"] += 1
                elif op == "WATER":
                    crop_water[day] += 1
                elif op == "FERTILIZE":
                    crop_fertilize += 1
                elif op == "HARVEST":
                    crops_harvested[day] += 1
                elif op == "FEED":
                    animal_feed += 1
                elif op == "CARE":
                    animal_care += 1
                elif op == "COLLECT_FERTILIZER":
                    fertilizer_collected += 1
                elif op == "DIG":
                    digs += 1

    final_obs = steps[-1][p_idx]["observation"]
    final_farm = final_obs.get("farms", [])[p_idx]
    final_shed = final_obs.get("private", {}).get("shed", {})
    final_seeds = final_obs.get("private", {}).get("seeds", {})
    final_tiles = final_farm.get("tiles", [])
    
    # Audit final tiles
    tile_counts = defaultdict(int)
    for r in range(len(final_tiles)):
        for c in range(len(final_tiles[r])):
            t_obj = final_tiles[r][c]
            if isinstance(t_obj, dict):
                k = t_obj.get("kind", "EMPTY")
                if k == "PLANT":
                    crop = t_obj.get("crop")
                    tile_counts[f"PLANT_{crop}"] += 1
                elif k in ["COOP", "PASTURE"]:
                    an = t_obj.get("animal")
                    tile_counts[f"{k}_{an}"] += 1
                else:
                    tile_counts[k] += 1
                    
    return {
        "p_idx": p_idx,
        "reward": steps[-1][p_idx].get("reward"),
        "final_money": cash_traj[-1][3],
        "quads": quads,
        "hires": dict(hires),
        "total_hires": sum(hires.values()),
        "crops_planted": dict(crops_planted),
        "crops_harvested_total": sum(crops_harvested.values()),
        "structures_built": dict(structures_built),
        "animals_bought": dict(animals_bought),
        "animals_placed": dict(animals_placed),
        "animal_care": animal_care,
        "animal_feed": animal_feed,
        "fertilizer_collected": fertilizer_collected,
        "crop_fertilize": crop_fertilize,
        "digs": digs,
        "sales_breakdown": dict(sales_breakdown),
        "seed_buys": dict(seed_buys),
        "product_buys": dict(product_buys),
        "final_shed": final_shed,
        "final_seeds": final_seeds,
        "tile_counts": dict(tile_counts)
    }

p0 = analyze_p(0)
p1 = analyze_p(1)

print("=== P0 (alfphafarm) vs P1 (Ankit0017) Summary ===")
print("P0:")
for k, v in p0.items():
    print(f"  {k}: {v}")
print("\nP1:")
for k, v in p1.items():
    print(f"  {k}: {v}")
