import json
from collections import defaultdict, Counter

with open(r"C:\Users\Manit\Downloads\93943624.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

def analyze_player(player_idx):
    p_name = f"Player {player_idx}"
    
    # State tracking
    money_trajectory = []
    quadrant_history = []
    hires_history = defaultdict(int)
    plantings = defaultdict(int)
    plant_events = []
    structures = []
    animal_purchases = defaultdict(int)
    animal_placements = defaultdict(int)
    care_actions = 0
    feed_actions = 0
    animal_harvests = 0
    fertilizer_collects = 0
    water_actions = 0
    dig_actions = 0
    crop_harvests = 0
    fertilize_crop_actions = 0
    
    market_orders_list = []
    sales_total = defaultdict(int)
    buys_total = defaultdict(int)
    seed_buys_total = defaultdict(int)
    product_buys_total = defaultdict(int)
    
    for step_idx, step in enumerate(steps):
        s = step[player_idx]
        obs = s.get("observation", {})
        act = s.get("action", {})
        
        day = obs.get("day", step_idx // 24)
        hour = obs.get("hour", step_idx % 24)
        
        farms = obs.get("farms", [])
        my_farm = farms[player_idx] if player_idx < len(farms) else {}
        money = my_farm.get("money", 0.0)
        money_trajectory.append((step_idx, day, hour, money))
        
        quads = my_farm.get("unlocked_quadrants", [])
        if len(quads) > len(quadrant_history):
            new_q = [q for q in quads if q not in quadrant_history]
            for nq in new_q:
                quadrant_history.append((nq, step_idx, day, hour, money))
                
        # Market orders
        mkt = act.get("market", []) if isinstance(act, dict) else []
        if mkt:
            market_orders_list.append((step_idx, day, hour, mkt))
            for o in mkt:
                if not isinstance(o, list) or len(o) == 0:
                    continue
                op = o[0]
                if op == "HIRE":
                    hires_history[day] += (o[1] if len(o) > 1 else 1)
                elif op == "BUY_LAND":
                    pass
                elif op == "BUY_ANIMAL":
                    an = o[1] if len(o) > 1 else "UNKNOWN"
                    qty = o[2] if len(o) > 2 else 1
                    animal_purchases[an] += qty
                elif op == "BUY_SEED":
                    seed = o[1] if len(o) > 1 else "UNKNOWN"
                    qty = o[2] if len(o) > 2 else 1
                    seed_buys_total[seed] += qty
                elif op == "BUY_PRODUCT":
                    prod = o[1] if len(o) > 1 else "UNKNOWN"
                    qty = o[2] if len(o) > 2 else 1
                    product_buys_total[prod] += qty
                elif op == "SELL":
                    item = o[1] if len(o) > 1 else "UNKNOWN"
                    qty = o[2] if len(o) > 2 else 1
                    sales_total[item] += qty
                    
        # Units actions
        farmer_act = act.get("farmer", []) if isinstance(act, dict) else []
        hands_act = act.get("hands", []) if isinstance(act, dict) else []
        all_unit_acts = [farmer_act] + hands_act
        
        for a in all_unit_acts:
            if not isinstance(a, list) or len(a) == 0:
                continue
            op = a[0]
            if op == "BUILD_COOP" or op == "BUILD_PASTURE":
                structures.append((step_idx, day, hour, op))
            elif op == "PLACE_ANIMAL":
                an = a[1] if len(a) > 1 else "UNKNOWN"
                animal_placements[an] += 1
            elif op == "PLANT":
                crop = a[1] if len(a) > 1 else "UNKNOWN"
                plantings[crop] += 1
                plant_events.append((step_idx, day, hour, crop))
            elif op == "FEED":
                feed_actions += 1
            elif op == "CARE":
                care_actions += 1
            elif op == "COLLECT_FERTILIZER":
                fertilizer_collects += 1
            elif op == "WATER":
                water_actions += 1
            elif op == "DIG":
                dig_actions += 1
            elif op == "FERTILIZE":
                fertilize_crop_actions += 1
            elif op == "HARVEST":
                crop_harvests += 1
                
    # Final step state
    final_obs = steps[-1][player_idx]["observation"]
    final_farm = final_obs.get("farms", [])[player_idx]
    final_shed = final_obs.get("private", {}).get("shed", {})
    final_seeds = final_obs.get("private", {}).get("seeds", {})
    final_tiles = final_farm.get("tiles", [])
    
    # Audit remaining tiles on field
    field_crops = defaultdict(int)
    field_animals = defaultdict(int)
    field_empty_structures = defaultdict(int)
    weeds = 0
    for r in range(len(final_tiles)):
        for c in range(len(final_tiles[r])):
            t = final_tiles[r][c]
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    field_crops[t.get("crop")] += 1
                elif k in ["COOP", "PASTURE"]:
                    an = t.get("animal")
                    if an:
                        field_animals[an] += 1
                    else:
                        field_empty_structures[k] += 1
                elif k == "WEED":
                    weeds += 1

    return {
        "player_idx": player_idx,
        "final_money": money_trajectory[-1][3],
        "reward": steps[-1][player_idx].get("reward"),
        "quadrant_history": quadrant_history,
        "hires_history": dict(hires_history),
        "total_hires": sum(hires_history.values()),
        "plantings": dict(plantings),
        "total_plantings": sum(plantings.values()),
        "structures": structures,
        "animal_purchases": dict(animal_purchases),
        "animal_placements": dict(animal_placements),
        "care_actions": care_actions,
        "feed_actions": feed_actions,
        "fertilizer_collects": fertilizer_collects,
        "water_actions": water_actions,
        "dig_actions": dig_actions,
        "crop_harvests": crop_harvests,
        "fertilize_crop_actions": fertilize_crop_actions,
        "sales_total": dict(sales_total),
        "seed_buys_total": dict(seed_buys_total),
        "product_buys_total": dict(product_buys_total),
        "final_shed": final_shed,
        "final_seeds": final_seeds,
        "field_crops": dict(field_crops),
        "field_animals": dict(field_animals),
        "field_empty_structures": dict(field_empty_structures),
        "weeds": weeds,
        "money_trajectory": money_trajectory,
    }

p0_stats = analyze_player(0)
p1_stats = analyze_player(1)

print("=== DETAILED STATS COMPARISON ===")
for k in ["player_idx", "reward", "final_money", "quadrant_history", "total_hires", "total_plantings", "plantings", "animal_purchases", "animal_placements", "care_actions", "feed_actions", "fertilizer_collects", "water_actions", "dig_actions", "crop_harvests", "fertilize_crop_actions", "sales_total", "seed_buys_total", "product_buys_total", "final_shed", "final_seeds", "field_crops", "field_animals", "field_empty_structures", "weeds"]:
    print(f"\n--- {k.upper()} ---")
    print(f"P0: {p0_stats[k]}")
    print(f"P1: {p1_stats[k]}")
