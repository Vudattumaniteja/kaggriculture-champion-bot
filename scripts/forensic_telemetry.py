import json
from collections import defaultdict, Counter

REPLAY_PATH = r"C:\Users\Manit\Downloads\93943624.json"

with open(REPLAY_PATH, "r", encoding="utf-8") as f:
    replay = json.load(f)

steps = replay["steps"]

# Market prices over time
commodity_prices = defaultdict(list)
market_trades = []

# Town shop history
town_history = []

# Turn by turn tracking
p0_stats = {
    "cash": [],
    "quadrants": [],
    "hires": defaultdict(int),
    "plantings": defaultdict(int),
    "harvests": defaultdict(int),
    "waterings": 0,
    "digs": 0,
    "fertilizings": 0,
    "feedings": 0,
    "cares": 0,
    "animal_harvests": defaultdict(int),
    "fertilizer_collects": 0,
    "buys": defaultdict(int),
    "sales": defaultdict(int),
    "sales_revenue": defaultdict(float),
    "buys_spend": defaultdict(float),
    "structures": defaultdict(int),
    "animals_placed": defaultdict(int),
}

p1_stats = {
    "cash": [],
    "quadrants": [],
    "hires": defaultdict(int),
    "plantings": defaultdict(int),
    "harvests": defaultdict(int),
    "waterings": 0,
    "digs": 0,
    "fertilizings": 0,
    "feedings": 0,
    "cares": 0,
    "animal_harvests": defaultdict(int),
    "fertilizer_collects": 0,
    "buys": defaultdict(int),
    "sales": defaultdict(int),
    "sales_revenue": defaultdict(float),
    "buys_spend": defaultdict(float),
    "structures": defaultdict(int),
    "animals_placed": defaultdict(int),
}

for t, step in enumerate(steps):
    s0 = step[0]
    s1 = step[1]
    obs0 = s0["observation"]
    obs1 = s1["observation"]
    
    day = obs0.get("day", t // 24)
    hr = obs0.get("hour", t % 24)
    
    # Prices
    mkt = obs0.get("market", {})
    prices = mkt.get("prices", {})
    for k, v in prices.items():
        commodity_prices[k].append(v)
        
    # Town
    town = obs0.get("town", {})
    unlocked_shops = town.get("unlocked_shops", [])
    if t % 24 == 0:
        town_history.append((day, list(unlocked_shops)))
        
    # Cash
    f0 = obs0["farms"][0]
    f1 = obs1["farms"][1]
    p0_stats["cash"].append((t, day, hr, f0["money"]))
    p1_stats["cash"].append((t, day, hr, f1["money"]))
    
    # Process actions
    for p_idx, s, stats, f, p_name in [(0, s0, p0_stats, f0, "P0"), (1, s1, p1_stats, f1, "P1")]:
        act = s.get("action", {})
        if not isinstance(act, dict):
            continue
        
        m_orders = act.get("market", [])
        for o in m_orders:
            if not isinstance(o, list) or len(o) == 0:
                continue
            op = o[0]
            if op == "HIRE":
                qty = o[1] if len(o) > 1 else 1
                stats["hires"][day] += qty
            elif op == "BUY_LAND":
                stats["quadrants"].append((t, day, hr, o[1] if len(o) > 1 else "NE"))
            elif op == "BUY_ANIMAL":
                an = o[1]
                qty = o[2] if len(o) > 2 else 1
                stats["buys"][f"ANIMAL_{an}"] += qty
                # Cost
                cost = {"GOOSE": 300, "COW": 400, "SHEEP": 500}.get(an, 0) * qty
                stats["buys_spend"][f"ANIMAL_{an}"] += cost
            elif op == "BUY_SEED":
                seed = o[1]
                qty = o[2] if len(o) > 2 else 1
                stats["buys"][f"SEED_{seed}"] += qty
                cost = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(seed, 0) * qty
                stats["buys_spend"][f"SEED_{seed}"] += cost
            elif op == "BUY_PRODUCT":
                prod = o[1]
                qty = o[2] if len(o) > 2 else 1
                stats["buys"][f"PROD_{prod}"] += qty
                cost = prices.get(prod, 0) * qty
                stats["buys_spend"][f"PROD_{prod}"] += cost
            elif op == "SELL":
                prod = o[1]
                qty = o[2] if len(o) > 2 else 1
                stats["sales"][prod] += qty
                p_unit = prices.get(prod, 0)
                stats["sales_revenue"][prod] += p_unit * qty
                market_trades.append((t, day, hr, p_name, prod, qty, p_unit))
                
        # Unit actions
        all_units = [act.get("farmer", [])] + act.get("hands", [])
        for u in all_units:
            if not isinstance(u, list) or len(u) == 0:
                continue
            op = u[0]
            if op == "PLANT":
                stats["plantings"][u[1]] += 1
            elif op == "HARVEST":
                stats["harvests"]["CROPS"] += 1
            elif op == "WATER":
                stats["waterings"] += 1
            elif op == "DIG":
                stats["digs"] += 1
            elif op == "FERTILIZE":
                stats["fertilizings"] += 1
            elif op == "FEED":
                stats["feedings"] += 1
            elif op == "CARE":
                stats["cares"] += 1
            elif op == "COLLECT_FERTILIZER":
                stats["fertilizer_collects"] += 1
            elif op == "BUILD_COOP":
                stats["structures"]["COOP"] += 1
            elif op == "BUILD_PASTURE":
                stats["structures"]["PASTURE"] += 1
            elif op == "PLACE_ANIMAL":
                stats["animals_placed"][u[1]] += 1
            elif op == "PICKUP":
                item = u[1] if len(u) > 1 else "ITEM"
                if item in ["GOOSE", "COW", "SHEEP", "EGG", "MILK", "WOOL"]:
                    stats["animal_harvests"][item] += 1

print("=== FINAL TELEMETRY SUMMARY ===")
print("P0 Cash:", p0_stats["cash"][-1], "Reward:", steps[-1][0]["reward"])
print("P1 Cash:", p1_stats["cash"][-1], "Reward:", steps[-1][1]["reward"])

print("\n--- P0 PLANTINGS ---")
for k, v in sorted(p0_stats["plantings"].items()):
    print(f"  {k}: {v}")
print("--- P1 PLANTINGS ---")
for k, v in sorted(p1_stats["plantings"].items()):
    print(f"  {k}: {v}")

print("\n--- P0 BUYS & SPEND ---")
for k, v in sorted(p0_stats["buys"].items()):
    print(f"  {k}: qty={v}, spend=${p0_stats['buys_spend'][k]}")
print("--- P1 BUYS & SPEND ---")
for k, v in sorted(p1_stats["buys"].items()):
    print(f"  {k}: qty={v}, spend=${p1_stats['buys_spend'][k]}")

print("\n--- P0 SALES & REVENUE ---")
for k, v in sorted(p0_stats["sales"].items()):
    print(f"  {k}: qty={v}, est_rev=${p0_stats['sales_revenue'][k]}")
print("--- P1 SALES & REVENUE ---")
for k, v in sorted(p1_stats["sales"].items()):
    print(f"  {k}: qty={v}, est_rev=${p1_stats['sales_revenue'][k]}")

print("\n--- P0 UNIT ACTIONS ---")
print(f"  Water: {p0_stats['waterings']}, Dig: {p0_stats['digs']}, Fertilize: {p0_stats['fertilizings']}, Harvest: {p0_stats['harvests']['CROPS']}")
print(f"  Feed: {p0_stats['feedings']}, Care: {p0_stats['cares']}, CollFert: {p0_stats['fertilizer_collects']}")
print("--- P1 UNIT ACTIONS ---")
print(f"  Water: {p1_stats['waterings']}, Dig: {p1_stats['digs']}, Fertilize: {p1_stats['fertilizings']}, Harvest: {p1_stats['harvests']['CROPS']}")
print(f"  Feed: {p1_stats['feedings']}, Care: {p1_stats['cares']}, CollFert: {p1_stats['fertilizer_collects']}")
