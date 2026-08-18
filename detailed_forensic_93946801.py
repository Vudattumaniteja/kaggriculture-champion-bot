import json
import os
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

def parse_day_by_day(p_idx):
    days = defaultdict(lambda: {
        "start_money": 0, "end_money": 0,
        "hires": 0,
        "seed_buys": defaultdict(int),
        "product_buys": defaultdict(int),
        "sales": defaultdict(int),
        "plantings": defaultdict(int),
        "harvests": defaultdict(int),
        "waters": 0,
        "fertilize": 0,
        "digs": 0,
        "structures": defaultdict(int),
        "animals_bought": defaultdict(int),
        "animals_placed": defaultdict(int),
        "feed": 0,
        "care": 0,
        "collect_fert": 0,
        "land_buys": []
    })
    
    for t, step in enumerate(steps):
        s = step[p_idx]
        obs = s.get("observation", {})
        act = s.get("action", {})
        day = obs.get("day", t // 24)
        hour = obs.get("hour", t % 24)
        farms = obs.get("farms", [])
        my_farm = farms[p_idx] if p_idx < len(farms) else {}
        money = my_farm.get("money", 0.0)
        
        if hour == 0:
            days[day]["start_money"] = money
        if hour == 23 or t == len(steps) - 1:
            days[day]["end_money"] = money
            
        if isinstance(act, dict):
            for o in act.get("market", []):
                if not isinstance(o, list) or not o:
                    continue
                op = o[0]
                if op == "HIRE":
                    days[day]["hires"] += (o[1] if len(o) > 1 else 1)
                elif op == "BUY_SEED":
                    days[day]["seed_buys"][o[1]] += (o[2] if len(o) > 2 else 1)
                elif op == "BUY_PRODUCT":
                    days[day]["product_buys"][o[1]] += (o[2] if len(o) > 2 else 1)
                elif op == "BUY_ANIMAL":
                    days[day]["animals_bought"][o[1]] += (o[2] if len(o) > 2 else 1)
                elif op == "BUY_LAND":
                    days[day]["land_buys"].append((o[1] if len(o) > 1 else "UNKNOWN", t, hour))
                elif op == "SELL":
                    days[day]["sales"][o[1]] += (o[2] if len(o) > 2 else 1)
                    
            for a in [act.get("farmer", [])] + act.get("hands", []):
                if not isinstance(a, list) or not a:
                    continue
                op = a[0]
                if op == "PLANT":
                    days[day]["plantings"][a[1] if len(a) > 1 else "UNKNOWN"] += 1
                elif op == "HARVEST":
                    days[day]["harvests"]["HARVEST"] += 1
                elif op == "WATER":
                    days[day]["waters"] += 1
                elif op == "FERTILIZE":
                    days[day]["fertilize"] += 1
                elif op == "DIG":
                    days[day]["digs"] += 1
                elif op == "BUILD_COOP":
                    days[day]["structures"]["COOP"] += 1
                elif op == "BUILD_PASTURE":
                    days[day]["structures"]["PASTURE"] += 1
                elif op == "PLACE_ANIMAL":
                    days[day]["animals_placed"][a[1] if len(a) > 1 else "UNKNOWN"] += 1
                elif op == "FEED":
                    days[day]["feed"] += 1
                elif op == "CARE":
                    days[day]["care"] += 1
                elif op == "COLLECT_FERTILIZER":
                    days[day]["collect_fert"] += 1
                    
    return days

d0 = parse_day_by_day(0)
d1 = parse_day_by_day(1)

print("=== DAY-BY-DAY BREAKDOWN ===")
for day in range(30):
    p0_d = d0[day]
    p1_d = d1[day]
    print(f"\n--- DAY {day:02d} ---")
    print(f"P0: Start=${p0_d['start_money']:.0f}, End=${p0_d['end_money']:.0f} | Hires={p0_d['hires']} | Plants={dict(p0_d['plantings'])} | Sales={dict(p0_d['sales'])} | SeedBuys={dict(p0_d['seed_buys'])} | Land={p0_d['land_buys']} | Structs={dict(p0_d['structures'])} | AnimBuys={dict(p0_d['animals_bought'])}")
    print(f"P1: Start=${p1_d['start_money']:.0f}, End=${p1_d['end_money']:.0f} | Hires={p1_d['hires']} | Plants={dict(p1_d['plantings'])} | Sales={dict(p1_d['sales'])} | SeedBuys={dict(p1_d['seed_buys'])} | Land={p1_d['land_buys']} | Structs={dict(p1_d['structures'])} | AnimBuys={dict(p1_d['animals_bought'])}")
