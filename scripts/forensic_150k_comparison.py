import json
import sys
import os
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def analyze_replay(path, target_player=None):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    steps = data['steps']
    info = data.get('info', {})
    team_names = info.get('TeamNames', ['P0', 'P1'])
    
    print(f"\n=======================================================")
    print(f"REPLAY: {os.path.basename(path)}")
    print(f"Teams: P0={team_names[0]} (${steps[-1][0].get('reward', 0):,.1f}) vs P1={team_names[1]} (${steps[-1][1].get('reward', 0):,.1f})")
    print(f"=======================================================")
    
    players_to_analyze = [target_player] if target_player is not None else [0, 1]
    
    for p in players_to_analyze:
        if p >= len(team_names):
            continue
        p_name = team_names[p]
        p_score = steps[-1][p].get('reward', 0)
        
        # Track detailed stats
        daily_cash = {}
        daily_quads = {}
        daily_living_cows = {}
        daily_living_sheep = {}
        daily_living_geese = {}
        daily_crop_counts = defaultdict(lambda: defaultdict(int))
        daily_hires = defaultdict(int)
        
        buys = Counter()
        buys_cost = defaultdict(float)
        sales = Counter()
        sales_rev = defaultdict(float)
        
        unit_ops = Counter()
        idle_count = 0
        total_unit_turns = 0
        
        quad_unlock_day = {}
        
        prev_quad_count = 1
        
        for t, step in enumerate(steps):
            day = t // 24
            hour = t % 24
            
            p_data = step[p]
            obs = p_data.get('observation', {})
            act = p_data.get('action', {})
            
            farms = obs.get('farms', [])
            if p < len(farms):
                my_farm = farms[p]
                money = my_farm.get('money', 0.0)
                quads = my_farm.get('unlocked_quadrants', ['NW'])
                if len(quads) > prev_quad_count:
                    for q in quads:
                        if q not in quad_unlock_day:
                            quad_unlock_day[q] = (day, hour)
                    prev_quad_count = len(quads)
                
                if hour == 23 or t == len(steps) - 1:
                    daily_cash[day] = money
                    daily_quads[day] = list(quads)
                
                # Scan tiles for animals and crops
                tiles = my_farm.get('tiles', [])
                cows = 0
                sheep = 0
                geese = 0
                crop_types = Counter()
                for r in range(len(tiles)):
                    for c in range(len(tiles[r])):
                        tile = tiles[r][c]
                        if isinstance(tile, dict):
                            k = tile.get('kind')
                            an = tile.get('animal')
                            if k == 'PASTURE':
                                if an == 'COW': cows += 1
                                elif an == 'SHEEP': sheep += 1
                            elif k == 'COOP':
                                if an == 'GOOSE': geese += 1
                            elif k == 'PLANT':
                                crop_types[tile.get('crop', 'UNKNOWN')] += 1
                
                if hour == 23 or t == len(steps) - 1:
                    daily_living_cows[day] = cows
                    daily_living_sheep[day] = sheep
                    daily_living_geese[day] = geese
                    for cname, cnt in crop_types.items():
                        daily_crop_counts[day][cname] = cnt

            # Process action
            if isinstance(act, dict):
                farmer_act = act.get('farmer', [])
                hands_act = act.get('hands', [])
                all_units = [farmer_act] + hands_act
                for u in all_units:
                    if not u: continue
                    total_unit_turns += 1
                    op = u[0]
                    unit_ops[op] += 1
                    if op == 'PASS':
                        idle_count += 1
                
                mkt_acts = act.get('market', [])
                mkt_prices = obs.get('market', {}).get('prices', {})
                for m in mkt_acts:
                    if not isinstance(m, list) or len(m) == 0: continue
                    mop = m[0]
                    if mop == 'HIRE':
                        daily_hires[day] += 1
                    elif mop == 'BUY_ANIMAL':
                        aname = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        buys[f"ANIMAL_{aname}"] += qty
                    elif mop == 'BUY_SEED':
                        cname = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        buys[f"SEED_{cname}"] += qty
                    elif mop == 'BUY_PRODUCT':
                        pname = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        buys[f"PRODUCT_{pname}"] += qty
                    elif mop == 'SELL':
                        pname = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        pr = mkt_prices.get(pname, 0.0)
                        sales[pname] += qty
                        sales_rev[pname] += qty * pr
        
        # Print summary
        print(f"\n--- Player {p}: {p_name} ---")
        print(f"Final Reward / Cash: ${p_score:,.1f}")
        print(f"Quadrant Unlocks: {quad_unlock_day}")
        print(f"Total Farmhands Hired: {sum(daily_hires.values())} (by day range: D0-6={sum(daily_hires[d] for d in range(7))}, D7-8={sum(daily_hires[d] for d in [7,8])}, D9-28={sum(daily_hires[d] for d in range(9,29))})")
        print(f"Total Buys: {dict(buys)}")
        print(f"Total Sales Qty: {dict(sales)}")
        print(f"Total Sales Rev (est): { {k: f'${v:,.0f}' for k, v in sales_rev.items()} }")
        print(f"Total Sales Sum (est): ${sum(sales_rev.values()):,.0f}")
        print(f"Unit Actions Breakdown: {dict(unit_ops.most_common(12))}")
        print(f"Idle Ratio: {idle_count}/{total_unit_turns} ({idle_count/max(1,total_unit_turns)*100:.1f}%)")
        
        print("\nDaily Progression (every 3 days):")
        print(f"{'Day':>4} | {'Cash':>9} | {'Quads':>2} | {'Cows':>4} | {'Sheep':>5} | {'Wheat':>5} | {'Straw':>5} | {'Melon':>5} | {'Carrot':>6} | {'Tomato':>6}")
        print("-" * 75)
        for d in range(30):
            if d % 3 == 0 or d in [1, 5, 10, 11, 12, 28, 29]:
                c_cash = daily_cash.get(d, 0)
                n_quads = len(daily_quads.get(d, ['NW']))
                c_cows = daily_living_cows.get(d, 0)
                c_sheep = daily_living_sheep.get(d, 0)
                crops = daily_crop_counts.get(d, {})
                print(f"{d:>4} | ${c_cash:>8,.0f} | {n_quads:>2} | {c_cows:>4} | {c_sheep:>5} | {crops.get('WHEAT', 0):>5} | {crops.get('STRAWBERRY', 0):>5} | {crops.get('MELON', 0):>5} | {crops.get('CARROT', 0):>6} | {crops.get('TOMATO', 0):>6}")

if __name__ == '__main__':
    replays = [
        "replays/top_competitors/episode-93954943-replay.json", # peikopon $155k vs kowalskii $152k
        "replays/top_competitors/episode-93926854-replay.json", # Efe Can $142.9k vs d0gied $131k
        "replays/top_competitors/episode-93941696-replay.json", # Galaxantic $141k vs One-For-All $135k
    ]
    for r in replays:
        if os.path.exists(r):
            analyze_replay(r)
