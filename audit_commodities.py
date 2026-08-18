import json
from collections import defaultdict

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

steps = replay['steps']

print("=" * 80)
print("1. FINANCIAL AND COMMODITY AUDIT")
print("=" * 80)

for p in range(2):
    print(f"\n====================== PLAYER {p} ======================")
    seeds_bought = defaultdict(int)
    animals_bought = defaultdict(int)
    products_bought = defaultdict(int)
    products_sold = defaultdict(int)
    
    seeds_cost = 0.0
    animals_cost = 0.0
    product_buy_cost = 0.0
    sales_revenue = 0.0
    hire_cost = 0.0
    land_cost = 0.0
    
    # Track tile states over time
    crops_planted = defaultdict(int)
    crops_harvested = defaultdict(int)
    animals_harvested = defaultdict(int)
    feed_actions = 0
    water_actions = 0
    fertilize_actions = 0
    care_actions = 0
    coop_builds = 0
    pasture_builds = 0
    
    for s_idx in range(len(steps)):
        step_data = steps[s_idx][p]
        obs = step_data.get('observation', {})
        act = step_data.get('action', {})
        
        # Actions
        farmer_act = act.get('farmer', [])
        hands_act = act.get('hands', [])
        market_act = act.get('market', [])
        
        all_unit_acts = [farmer_act] + hands_act
        for u_act in all_unit_acts:
            if not u_act:
                continue
            op = u_act[0]
            if op == 'PLANT':
                crops_planted[u_act[1]] += 1
            elif op == 'BUILD_COOP':
                coop_builds += 1
            elif op == 'BUILD_PASTURE':
                pasture_builds += 1
            elif op == 'FEED':
                feed_actions += 1
            elif op == 'WATER':
                water_actions += 1
            elif op == 'CARE':
                care_actions += 1
            elif op == 'FERTILIZE':
                fertilize_actions += 1
        
        # Market
        for m in market_act:
            if not m:
                continue
            op = m[0]
            if op == 'HIRE':
                hire_cost += 2 # or whatever cost
            elif op == 'BUY_LAND':
                land_cost += 1000 # or tier
            elif op == 'BUY_SEED':
                crop, qty = m[1], m[2]
                seeds_bought[crop] += qty
            elif op == 'BUY_ANIMAL':
                aname, qty = m[1], m[2]
                animals_bought[aname] += qty
            elif op == 'BUY_PRODUCT':
                prod, qty = m[1], m[2]
                products_bought[prod] += qty
            elif op == 'SELL':
                prod, qty = m[1], m[2]
                products_sold[prod] += qty

    print("Crops Planted:")
    for c, cnt in sorted(crops_planted.items()):
        print(f"  {c:<12}: {cnt} planted (Seeds bought: {seeds_bought.get(c, 0)})")
    
    print("\nStructures Built:")
    print(f"  COOP    : {coop_builds}")
    print(f"  PASTURE : {pasture_builds}")
    
    print("\nAnimal / Farming Actions:")
    print(f"  FEED actions     : {feed_actions}")
    print(f"  CARE actions     : {care_actions}")
    print(f"  WATER actions    : {water_actions}")
    print(f"  FERTILIZE actions: {fertilize_actions}")
    
    print("\nProducts Sold (Units):")
    for prod, qty in sorted(products_sold.items()):
        print(f"  {prod:<12}: {qty} units")

    print("\nProducts Bought (Units):")
    for prod, qty in sorted(products_bought.items()):
        print(f"  {prod:<12}: {qty} units")
