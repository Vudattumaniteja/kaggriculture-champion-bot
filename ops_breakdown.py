import json
from collections import defaultdict

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

steps = replay['steps']

print("=" * 70)
print("ACTION & OPERATION BREAKDOWN ACROSS 720 TURNS")
print("=" * 70)

for p in range(2):
    farmer_ops = defaultdict(int)
    hand_ops = defaultdict(int)
    market_ops = defaultdict(int)
    pass_turns = 0
    total_hands_deployed = 0
    
    seeds_bought = defaultdict(int)
    animals_bought = defaultdict(int)
    products_bought = defaultdict(int)
    products_sold = defaultdict(int)
    revenue_by_product = defaultdict(float)
    cost_by_category = defaultdict(float)

    for s_idx, step in enumerate(steps):
        p_data = step[p]
        act = p_data.get('action', {})
        
        # Farmer action
        f_act = act.get('farmer', ['PASS'])
        if f_act:
            f_op = f_act[0]
            farmer_ops[f_op] += 1
            if f_op == 'PASS':
                pass_turns += 1

        # Hands actions
        h_acts = act.get('hands', [])
        total_hands_deployed += len(h_acts)
        for h_act in h_acts:
            if h_act:
                h_op = h_act[0]
                hand_ops[h_op] += 1

        # Market actions
        m_acts = act.get('market', [])
        for m_act in m_acts:
            if m_act:
                m_op = m_act[0]
                market_ops[m_op] += 1
                if m_op == 'BUY_SEED':
                    seeds_bought[m_act[1]] += m_act[2]
                elif m_op == 'BUY_ANIMAL':
                    animals_bought[m_act[1]] += m_act[2]
                elif m_op == 'BUY_PRODUCT':
                    products_bought[m_act[1]] += m_act[2]
                elif m_op == 'SELL':
                    products_sold[m_act[1]] += m_act[2]

    print(f"\n>>> PLAYER {p} <<<")
    print(f"Total Steps: {len(steps)}, Farmer PASS steps: {farmer_ops['PASS']}/{len(steps)} ({farmer_ops['PASS']/len(steps)*100:.1f}%)")
    print("Farmer Operations:")
    for op, cnt in sorted(farmer_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:<20}: {cnt:>4} turns ({cnt/len(steps)*100:.1f}%)")

    print(f"\nFarmhand Operations (Total hands deployed across all turns: {total_hands_deployed}):")
    for op, cnt in sorted(hand_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:<20}: {cnt:>4} times")

    print("\nMarket Operations:")
    for op, cnt in sorted(market_ops.items(), key=lambda x: -x[1]):
        print(f"  {op:<20}: {cnt:>4} times")

    print("\nSeeds Purchased:")
    for seed, qty in sorted(seeds_bought.items()):
        print(f"  {seed:<15}: {qty} seeds")

    print("\nAnimals Purchased:")
    for animal, qty in sorted(animals_bought.items()):
        print(f"  {animal:<15}: {qty} animals")

    print("\nProducts Purchased (Market Arbitrage / Feed / Fertilizer):")
    for prod, qty in sorted(products_bought.items()):
        print(f"  {prod:<15}: {qty} units")

    print("\nProducts Sold:")
    for prod, qty in sorted(products_sold.items()):
        print(f"  {prod:<15}: {qty} units")
