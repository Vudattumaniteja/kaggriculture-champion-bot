import json

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

steps = replay['steps']

# Calculate total cash flow breakdown
for p in range(2):
    print(f"\n==================== FULL CASH FLOW BREAKDOWN: PLAYER {p} ====================")
    money_start = 3000.0
    
    # Track all cash inflows and outflows
    inflows = defaultdict(float)
    outflows = defaultdict(float)
    
    prev_money = money_start
    for s_idx in range(len(steps)):
        obs = steps[s_idx][p].get('observation', {})
        farms = obs.get('farms', [])
        if p >= len(farms):
            continue
        curr_money = farms[p].get('money', 0.0)
        
        # Check diff
        diff = curr_money - prev_money
        act = steps[s_idx][p].get('action', {})
        
        # Market actions
        m_acts = act.get('market', [])
        for m in m_acts:
            op = m[0]
            if op == 'BUY_SEED':
                crop, qty = m[1], m[2]
                outflows[f"SEED_{crop}"] += qty # we'll tally units
            elif op == 'BUY_ANIMAL':
                aname, qty = m[1], m[2]
                outflows[f"ANIMAL_{aname}"] += qty
            elif op == 'BUY_PRODUCT':
                prod, qty = m[1], m[2]
                outflows[f"PRODUCT_{prod}"] += qty
            elif op == 'SELL':
                prod, qty = m[1], m[2]
                inflows[f"SELL_{prod}"] += qty
            elif op == 'HIRE':
                outflows["HIRE_COUNT"] += 1
                
        prev_money = curr_money

    print("Purchases Summary (Units):")
    for k, v in sorted(outflows.items()):
        print(f"  {k:<20}: {v}")
        
    print("\nSales Summary (Units):")
    for k, v in sorted(inflows.items()):
        print(f"  {k:<20}: {v}")

    final_money = steps[-1][p].get('observation', {}).get('farms', [{}, {}])[p].get('money', 0.0)
    print(f"\nStarting Money: ${money_start:.2f}")
    print(f"Ending Money  : ${final_money:.2f}")
    print(f"Net Profit    : ${final_money - money_start:+.2f}")
