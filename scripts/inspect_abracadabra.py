import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('replays/latest_frontier/episode-93981122-replay.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

steps = d['steps']
print(f"Total steps: {len(steps)}")

# Print Abracadabra market actions
for s_idx in range(len(steps)):
    s0 = steps[s_idx][0]
    act0 = s0.get('action', {})
    m = act0.get('market', [])
    p_fert = s0['observation']['market']['prices']['FERTILIZER']
    p_wheat = s0['observation']['market']['prices']['WHEAT']
    p_wool = s0['observation']['market']['prices']['WOOL']
    p_milk = s0['observation']['market']['prices']['MILK']
    
    if m:
        for order in m:
            if order and order[0] in ['SELL', 'BUY', 'BUY_PRODUCT']:
                if s_idx % 24 == 0 or s_idx < 100 or 'FERTILIZER' in order:
                    print(f"Step {s_idx:3d} (D{s_idx//24:02d}:H{s_idx%24:02d}) Abracadabra: {order} | Fert=${p_fert}, Wheat=${p_wheat}, Wool=${p_wool}, Milk=${p_milk}")
