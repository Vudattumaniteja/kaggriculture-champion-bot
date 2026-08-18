import json

with open('replays/top_competitors/episode-93954943-replay.json') as f:
    replay = json.load(f)

steps = replay['steps']

print("=== DAY 0 ACTIONS (Player 0) ===")
for t in range(24):
    s = steps[t][0]
    obs = s['observation']
    act = s.get('action', {})
    f = obs['farms'][0]
    p = obs['private']
    farmer = act.get('farmer', [])
    hands = act.get('hands', [])
    market = act.get('market', [])
    print(f"Turn {t:02d} (H{obs['hour']:02d}) Cash=${f['money']:<6.1f} | Farmer: {farmer} @ {f['farmer']} | Market: {market}")
    print(f"   Hands ({len(hands)}): {hands}")
    print(f"   Inventories: {p.get('inventories')}")
    print(f"   Shed: {p.get('shed')}")
    print(f"   Seeds: {p.get('seeds')}")
