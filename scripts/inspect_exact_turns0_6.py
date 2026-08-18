import json

with open('replays/top_competitors/episode-93954943-replay.json') as f:
    replay = json.load(f)

steps = replay['steps']

print("=== EXACT ACTIONS TURNS 0-6 ===")
for t in range(7):
    s = steps[t][0]
    obs = s['observation']
    act = s['action']
    f = obs['farms'][0]
    p = obs['private']
    print(f"\n--- TURN {t} ---")
    print("Farmer pos:", f['farmer'], "Action:", act.get('farmer'))
    print("Hands pos:", f['hands'], "Actions:", act.get('hands'))
    print("Inventories:", p.get('inventories'))
    print("Shed:", p.get('shed'))
    print("Market:", act.get('market'))
