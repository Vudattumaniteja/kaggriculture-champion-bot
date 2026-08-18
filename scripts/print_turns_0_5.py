import json
with open('replays/top_competitors/episode-93954943-replay.json') as f:
    replay = json.load(f)
steps = replay['steps']
for t in range(6):
    s = steps[t][0]
    print(f"Turn {t}: Farmer: {s['action'].get('farmer')} @ {s['observation']['farms'][0]['farmer']} | Hands: {s['action'].get('hands')}")
