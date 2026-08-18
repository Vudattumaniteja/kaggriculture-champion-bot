import json

with open('replays/top_competitors/episode-93954943-replay.json') as f:
    replay = json.load(f)

steps = replay['steps']

print("=== MARKET ORDERS IN DAYS 0-3 ===")
for t in range(24 * 4):
    m = steps[t][0].get('action', {}).get('market', [])
    if m:
        d = t // 24
        h = t % 24
        print(f"Turn {t:03d} (D{d:02d}:H{h:02d}): {m}")

print("\n=== MARKET ORDERS IN DAYS 10-14 ===")
for t in range(24 * 10, 24 * 14):
    m = steps[t][0].get('action', {}).get('market', [])
    if m:
        d = t // 24
        h = t % 24
        print(f"Turn {t:03d} (D{d:02d}:H{h:02d}): {m}")

print("\n=== MARKET ORDERS IN DAYS 27-29 ===")
for t in range(24 * 27, 24 * 30):
    m = steps[t][0].get('action', {}).get('market', [])
    if m:
        d = t // 24
        h = t % 24
        print(f"Turn {t:03d} (D{d:02d}:H{h:02d}): {m}")
