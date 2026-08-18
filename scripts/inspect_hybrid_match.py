import json

with open('replays/top_competitors/episode-93954943-replay.json') as f:
    replay = json.load(f)

steps = replay['steps']

print('Turn 0 Action P0:', steps[0][0].get('action'))
print('Turn 0 Market P0:', steps[0][0].get('action', {}).get('market'))

for d in [0, 1, 2, 5, 8, 10, 12, 15, 20, 25, 28, 29]:
    idx = min(d * 24, len(steps) - 1)
    obs = steps[idx][0]['observation']
    act = steps[idx][0].get('action', {})
    f = obs['farms'][0]
    p = obs['private']
    m = obs['market']
    t = obs['town']
    print(f"\n=== Day {d:02d} (Turn {idx}) ===")
    print(f"  Money: {f.get('money', 0):.1f}, Unlocked: {f.get('unlocked_quadrants')}, Hires: {f.get('hires_today')}")
    print(f"  Shed: {p.get('shed')}")
    print(f"  Seeds: {p.get('seeds')}")
    print(f"  Prices: {m.get('prices')}")
    print(f"  Shops: {t.get('unlocked_shops')}")
    crops = {}
    animals = {}
    for row in f.get('tiles', []):
        for tile in row:
            if isinstance(tile, dict):
                k = tile.get('kind')
                if k == 'PLANT':
                    c = tile.get('crop')
                    crops[c] = crops.get(c, 0) + 1
                elif k in ('PASTURE', 'COOP'):
                    a = tile.get('animal', 'EMPTY')
                    animals[a] = animals.get(a, 0) + 1
    print(f"  On-Board Crops: {crops}")
    print(f"  On-Board Animals: {animals}")
