import json

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

steps = replay['steps']

print("=" * 80)
print("MELON AND CROP TIMELINE INVESTIGATION")
print("=" * 80)

for p in range(2):
    print(f"\n##################### PLAYER {p} #####################")
    for d in range(30):
        # Inspect at hour 0 of day d
        step_idx = d * 24
        if step_idx >= len(steps):
            break
        obs = steps[step_idx][p].get('observation', {})
        farms = obs.get('farms', [])
        if p >= len(farms):
            continue
        tiles = farms[p].get('tiles', [])
        
        melon_tiles = []
        tomato_tiles = []
        wheat_tiles = []
        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                if isinstance(t, dict) and t.get('kind') == 'PLANT':
                    crop = t.get('crop')
                    age = d - t.get('planted_day', d)
                    watered = t.get('watered_today')
                    yield_u = t.get('yield_units', 0)
                    info_str = f"({c},{r}):age={age},yield={yield_u},watered={watered}"
                    if crop == 'MELON':
                        melon_tiles.append(info_str)
                    elif crop == 'TOMATO':
                        tomato_tiles.append(info_str)
                    elif crop == 'WHEAT':
                        wheat_tiles.append(info_str)
        
        print(f"Day {d:>2} (Step {step_idx:>3}): Melons({len(melon_tiles)}): {melon_tiles[:3]} | Tomatoes({len(tomato_tiles)}): {tomato_tiles[:2]} | Wheat({len(wheat_tiles)})")
