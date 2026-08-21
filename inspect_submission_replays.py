import json

d = json.load(open('replays/latest_submission/episode-94617497-replay.json'))
steps = d['steps']

print('--- Lifecycle for alfphafarm (Player 1) in Episode 94617497 ---')
for step_idx in range(len(steps)):
    s = steps[step_idx]
    obs = s[1].get('observation', {})
    my_farm = obs.get('farms', [{}, {}])[1]
    tiles = my_farm.get('tiles', [])
    act = s[1].get('action', {}) or {}
    
    plant_kinds = {}
    pastures = 0
    animals = {'COW': 0, 'SHEEP': 0}
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if isinstance(t, dict):
                k = t.get('kind')
                if k == 'PLANT':
                    crop = t.get('crop')
                    plant_kinds[crop] = plant_kinds.get(crop, 0) + 1
                elif k == 'PASTURE':
                    pastures += 1
                    an = t.get('animal')
                    if an:
                        animals[an] = animals.get(an, 0) + 1
                    
    if step_idx % 24 == 0 or step_idx in [1, 2, 3, 4, 5, 23, 24, 25, 48, 72, 120, 240, 480, 719]:
        day = step_idx // 24
        hour = step_idx % 24
        shed = (obs.get('private', {}) or {}).get('shed', {})
        seeds = (obs.get('private', {}) or {}).get('seeds', {})
        money = my_farm.get('money', 0)
        print(f"Turn {step_idx:3d} (D{day:2d} h{hour:2d}): money=${money:7.1f} | plants={plant_kinds} | pastures={pastures} (cows={animals['COW']}, sheep={animals['SHEEP']}) | shed_melon={shed.get('MELON',0)}, shed_wheat={shed.get('WHEAT',0)} | seeds={dict(seeds)}")
