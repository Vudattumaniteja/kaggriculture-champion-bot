import json, glob

for f in sorted(glob.glob('replays/latest_submission/*.json')):
    d = json.load(open(f))
    names = d.get('info', {}).get('TeamNames', ['P0', 'P1'])
    steps = d['steps']
    print('='*70)
    print(f"Match: {names[0]} vs {names[1]}")
    
    for p_idx in [0, 1]:
        if names[p_idx] != 'alfphafarm':
            continue
        print(f"  alfphafarm (P{p_idx}):")
        for day in range(6):
            step_idx = day * 24
            obs = steps[step_idx][p_idx].get('observation', {})
            farm = obs.get('farms', [{}, {}])[p_idx]
            priv = obs.get('private', {}) or {}
            tiles = farm.get('tiles', [])
            
            cows = 0
            sheep = 0
            melons = 0
            wheats = 0
            for r in range(len(tiles)):
                for c in range(len(tiles[r])):
                    t = tiles[r][c]
                    if isinstance(t, dict):
                        if t.get('kind') == 'PASTURE':
                            if t.get('animal') == 'COW': cows += 1
                            elif t.get('animal') == 'SHEEP': sheep += 1
                        elif t.get('kind') == 'PLANT':
                            if t.get('crop') == 'MELON': melons += 1
                            elif t.get('crop') == 'WHEAT': wheats += 1
                            
            print(f"    Day {day}: Money=${farm.get('money',0):6.1f}, Cows={cows}, Sheep={sheep}, Melons={melons}, Wheats={wheats}, ShedWheat={priv.get('shed',{}).get('WHEAT',0)}, Hands={len(farm.get('hands',[]))}")
