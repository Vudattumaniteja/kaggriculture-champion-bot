import json

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

steps = replay['steps']

print("=" * 80)
print("INVESTIGATING WHY CROPS DISAPPEARED ON DAY 4-5 AND DAY 11-12")
print("=" * 80)

for p in range(2):
    print(f"\n##################### PLAYER {p} (Turns 96 - 130) #####################")
    for s_idx in range(96, 130):
        obs = steps[s_idx][p].get('observation', {})
        day = obs.get('day', 0)
        hour = obs.get('hour', 0)
        act = steps[s_idx][p].get('action', {})
        f_act = act.get('farmer', [])
        h_acts = act.get('hands', [])
        m_acts = act.get('market', [])
        
        # Count plants in farm tiles
        tiles = obs.get('farms', [{}, {}])[p].get('tiles', [])
        plants = []
        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                if isinstance(t, dict) and t.get('kind') == 'PLANT':
                    plants.append(f"({c},{r}):{t.get('crop')},age={day-t.get('planted_day',day)},y={t.get('yield_units')}")
        
        print(f"Step {s_idx:>3} (Day {day:>2}, Hr {hour:>2}): Plants={len(plants)} {plants[:2]} | Farmer={f_act} | Hands={h_acts} | Mkt={m_acts}")
