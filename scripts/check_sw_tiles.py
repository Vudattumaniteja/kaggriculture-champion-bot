import json

with open(".scratch/our_run.json") as f:
    data = json.load(f)

steps = data["steps"]

# Check when SW was unlocked and what was on (3,5), (3,6), (2,5), (2,6)
sw_tiles = [(3, 5), (3, 6), (2, 5), (2, 6)]

for s in [0, 6*24, 9*24, 11*24, 12*24, 15*24, 20*24, 25*24]:
    obs = steps[s][0]["observation"]
    day = obs["day"]
    f0 = obs["farms"][0]
    unlocked = f0["unlocked_quadrants"]
    tiles = f0["tiles"]
    print(f"Day {day:02d} | Unlocked Quads: {unlocked} | Money: ${f0['money']:<6.0f}")
    for tx, ty in sw_tiles:
        t_val = tiles[ty][tx]
        print(f"  Tile ({tx},{ty}): {t_val}")
