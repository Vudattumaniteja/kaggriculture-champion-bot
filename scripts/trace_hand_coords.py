import json

with open(".scratch/our_run.json") as f:
    data = json.load(f)

steps = data["steps"]

# Trace step 12*24 + 16 (D12 H16) and subsequent steps
for s in range(12*24 + 15, 12*24 + 20):
    obs = steps[s][0]["observation"]
    day = obs["day"]
    hour = obs["hour"]
    f0 = obs["farms"][0]
    hands = f0["hands"]
    tiles = f0["tiles"]
    act = steps[s][0].get("action", {})
    print(f"D{day:02d} H{hour:02d} | Hands pos: {hands}")
    print(f"  Actions: Farmer={act.get('farmer')} | Hands={act.get('hands')}")
    print(f"  Tile (3,5) kind: {tiles[5][3]}")
