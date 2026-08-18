import json

with open(".scratch/our_run.json") as f:
    data = json.load(f)

steps = data["steps"]

# Trace Day 16 (steps 16*24 to 17*24)
for s in range(16*24, 18*24):
    obs = steps[s][0]["observation"]
    day = obs["day"]
    hour = obs["hour"]
    act = steps[s][0].get("action", {})
    farmer_act = act.get("farmer", [])
    hands_act = act.get("hands", [])

    water_cnt = (1 if farmer_act and farmer_act[0] == "WATER" else 0) + sum(1 for h in hands_act if h and h[0] == "WATER")
    feed_cnt = (1 if farmer_act and farmer_act[0] == "FEED" else 0) + sum(1 for h in hands_act if h and h[0] == "FEED")
    fert_cnt = (1 if farmer_act and farmer_act[0] == "FERTILIZE" else 0) + sum(1 for h in hands_act if h and h[0] == "FERTILIZE")
    print(f"D{day:02d} H{hour:02d} | Water: {water_cnt} | Feed: {feed_cnt} | Fert: {fert_cnt} | Farmer: {farmer_act} | Hand0: {hands_act[0] if hands_act else []}")
