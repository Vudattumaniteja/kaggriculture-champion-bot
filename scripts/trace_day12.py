import json

with open(".scratch/our_run.json") as f:
    data = json.load(f)

steps = data["steps"]

# Trace Day 12 (steps 12*24 to 13*24)
for s in range(12*24, 13*24):
    obs = steps[s][0]["observation"]
    day = obs["day"]
    hour = obs["hour"]
    f0 = obs["farms"][0]
    shed = obs["private"]["shed"]
    act = steps[s][0].get("action", {})
    farmer_act = act.get("farmer", [])
    hands_act = act.get("hands", [])
    print(f"D{day:02d} H{hour:02d} | Money: ${f0['money']:<6.0f} | Shed: {shed} | Farmer: {farmer_act} | Hand0: {hands_act[0] if hands_act else []}")
