import json

with open(".scratch/our_run.json") as f:
    data = json.load(f)

steps = data["steps"]

# Trace Day 11 (steps 11*24 to 12*24)
for s in range(11*24, 12*24):
    obs = steps[s][0]["observation"]
    day = obs["day"]
    hour = obs["hour"]
    f0 = obs["farms"][0]
    shed = obs["private"]["shed"]
    act = steps[s][0].get("action", {})
    m_orders = act.get("market", [])
    print(f"D{day:02d} H{hour:02d} | Money: ${f0['money']:<7.1f} | Shed Melons: {shed.get('MELON', 0)} | Market: {m_orders}")
