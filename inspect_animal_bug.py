import json

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

print("=== P0 ACTIONS TURNS 310 to 360 ===")
for t in range(310, 360):
    s = steps[t][0]
    obs = s["observation"]
    act = s.get("action", {})
    day = obs["day"]
    hour = obs["hour"]
    farm = obs["farms"][0]
    farmer_pos = farm.get("farmer")
    hands_pos = farm.get("hands", [])
    shed = obs["private"]["shed"]
    active_shed = {k: v for k, v in shed.items() if v > 0}
    money = farm["money"]
    
    farmer_act = act.get("farmer")
    hands_act = act.get("hands")
    mkt_act = act.get("market")
    
    print(f"T{t:03d} (D{day:02d} H{hour:02d}) Cash=${money:<6.0f} Pos={farmer_pos} | FarmerAct: {farmer_act} | Mkt: {mkt_act} | Shed: {active_shed}")
