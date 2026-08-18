import json

with open(r"C:\Users\Manit\Downloads\93943624.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

print("=== SEED BUYING COMPARISON ===")
for step_idx in range(len(steps)):
    s0 = steps[step_idx][0]
    s1 = steps[step_idx][1]
    
    mkt0 = s0.get("action", {}).get("market", [])
    mkt1 = s1.get("action", {}).get("market", [])
    
    seed0 = [o for o in mkt0 if isinstance(o, list) and len(o) > 0 and o[0] == "BUY_SEED"]
    seed1 = [o for o in mkt1 if isinstance(o, list) and len(o) > 0 and o[0] == "BUY_SEED"]
    
    if seed0 != seed1:
        print(f"Step {step_idx:03d} (D{s0['observation']['day']:02d}:H{s0['observation']['hour']:02d}):")
        print(f"  P0 seeds bought: {seed0}, Money before: ${s0['observation']['farms'][0]['money']}")
        print(f"  P1 seeds bought: {seed1}, Money before: ${s1['observation']['farms'][1]['money']}")
        print(f"  P0 private seeds: {s0['observation']['private']['seeds']}")
        print(f"  P1 private seeds: {s1['observation']['private']['seeds']}")
