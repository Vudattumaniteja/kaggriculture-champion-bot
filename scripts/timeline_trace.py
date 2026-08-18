import json
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93943624.json", "r", encoding="utf-8") as f:
    replay = json.load(f)

steps = replay["steps"]

print("=== TIMELINE OF TURNS 400 TO 540 ===")
for t in range(400, 540):
    s0 = steps[t][0]
    s1 = steps[t][1]
    obs0 = s0["observation"]
    obs1 = s1["observation"]
    m0 = obs0["farms"][0]["money"]
    m1 = obs1["farms"][1]["money"]
    a0 = s0.get("action", {})
    a1 = s1.get("action", {})
    
    # print if actions differ or money changed
    if a0 != a1 or m0 != m1:
        mkt0 = a0.get("market", [])
        mkt1 = a1.get("market", [])
        print(f"Turn {t:3d} (D{t//24:02d}H{t%24:02d}) | P0 Cash: {m0:7.1f}, P1 Cash: {m1:7.1f} | Diff(P0-P1): {m0-m1:+7.1f}")
        if a0 != a1:
            print(f"   P0 act: farmer={a0.get('farmer')}, hands={a0.get('hands')}, market={mkt0}")
            print(f"   P1 act: farmer={a1.get('farmer')}, hands={a1.get('hands')}, market={mkt1}")
        if obs0["private"]["seeds"] != obs1["private"]["seeds"]:
            print(f"   P0 seeds: {obs0['private']['seeds']}")
            print(f"   P1 seeds: {obs1['private']['seeds']}")
