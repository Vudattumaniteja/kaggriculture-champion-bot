import json

REPLAY_PATH = r"C:\Users\Manit\Downloads\93943624.json"

with open(REPLAY_PATH, "r") as f:
    replay = json.load(f)

steps = replay["steps"]

for step_idx, step in enumerate(steps):
    s0, s1 = step[0], step[1]
    obs0 = s0.get("observation", {})
    obs1 = s1.get("observation", {})
    farms0 = obs0.get("farms", [])
    m0 = farms0[0]["money"] if len(farms0) > 0 else 0
    m1 = farms0[1]["money"] if len(farms0) > 1 else 0
    
    if m0 != m1:
        print(f"First money diff at Step {step_idx} (Day {obs0.get('day')}, Hr {obs0.get('hour')}): P0=${m0}, P1=${m1}, diff=${m1-m0}")
        print("Previous step actions (Step", step_idx - 1, "):")
        print("  P0 Act:", steps[step_idx-1][0].get("action"))
        print("  P1 Act:", steps[step_idx-1][1].get("action"))
        print("  P0 Obs market:", steps[step_idx-1][0].get("observation", {}).get("market"))
        print("  P0 Obs shed:", steps[step_idx-1][0].get("observation", {}).get("private", {}).get("shed"))
        print("  P1 Obs shed:", steps[step_idx-1][1].get("observation", {}).get("private", {}).get("shed"))
        break

# Let's also check all steps where money changed for either player
money_events = []
prev_m0, prev_m1 = 3000.0, 3000.0
for step_idx, step in enumerate(steps):
    s0 = step[0]
    obs0 = s0.get("observation", {})
    farms0 = obs0.get("farms", [])
    m0 = farms0[0]["money"] if len(farms0) > 0 else 0
    m1 = farms0[1]["money"] if len(farms0) > 1 else 0
    
    if m0 != prev_m0 or m1 != prev_m1:
        money_events.append((step_idx, obs0.get("day"), obs0.get("hour"), prev_m0, m0, prev_m1, m1, steps[step_idx-1][0].get("action"), steps[step_idx-1][1].get("action")))
        prev_m0, prev_m1 = m0, m1

print("\n--- ALL MONEY CHANGE EVENTS ---")
for e in money_events:
    step_idx, day, hr, pm0, m0, pm1, m1, act0, act1 = e
    print(f"Step {step_idx} (D{day:02d}:H{hr:02d}): P0 ${pm0:.0f} -> ${m0:.0f} (d={m0-pm0:.0f}) | P1 ${pm1:.0f} -> ${m1:.0f} (d={m1-pm1:.0f})")
    if (m0-pm0) != (m1-pm1):
        print(f"   *** DIVERGENCE: P0 delta = {m0-pm0}, P1 delta = {m1-pm1} ***")
        print(f"   Act0 market: {act0.get('market')}")
        print(f"   Act1 market: {act1.get('market')}")
