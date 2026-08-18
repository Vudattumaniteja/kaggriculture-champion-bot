import json
from collections import defaultdict

REPLAY_PATH = r"C:\Users\Manit\Downloads\93943624.json"

with open(REPLAY_PATH, "r") as f:
    replay = json.load(f)

steps = replay["steps"]

# Let's find every single step where P0 and P1 differed in actions, observations, shed, money, etc.
diff_steps = []

for step_idx, step in enumerate(steps):
    s0, s1 = step[0], step[1]
    obs0, obs1 = s0.get("observation", {}), s1.get("observation", {})
    act0, act1 = s0.get("action", {}), s1.get("action", {})
    
    farms0 = obs0.get("farms", [])
    m0 = farms0[0]["money"] if len(farms0) > 0 else 0
    m1 = farms0[1]["money"] if len(farms0) > 1 else 0
    
    # Check action differences
    f0, h0, mkt0 = act0.get("farmer", []), act0.get("hands", []), act0.get("market", [])
    f1, h1, mkt1 = act1.get("farmer", []), act1.get("hands", []), act1.get("market", [])
    
    day = obs0.get("day", step_idx // 24)
    hour = obs0.get("hour", step_idx % 24)
    
    action_diff = (f0 != f1) or (h0 != h1) or (mkt0 != mkt1)
    money_diff = (m0 != m1)
    
    if action_diff or money_diff:
        diff_steps.append({
            "step": step_idx,
            "day": day,
            "hour": hour,
            "m0": m0,
            "m1": m1,
            "m_diff": m1 - m0,
            "act0": act0,
            "act1": act1,
            "farms0_farmer": farms0[0]["farmer"] if farms0 else None,
            "farms1_farmer": farms0[1]["farmer"] if len(farms0) > 1 else None,
            "hands0": farms0[0]["hands"] if farms0 else None,
            "hands1": farms0[1]["hands"] if len(farms0) > 1 else None,
        })

print(f"Total steps with differences: {len(diff_steps)}")
print("\nFirst 10 diff steps:")
for d in diff_steps[:10]:
    print(f"Step {d['step']} (Day {d['day']}, Hr {d['hour']}): m0={d['m0']}, m1={d['m1']}, diff={d['m_diff']}")
    print(f"  P0 Act: {d['act0']}")
    print(f"  P1 Act: {d['act1']}")
    print(f"  P0 Pos: farmer={d['farms0_farmer']}, hands={d['hands0']}")
    print(f"  P1 Pos: farmer={d['farms1_farmer']}, hands={d['hands1']}")

print("\nLast 10 diff steps:")
for d in diff_steps[-10:]:
    print(f"Step {d['step']} (Day {d['day']}, Hr {d['hour']}): m0={d['m0']}, m1={d['m1']}, diff={d['m_diff']}")
    print(f"  P0 Act: {d['act0']}")
    print(f"  P1 Act: {d['act1']}")
