import json
import os
from collections import defaultdict, Counter

REPLAY_PATH = r"C:\Users\Manit\Downloads\93943624.json"

with open(REPLAY_PATH, "r", encoding="utf-8") as f:
    replay = json.load(f)

steps = replay["steps"]
total_steps = len(steps)

print(f"Total Steps: {total_steps}")
print(f"Rewards: {replay.get('rewards')}")
print(f"Info: {replay.get('info')}")

# Trace everything
p0_money = []
p1_money = []
p0_shed = []
p1_shed = []
p0_seeds = []
p1_seeds = []
diff_money = []

for t in range(total_steps):
    s0 = steps[t][0]
    s1 = steps[t][1]
    obs0 = s0["observation"]
    obs1 = s1["observation"]
    
    m0 = obs0["farms"][0]["money"]
    m1 = obs1["farms"][1]["money"]
    p0_money.append(m0)
    p1_money.append(m1)
    
    sh0 = obs0["private"]["shed"]
    sh1 = obs1["private"]["shed"]
    p0_shed.append(sh0)
    p1_shed.append(sh1)
    
    sd0 = obs0["private"]["seeds"]
    sd1 = obs1["private"]["seeds"]
    p0_seeds.append(sd0)
    p1_seeds.append(sd1)
    
    diff = m0 - m1
    if len(diff_money) == 0 or diff_money[-1][1] != diff:
        diff_money.append((t, diff, m0, m1, t // 24, t % 24))

print("\n=== MONEY DIFFERENCE TIMELINE ===")
for t, diff, m0, m1, day, hr in diff_money:
    print(f"Turn {t:3d} (Day {day:02d} Hr {hr:02d}) | P0: ${m0:7.1f} | P1: ${m1:7.1f} | P0-P1: {diff:+7.1f}")

# Check differences between P0 and P1 actions and private states around the divergence points
print("\n=== DETAILED DIVERGENCE ANALYSIS ===")
for t in range(total_steps):
    s0 = steps[t][0]
    s1 = steps[t][1]
    act0 = s0.get("action", {})
    act1 = s1.get("action", {})
    
    if act0 != act1:
        day = t // 24
        hr = t % 24
        print(f"\n--- Turn {t} (Day {day:02d} Hr {hr:02d}) ---")
        print(f"P0 Action: {act0}")
        print(f"P1 Action: {act1}")
        print(f"P0 Money: {p0_money[t]}, Seeds: {p0_seeds[t]}, Shed: {p0_shed[t]}")
        print(f"P1 Money: {p1_money[t]}, Seeds: {p1_seeds[t]}, Shed: {p1_shed[t]}")
