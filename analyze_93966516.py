import json
from collections import defaultdict

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

print("=" * 60)
print("METADATA")
print("=" * 60)
print("Episode ID:", replay.get('id'))
print("Info:", replay.get('info'))
print("Configuration:", replay.get('configuration'))
print("Final Rewards:", replay.get('rewards'))
print("Statuses:", replay.get('statuses'))
print("Total Steps:", len(replay['steps']))

steps = replay['steps']

# Let's inspect final step for both players
for p in range(2):
    obs = steps[-1][p].get('observation', {})
    farms = obs.get('farms', [])
    priv = obs.get('private', {})
    print(f"\n--- Player {p} Final State (Turn 719, Day {obs.get('day')}, Hour {obs.get('hour')}) ---")
    print(f"Reward: {steps[-1][p].get('reward')}")
    if p < len(farms):
        farm = farms[p]
        print(f"Money: {farm.get('money')}")
        print(f"Unlocked Quadrants: {farm.get('unlocked_quadrants')}")
        print(f"Farmer: {farm.get('farmer')}")
        print(f"Hands: {farm.get('hands')}")
        print(f"Hires Today: {farm.get('hires_today')}")
    print(f"Shed: {priv.get('shed')}")
    print(f"Seeds: {priv.get('seeds')}")
    print(f"Inventories: {priv.get('inventories')}")

# Day-by-day money tracking
print("\n" + "=" * 60)
print("DAY-BY-DAY MONEY COMPARISON (at Hour 0 of each day)")
print("=" * 60)
print(f"{'Day':<5} | {'P0 Money':<12} | {'P1 Money':<12} | {'Diff (P0 - P1)':<15}")
print("-" * 50)

for d in range(30):
    step_idx = d * 24
    if step_idx < len(steps):
        obs0 = steps[step_idx][0].get('observation', {})
        obs1 = steps[step_idx][1].get('observation', {})
        f0 = obs0.get('farms', [{}, {}])[0]
        f1 = obs1.get('farms', [{}, {}])[1]
        m0 = f0.get('money', 0.0)
        m1 = f1.get('money', 0.0)
        print(f"{d:<5} | {m0:<12.1f} | {m1:<12.1f} | {m0 - m1:<+15.1f}")

# Final turn
obs0 = steps[-1][0].get('observation', {})
obs1 = steps[-1][1].get('observation', {})
m0 = obs0.get('farms', [{}, {}])[0].get('money', 0.0)
m1 = obs1.get('farms', [{}, {}])[1].get('money', 0.0)
print(f"{'Final':<5} | {m0:<12.1f} | {m1:<12.1f} | {m0 - m1:<+15.1f}")
