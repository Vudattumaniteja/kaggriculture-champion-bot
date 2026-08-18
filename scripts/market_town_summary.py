import json
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93943624.json", "r", encoding="utf-8") as f:
    replay = json.load(f)

steps = replay["steps"]

# Town shop unlock days
unlocked_shops_per_day = {}
for day in range(30):
    t = day * 24
    town = steps[t][0]["observation"].get("town", {})
    unlocked_shops_per_day[day] = town.get("unlocked_shops", [])

print("=== TOWN SHOP UNLOCK SCHEDULE ===")
prev_shops = []
for day, shops in unlocked_shops_per_day.items():
    if shops != prev_shops:
        new_shops = [s for s in shops if s not in prev_shops]
        print(f"Day {day:02d}: Newly Unlocked -> {new_shops} | Total Active: {shops}")
        prev_shops = shops

# Commodity price changes
prices_summary = {}
for p in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]:
    history = [steps[t][0]["observation"]["market"]["prices"][p] for t in range(len(steps))]
    prices_summary[p] = {
        "start": history[0],
        "min": min(history),
        "max": max(history),
        "end": history[-1],
        "avg": sum(history) / len(history),
    }

print("\n=== COMMODITY PRICE SUMMARY ===")
for p, s in prices_summary.items():
    print(f"{p:12s} | Start: ${s['start']:3d} | Min: ${s['min']:3d} | Max: ${s['max']:3d} | End: ${s['end']:3d} | Avg: ${s['avg']:.1f}")
