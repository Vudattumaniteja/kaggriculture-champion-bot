import json
import os
from collections import defaultdict, Counter

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

# Track town shops unlocked over time
town_shops = {}
market_prices = defaultdict(dict)

for t in range(0, len(steps), 24):
    obs = steps[t][0]["observation"]
    day = obs.get("day", t // 24)
    shops = obs.get("town", {}).get("shops", [])
    prices = obs.get("market", {}).get("prices", {})
    town_shops[day] = shops
    market_prices[day] = prices

print("=== TOWN SHOPS UNLOCKED PER DAY ===")
for day, shops in town_shops.items():
    print(f"Day {day:02d}: {shops}")

print("\n=== MARKET PRICES PER DAY ===")
products = ["CARROT", "WHEAT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
header = f"{'Day':<4} | " + " | ".join([f"{p[:5]:<5}" for p in products])
print(header)
print("-" * len(header))
for day in range(30):
    p_dict = market_prices[day]
    row = f"{day:<4} | " + " | ".join([f"{p_dict.get(p, 0):<5}" for p in products])
    print(row)
