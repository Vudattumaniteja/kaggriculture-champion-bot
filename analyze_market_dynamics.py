import json
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93943624.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

# Track prices at start, min, max, end, and dump events
prices = defaultdict(list)
sales_events = []

for step_idx, step in enumerate(steps):
    s0 = step[0]
    obs0 = s0["observation"]
    day = obs0.get("day", step_idx // 24)
    hour = obs0.get("hour", step_idx % 24)
    m_info = obs0.get("market", {})
    p = m_info.get("prices", {})
    for k, v in p.items():
        prices[k].append(v)
        
    mo0 = step[0].get("action", {}).get("market", [])
    mo1 = step[1].get("action", {}).get("market", [])
    
    for player, mo in [(0, mo0), (1, mo1)]:
        for o in mo:
            if isinstance(o, list) and len(o) > 0 and o[0] == "SELL":
                item = o[1]
                qty = o[2]
                sales_events.append((step_idx, day, hour, player, item, qty, p.get(item)))

print("=== COMMODITY PRICE DYNAMICS ===")
for k in sorted(prices.keys()):
    vals = prices[k]
    print(f"{k:12s}: Start=${vals[0]:3d} | Min=${min(vals):3d} | Max=${max(vals):3d} | Final=${vals[-1]:3d} | Avg=${sum(vals)/len(vals):.1f}")

print("\n=== MAJOR COMMODITY DUMP EVENTS ===")
# Group sales by (day, hour, item)
grouped_sales = defaultdict(lambda: {0: 0, 1: 0, "price": 0})
for step_idx, day, hr, player, item, qty, price in sales_events:
    key = (day, hr, item)
    grouped_sales[key][player] += qty
    grouped_sales[key]["price"] = price

for (day, hr, item), data in sorted(grouped_sales.items()):
    p0_q = data[0]
    p1_q = data[1]
    price = data["price"]
    print(f"Day {day:02d}:Hr {hr:02d} | Item: {item:11s} | Sold: P0={p0_q:2d}, P1={p1_q:2d} (Total={p0_q+p1_q:2d}) | Price: ${price}")
