import json

with open(r"C:\Users\Manit\Downloads\93943624.json", "r", encoding="utf-8") as f:
    replay = json.load(f)

steps = replay["steps"]

print("=== ALL SALES TRANSACTIONS (P0 vs P1) ===")
sales = []
for t, step in enumerate(steps):
    for p_idx in [0, 1]:
        act = step[p_idx].get("action", {})
        mkt = act.get("market", [])
        for o in mkt:
            if isinstance(o, list) and len(o) > 0 and o[0] == "SELL":
                item = o[1]
                qty = o[2] if len(o) > 2 else 1
                price = step[p_idx]["observation"]["market"]["prices"].get(item, 0)
                sales.append((t, t // 24, t % 24, f"P{p_idx}", item, qty, price, qty * price))

for t, day, hr, p, item, qty, price, total in sales:
    print(f"Turn {t:3d} (Day {day:02d} Hr {hr:02d}) | {p} SOLD {qty:2d}x {item:11s} @ ${price:3d} each -> Total = ${total:5d}")
