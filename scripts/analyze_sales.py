import json

with open("replays/top_competitors/episode-93954943-replay.json") as f:
    abra_data = json.load(f)

with open(".scratch/our_run.json") as f:
    our_data = json.load(f)

def analyze_sales(data, name):
    total_sales = {}
    total_revenue = 0.0
    for s_idx in range(len(data["steps"])):
        act = data["steps"][s_idx][0].get("action", {})
        m = act.get("market", [])
        obs = data["steps"][s_idx][0]["observation"]
        prices = obs["market"]["prices"]
        for order in m:
            if order and order[0] == "SELL":
                prod = order[1]
                qty = order[2] if len(order) > 2 else 1
                rev = qty * prices.get(prod, 0)
                total_sales[prod] = total_sales.get(prod, 0) + qty
                total_revenue += rev
    print(f"=== {name} Sales Summary ===")
    for prod, qty in sorted(total_sales.items()):
        print(f"  {prod:<12}: {qty:>4} units")
    print(f"  Total Estimated Revenue: ${total_revenue:,.2f}")

analyze_sales(abra_data, "Abracadabra ($155k)")
print()
analyze_sales(our_data, "Our Run ($55k)")
