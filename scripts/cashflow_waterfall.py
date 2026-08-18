import json
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93943624.json", "r", encoding="utf-8") as f:
    replay = json.load(f)

steps = replay["steps"]

# Track detailed cash balance breakdown
# Cash(t) = Cash(t-1) + Sales - Buys - Labor - Land - Structures (if cost applied via env)

def audit_player_cash_flow(p_idx):
    sales_by_item = defaultdict(lambda: {"qty": 0, "revenue": 0.0})
    buys_by_item = defaultdict(lambda: {"qty": 0, "cost": 0.0})
    labor_cost = 0.0
    land_cost = 0.0
    starting_cash = 3000.0
    
    for t in range(len(steps)):
        s = steps[t][p_idx]
        obs = s["observation"]
        act = s.get("action", {})
        prices = obs["market"]["prices"]
        day = t // 24
        
        # Check market actions executed at t
        # In environment, action submitted at t is executed, modifying state at t+1
        mkt = act.get("market", []) if isinstance(act, dict) else []
        for o in mkt:
            if not isinstance(o, list) or len(o) == 0:
                continue
            op = o[0]
            if op == "HIRE":
                qty = o[1] if len(o) > 1 else 1
                # Labor hiring cost
                # Let's verify hiring cost: Base cost 40 * (1 + 0.1 * total_hired_prior) or similar?
                pass
            elif op == "BUY_LAND":
                land_cost += 1000.0
            elif op == "BUY_ANIMAL":
                an = o[1]
                qty = o[2] if len(o) > 2 else 1
                c = {"GOOSE": 300, "COW": 400, "SHEEP": 500}[an] * qty
                buys_by_item[f"ANIMAL_{an}"]["qty"] += qty
                buys_by_item[f"ANIMAL_{an}"]["cost"] += c
            elif op == "BUY_SEED":
                sd = o[1]
                qty = o[2] if len(o) > 2 else 1
                c = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}[sd] * qty
                buys_by_item[f"SEED_{sd}"]["qty"] += qty
                buys_by_item[f"SEED_{sd}"]["cost"] += c
            elif op == "BUY_PRODUCT":
                pr = o[1]
                qty = o[2] if len(o) > 2 else 1
                c = prices[pr] * qty
                buys_by_item[f"PROD_{pr}"]["qty"] += qty
                buys_by_item[f"PROD_{pr}"]["cost"] += c
            elif op == "SELL":
                pr = o[1]
                qty = o[2] if len(o) > 2 else 1
                r = prices[pr] * qty
                sales_by_item[pr]["qty"] += qty
                sales_by_item[pr]["revenue"] += r

    total_rev = sum(v["revenue"] for v in sales_by_item.values())
    total_buy = sum(v["cost"] for v in buys_by_item.values())
    
    # Check actual end money
    end_money = steps[-1][p_idx]["observation"]["farms"][p_idx]["money"]
    total_labor = starting_cash + total_rev - total_buy - land_cost - end_money
    
    return {
        "starting_cash": starting_cash,
        "sales": dict(sales_by_item),
        "total_revenue": total_rev,
        "buys": dict(buys_by_item),
        "total_buys": total_buy,
        "land_cost": land_cost,
        "labor_cost": total_labor,
        "end_money": end_money,
    }

p0_cf = audit_player_cash_flow(0)
p1_cf = audit_player_cash_flow(1)

print("=== P0 CASH FLOW WATERFALL ===")
print(f"Starting Cash:    +${p0_cf['starting_cash']:8.2f}")
print(f"Total Revenue:    +${p0_cf['total_revenue']:8.2f}")
print(f"Total Purchases:  -${p0_cf['total_buys']:8.2f}")
print(f"Land Purchases:   -${p0_cf['land_cost']:8.2f}")
print(f"Labor Costs:      -${p0_cf['labor_cost']:8.2f}")
print(f"Calculated End:   =${p0_cf['starting_cash'] + p0_cf['total_revenue'] - p0_cf['total_buys'] - p0_cf['land_cost'] - p0_cf['labor_cost']:8.2f}")
print(f"Actual End Cash:  =${p0_cf['end_money']:8.2f}")

print("\n=== P1 CASH FLOW WATERFALL ===")
print(f"Starting Cash:    +${p1_cf['starting_cash']:8.2f}")
print(f"Total Revenue:    +${p1_cf['total_revenue']:8.2f}")
print(f"Total Purchases:  -${p1_cf['total_buys']:8.2f}")
print(f"Land Purchases:   -${p1_cf['land_cost']:8.2f}")
print(f"Labor Costs:      -${p1_cf['labor_cost']:8.2f}")
print(f"Calculated End:   =${p1_cf['starting_cash'] + p1_cf['total_revenue'] - p1_cf['total_buys'] - p1_cf['land_cost'] - p1_cf['labor_cost']:8.2f}")
print(f"Actual End Cash:  =${p1_cf['end_money']:8.2f}")
