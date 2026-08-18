import json
import os
from collections import defaultdict

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

def audit_cash_flow(p_idx):
    inflows = defaultdict(float)
    outflows = defaultdict(float)
    
    # Track inventory and prices turn by turn
    sales_log = []
    purchase_log = []
    
    for t in range(len(steps) - 1):
        s_curr = steps[t][p_idx]
        s_next = steps[t+1][p_idx]
        
        m_curr = s_curr.get("observation", {}).get("farms", [])[p_idx].get("money", 0.0)
        m_next = s_next.get("observation", {}).get("farms", [])[p_idx].get("money", 0.0)
        
        delta = m_next - m_curr
        day = s_curr.get("observation", {}).get("day", t // 24)
        hour = s_curr.get("observation", {}).get("hour", t % 24)
        
        act = s_curr.get("action", {})
        
        if delta > 0:
            # Money increased -> sales revenue
            # Check market actions
            sold_items = []
            if isinstance(act, dict):
                for o in act.get("market", []):
                    if isinstance(o, list) and len(o) > 0 and o[0] == "SELL":
                        sold_items.append((o[1], o[2] if len(o) > 2 else 1))
            sales_log.append((t, day, hour, delta, sold_items))
            if len(sold_items) == 1:
                inflows[sold_items[0][0]] += delta
            else:
                inflows["MULTI_OR_OTHER"] += delta
        elif delta < 0:
            # Money decreased -> expenditure
            spent = -delta
            outflows["TOTAL_SPENT"] += spent
            # Classify
            mkt = act.get("market", []) if isinstance(act, dict) else []
            # Check if land, hires, seeds, animals, structures
            # Land: 1000, 2000, 4000
            # Labor: 1, 1, 2, 3, 5...
            # Seeds: Melon 80, Carrot 10, Tomato 50, Wheat 20, Strawberry 80
            # Animals: Cow 1000, Goose 100, Sheep 400
            # Structure: Coop 50, Pasture 100 (let's check unit action building cost)
            purchase_log.append((t, day, hour, spent, mkt, act.get("farmer", []), act.get("hands", [])))
            
    return {
        "p_idx": p_idx,
        "total_revenue": sum(x[3] for x in sales_log),
        "total_spent": sum(x[3] for x in purchase_log),
        "sales_log": sales_log,
        "inflows": dict(inflows),
        "purchase_log_count": len(purchase_log)
    }

p0_flow = audit_cash_flow(0)
p1_flow = audit_cash_flow(1)

print("=== P0 Cash Flow ===")
print("Total Revenue:", p0_flow["total_revenue"])
print("Total Spent:", p0_flow["total_spent"])
print("Inflows by item:", p0_flow["inflows"])

print("\n=== P1 Cash Flow ===")
print("Total Revenue:", p1_flow["total_revenue"])
print("Total Spent:", p1_flow["total_spent"])
print("Inflows by item:", p1_flow["inflows"])
