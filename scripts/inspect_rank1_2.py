import sys, json, numpy as np
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(".scratch/competitor_match_aggregates.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for rank in [1, 2]:
    matches = data[str(rank)]
    t_name = matches[0]["team_name"]
    print(f"=== RANK {rank}: {t_name} ===")
    print(f"Match Count: {len(matches)}")
    print(f"Final Rewards: {[m['final_reward'] for m in matches]}")
    print(f"Final Score Mean: ${np.mean([m['final_reward'] for m in matches]):,.2f}")
    
    days = list(range(30))
    daily_cash_matrix = []
    for m in matches:
        row = [m["daily_cash"].get(str(d), m["daily_cash"].get(d, 0)) for d in days]
        daily_cash_matrix.append(row)
    daily_cash_avg = np.mean(daily_cash_matrix, axis=0)
    checkpoints = [0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 29]
    cp_strs = [f"D{d}: ${daily_cash_avg[d]:,.0f}" for d in checkpoints]
    print("Cash Trajectory:", " -> ".join(cp_strs))
    
    crop_totals = Counter()
    for m in matches:
        for crop, cnt in m["crops_planted"].items():
            crop_totals[crop] += cnt / len(matches)
    print("Crop Portfolio:", dict(crop_totals))
    
    anim_totals = Counter()
    feed_totals = Counter()
    struct_totals = Counter()
    for m in matches:
        for a, cnt in m["animals_bought"].items():
            anim_totals[a] += cnt / len(matches)
        for feed, cnt in m["feed_bought"].items():
            feed_totals[feed] += cnt / len(matches)
        for s, cnt in m["structures_built"].items():
            struct_totals[s] += cnt / len(matches)
    print("Livestock:", dict(anim_totals), "| Feed:", dict(feed_totals), "| Structures:", dict(struct_totals))
    
    sales_totals = Counter()
    for m in matches:
        for it, cnt in m["market_sales"].items():
            sales_totals[it] += cnt / len(matches)
    print("Market Sales:", dict(sales_totals))
