"""
Detailed Statistical Extractor for Top 10 Report Generation
"""

import sys
import json
from collections import defaultdict, Counter
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(".scratch/competitor_match_aggregates.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("="*80)
print("TOP 10 COMPETITORS FORENSIC SUMMARY")
print("="*80)

for rank_str in sorted(data.keys(), key=lambda x: int(x)):
    rank = int(rank_str)
    matches = data[rank_str]
    t_name = matches[0]["team_name"]
    
    final_rewards = [m["final_reward"] for m in matches]
    peak_cashes = [m["peak_cash"] for m in matches]
    total_hires_list = [m["total_hires"] for m in matches]
    
    print(f"\nRANK {rank:2d}: {t_name} ({len(matches)} matches analyzed)")
    print(f"  Final Reward: Mean=${np.mean(final_rewards):,.0f} | Min=${np.min(final_rewards):,.0f} | Max=${np.max(final_rewards):,.0f}")
    print(f"  Peak Cash:    Mean=${np.mean(peak_cashes):,.0f} | Min=${np.min(peak_cashes):,.0f} | Max=${np.max(peak_cashes):,.0f}")
    print(f"  Total Hires:  Mean={np.mean(total_hires_list):.1f} farmhands")
    
    # Financial trajectory
    day_snapshots = [0, 1, 3, 5, 8, 10, 15, 20, 25, 29]
    cash_by_day = defaultdict(list)
    for m in matches:
        for d in day_snapshots:
            if str(d) in m["daily_cash"]:
                cash_by_day[d].append(m["daily_cash"][str(d)])
            elif d in m["daily_cash"]:
                cash_by_day[d].append(m["daily_cash"][d])
                
    cash_str = " | ".join([f"D{d}: ${np.mean(cash_by_day[d]):,.0f}" if cash_by_day[d] else f"D{d}: N/A" for d in day_snapshots])
    print(f"  Cash Curve:   {cash_str}")
    
    # Quadrant unlock timings
    ne_days = [m["quadrant_unlocks"]["NE"]["day"] for m in matches if "NE" in m["quadrant_unlocks"]]
    sw_days = [m["quadrant_unlocks"]["SW"]["day"] for m in matches if "SW" in m["quadrant_unlocks"]]
    se_days = [m["quadrant_unlocks"]["SE"]["day"] for m in matches if "SE" in m["quadrant_unlocks"]]
    print(f"  Unlocks:      NE: Day {np.mean(ne_days):.1f} | SW: Day {np.mean(sw_days):.1f} | SE: Day {np.mean(se_days):.1f}" if ne_days and sw_days and se_days else f"  Unlocks: {ne_days}, {sw_days}, {se_days}")
    
    # Crops planted
    all_crops = Counter()
    for m in matches:
        all_crops.update(m["crops_planted"])
    crop_str = ", ".join([f"{k}: {v/len(matches):.1f}" for k, v in all_crops.most_common()])
    print(f"  Avg Crops Planted: {crop_str}")
    
    # Animals & Structures
    all_animals = Counter()
    all_structs = Counter()
    all_feed = Counter()
    for m in matches:
        all_animals.update(m["animals_bought"])
        all_structs.update(m["structures_built"])
        all_feed.update(m["feed_bought"])
    print(f"  Avg Structures: {dict(all_structs)} | Avg Animals: {dict(all_animals)} | Avg Feed: {dict(all_feed)}")
    
    # Market sales vs Town Orders
    all_sales = Counter()
    all_town = Counter()
    for m in matches:
        all_sales.update(m["market_sales"])
        all_town.update(m["town_orders"])
    print(f"  Avg Market Sales: {dict(all_sales)}")
    print(f"  Avg Town Orders:  {dict(all_town)}")
    
    # Daily hiring curve
    hire_curves = np.array([m["daily_hires"] for m in matches])
    mean_hires = np.mean(hire_curves, axis=0)
    print(f"  Daily Hires (D0-D10): {[round(x, 1) for x in mean_hires[:11]]}")
    print(f"  Daily Hires (D11-D20): {[round(x, 1) for x in mean_hires[11:21]]}")
    print(f"  Daily Hires (D21-D29): {[round(x, 1) for x in mean_hires[21:]]}")
