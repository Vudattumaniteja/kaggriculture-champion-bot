"""
Deep Inspector for Team Strategies, Turn 1 openings, and daily dynamics
"""

import sys
import json
from collections import defaultdict, Counter
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(".scratch/competitor_match_aggregates.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 90)
print("DEEP STRATEGY AUDIT ACROSS ALL 10 LEADERBOARD RANKS")
print("=" * 90)

for rank_str in sorted(data.keys(), key=lambda x: int(x)):
    rank = int(rank_str)
    matches = data[rank_str]
    t_name = matches[0]["team_name"]
    
    print(f"\n{'='*40} RANK {rank:2d}: {t_name} {'='*40}")
    
    # Financial trajectory averages
    days = list(range(30))
    daily_cash_matrix = []
    for m in matches:
        row = []
        for d in days:
            val = m["daily_cash"].get(str(d), m["daily_cash"].get(d, 0))
            row.append(val)
        daily_cash_matrix.append(row)
    daily_cash_avg = np.mean(daily_cash_matrix, axis=0)
    
    # Unlocks
    ne_turns = [m["quadrant_unlocks"]["NE"]["turn"] for m in matches if "NE" in m["quadrant_unlocks"]]
    sw_turns = [m["quadrant_unlocks"]["SW"]["turn"] for m in matches if "SW" in m["quadrant_unlocks"]]
    se_turns = [m["quadrant_unlocks"]["SE"]["turn"] for m in matches if "SE" in m["quadrant_unlocks"]]
    
    print(f"Match Count: {len(matches)}")
    print(f"Final Score: Mean=${np.mean([m['final_reward'] for m in matches]):,.2f} (Max=${np.max([m['final_reward'] for m in matches]):,.2f})")
    print(f"Quadrant Unlocks (Turn/Day):")
    print(f"  NE ($1,000): Turn {np.mean(ne_turns):.0f} (Day {np.mean(ne_turns)/24:.1f})" if ne_turns else "  NE: None")
    print(f"  SW ($2,000): Turn {np.mean(sw_turns):.0f} (Day {np.mean(sw_turns)/24:.1f})" if sw_turns else "  SW: None")
    print(f"  SE ($4,000): Turn {np.mean(se_turns):.0f} (Day {np.mean(se_turns)/24:.1f})" if se_turns else "  SE: None")
    
    # Crop planting by day
    crop_totals = Counter()
    for m in matches:
        for crop, cnt in m["crops_planted"].items():
            crop_totals[crop] += cnt / len(matches)
    print(f"Crop Portfolio: {dict(crop_totals)}")
    
    # Livestock & Feed
    anim_totals = Counter()
    feed_totals = Counter()
    struct_totals = Counter()
    for m in matches:
        for a, cnt in m["animals_bought"].items():
            anim_totals[a] += cnt / len(matches)
        for f, cnt in m["feed_bought"].items():
            feed_totals[f] += cnt / len(matches)
        for s, cnt in m["structures_built"].items():
            struct_totals[s] += cnt / len(matches)
    print(f"Livestock: {dict(anim_totals)} | Feed: {dict(feed_totals)} | Structures: {dict(struct_totals)}")
    
    # Market sales
    sales_totals = Counter()
    for m in matches:
        for it, cnt in m["market_sales"].items():
            sales_totals[it] += cnt / len(matches)
    print(f"Market Sales: {dict(sales_totals)}")
    
    # Cash trajectory checkpoints
    checkpoints = [0, 1, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 29]
    cp_strs = [f"D{d}: ${daily_cash_avg[d]:,.0f}" for d in checkpoints]
    print(f"Cash Trajectory: {' -> '.join(cp_strs)}")
