"""
Exhaustive Forensic Analysis & Report Generator for Top 10 Kaggriculture Competitors
"""

import os
import sys
import json
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

TOP_TEAMS = [
    {"rank": 1, "team_id": 16677252, "name": "カワシギ", "public_score": 3214.0},
    {"rank": 2, "team_id": 16719123, "name": "Thomas Tschinkel", "public_score": 3118.9},
    {"rank": 3, "team_id": 16696304, "name": "Utkarsh #2", "public_score": 3020.7},
    {"rank": 4, "team_id": 16703045, "name": "ReCurSiON", "public_score": 2996.4},
    {"rank": 5, "team_id": 16715733, "name": "peikopon", "public_score": 2987.5},
    {"rank": 6, "team_id": 16668721, "name": "Efe Can Celiksoy", "public_score": 2968.6},
    {"rank": 7, "team_id": 16662121, "name": "One-For-All", "public_score": 2968.1},
    {"rank": 8, "team_id": 16712189, "name": "Kostiantyn Isaienkov", "public_score": 2962.5},
    {"rank": 9, "team_id": 16667020, "name": "SCLim2022080004", "public_score": 2959.3},
    {"rank": 10, "team_id": 16723379, "name": "Matteo123383iend", "public_score": 2957.0}
]

with open(".scratch/parsed_top10_replays.json", "r", encoding="utf-8") as f:
    episodes = json.load(f)

# Group episodes by competitor rank
competitor_matches = defaultdict(list)

for ep in episodes:
    meta = ep.get("meta", {})
    t_rank = meta.get("team_rank")
    t_name = meta.get("team_name")
    
    # Check player 0 and player 1
    for p_idx in ["0", "1"]:
        p = ep["players"][p_idx]
        p_name = p["name"]
        opp_idx = "1" if p_idx == "0" else "0"
        opp = ep["players"][opp_idx]
        
        # Match player to a top 10 team
        matched = None
        for t in TOP_TEAMS:
            if t["name"] == p_name or (t_rank == t["rank"] and t_name == p_name):
                matched = t
                break
        if not matched and t_rank:
            # Check if this match was downloaded for target rank and player index matches
            if (t_name and t_name in p_name) or (p_name == t_name):
                matched = next((t for t in TOP_TEAMS if t["rank"] == t_rank), None)
                
        if matched:
            record = {
                "rank": matched["rank"],
                "team_name": matched["name"],
                "episode_id": ep["episode_id"],
                "seed": ep["seed"],
                "player_idx": int(p_idx),
                "opponent_name": opp["name"],
                "opponent_reward": opp["final_reward"],
                "final_reward": p["final_reward"],
                "peak_cash": p["peak_cash"],
                "daily_cash": {int(k): v for k, v in p["daily_cash"].items()},
                "quadrant_unlocks": p["quadrant_unlocks"],
                "daily_hires": p["daily_hires"],
                "total_hires": p["total_hires"],
                "crops_planted": p["crops_planted"],
                "daily_planted": {int(k): v for k, v in p["daily_planted"].items()},
                "seeds_bought": p["seeds_bought"],
                "animals_bought": p["animals_bought"],
                "feed_bought": p["feed_bought"],
                "structures_built": p["structures_built"],
                "market_sales": p["market_sales"],
                "town_orders": p["town_orders_fulfilled"],
                "action_counts": p["action_counts"],
                "hands_action_counts": p["hands_action_counts"],
                "deadweight_turns": p["deadweight_turns"]
            }
            competitor_matches[matched["rank"]].append(record)

print(f"Aggregated matches for {len(competitor_matches)} top teams.")
for t in TOP_TEAMS:
    r = t["rank"]
    print(f"Rank {r:2d} ({t['name']:20s}): {len(competitor_matches[r])} match instances")

# Save structured summary to assist deep reporting
with open(".scratch/competitor_match_aggregates.json", "w", encoding="utf-8") as f:
    json.dump(competitor_matches, f, indent=2)

