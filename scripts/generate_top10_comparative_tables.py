"""
Generate Top 10 Competitors Comparative Analytics & Markdown Tables
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

with open(".scratch/parsed_top10_replays.json", "r", encoding="utf-8") as f:
    episodes = json.load(f)

# Group episodes by competitor team
# Note: In each episode, player 0 or player 1 might be the team under study
team_data = defaultdict(list)

TOP_TEAMS = [
    {"rank": 1, "team_id": 16677252, "name": "カワシギ"},
    {"rank": 2, "team_id": 16719123, "name": "Thomas Tschinkel"},
    {"rank": 3, "team_id": 16696304, "name": "Utkarsh #2"},
    {"rank": 4, "team_id": 16703045, "name": "ReCurSiON"},
    {"rank": 5, "team_id": 16715733, "name": "peikopon"},
    {"rank": 6, "team_id": 16668721, "name": "Efe Can Celiksoy"},
    {"rank": 7, "team_id": 16662121, "name": "One-For-All"},
    {"rank": 8, "team_id": 16712189, "name": "Kostiantyn Isaienkov"},
    {"rank": 9, "team_id": 16667020, "name": "SCLim2022080004"},
    {"rank": 10, "team_id": 16723379, "name": "Matteo123383iend"}
]

team_id_map = {t["team_id"]: t for t in TOP_TEAMS}
team_name_map = {t["name"]: t for t in TOP_TEAMS}

# Process each episode
for ep in episodes:
    meta = ep.get("meta", {})
    target_team_id = meta.get("team_id")
    target_rank = meta.get("team_rank")
    target_name = meta.get("team_name")
    
    # Identify which player index corresponds to target team
    for p_idx, p in ep["players"].items():
        p_idx = int(p_idx)
        p_name = p["name"]
        
        # Check if this player is the target team or matches team names
        is_target = False
        matched_rank = None
        matched_team = None
        
        if target_team_id and target_name and (p_name == target_name or target_name in p_name):
            is_target = True
            matched_rank = target_rank
            matched_team = target_name
        else:
            for t in TOP_TEAMS:
                if t["name"] == p_name or t["name"] in p_name:
                    is_target = True
                    matched_rank = t["rank"]
                    matched_team = t["name"]
                    break
                    
        # If this episode was downloaded for a specific team, and player name matches or is player 0/1
        if not is_target and target_team_id:
            # Let's check player names
            # Sometimes Kaggle agent name in info is username or team name
            # Let's check metadata
            pass
            
        record = {
            "episode_id": ep["episode_id"],
            "seed": ep["seed"],
            "player_idx": p_idx,
            "player_name": p_name,
            "is_target_competitor": is_target,
            "matched_rank": matched_rank,
            "matched_team": matched_team,
            "target_meta": meta,
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
        
        if matched_rank:
            team_data[matched_rank].append(record)
        elif target_rank and (p_idx == 0): # fallback if target rank is known
            team_data[target_rank].append(record)

print("=== Competitors Replay Inventory Summary ===")
for t in TOP_TEAMS:
    r = t["rank"]
    recs = team_data[r]
    print(f"Rank {r:2d} | {t['name']:25s} | Matches analyzed: {len(recs)}")

