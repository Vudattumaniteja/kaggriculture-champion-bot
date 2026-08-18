"""
Comprehensive Turn-by-Turn Forensic Analyzer for Top 10 Kaggriculture Competitors
"""

import os
import sys
import json
import glob
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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

def load_manifest():
    p = "replays/top_competitors/team_episode_manifest.json"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"downloads": []}

def parse_replay_forensics(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    ep_id = info.get("EpisodeId") or data.get("id", "Unknown")
    seed = info.get("seed", "Unknown")
    team_names = info.get("TeamNames") or [a.get("Name") for a in info.get("Agents", [])] or ["Player 0", "Player 1"]
    
    steps = data.get("steps", [])
    num_steps = len(steps)
    
    results = {
        "episode_id": ep_id,
        "seed": seed,
        "total_steps": num_steps,
        "players": {}
    }
    
    for p_idx in [0, 1]:
        p_name = team_names[p_idx] if p_idx < len(team_names) else f"Player {p_idx}"
        results["players"][p_idx] = {
            "index": p_idx,
            "name": p_name,
            "final_reward": 0.0,
            "peak_cash": 0.0,
            "daily_cash": {}, # day: cash at turn 23 of day
            "quadrant_unlocks": {}, # quad: {turn, day}
            "daily_hires": [0] * 30, # hires per day
            "total_hires": 0,
            "crops_planted": Counter(),
            "daily_planted": defaultdict(lambda: Counter()), # day -> crop -> count
            "seeds_bought": Counter(),
            "animals_bought": Counter(),
            "feed_bought": Counter(),
            "structures_built": Counter(),
            "market_sales": Counter(), # item -> units sold
            "market_sales_revenue": 0.0,
            "town_orders_fulfilled": Counter(),
            "town_revenue": 0.0,
            "action_counts": Counter(), # FARMER_PLANT, FARMER_WATER, etc.
            "hands_action_counts": Counter(),
            "deadweight_turns": 0,
            "cash_curve": []
        }

    prev_quads = [set(["NW"]), set(["NW"])]
    
    for t in range(num_steps):
        day = t // 24
        hour = t % 24
        step = steps[t]
        
        for p_idx in [0, 1]:
            if p_idx >= len(step):
                continue
            p_data = step[p_idx]
            obs = p_data.get("observation", {})
            act = p_data.get("action", {})
            reward = p_data.get("reward", 0.0)
            
            p_res = results["players"][p_idx]
            
            farms = obs.get("farms", [])
            if len(farms) > p_idx:
                farm = farms[p_idx]
                money = farm.get("money", 0.0)
                unlocked = set(farm.get("unlocked_quadrants", []))
                hires_today = farm.get("hires_today", 0)
                
                p_res["cash_curve"].append(money)
                p_res["peak_cash"] = max(p_res["peak_cash"], money)
                
                if hour == 23 or t == num_steps - 1:
                    p_res["daily_cash"][day] = money
                    
                if day < 30:
                    p_res["daily_hires"][day] = max(p_res["daily_hires"][day], hires_today)
                    
                new_q = unlocked - prev_quads[p_idx]
                for q in new_q:
                    p_res["quadrant_unlocks"][q] = {"turn": t, "day": day, "hour": hour}
                prev_quads[p_idx] = unlocked
                
            # Parse actions
            if isinstance(act, dict):
                # Farmer action
                farmer_act = act.get("farmer", [])
                if farmer_act and len(farmer_act) > 0:
                    f_op = farmer_act[0]
                    p_res["action_counts"][f_op] += 1
                    if f_op == "PLANT" and len(farmer_act) > 1:
                        crop = farmer_act[1]
                        p_res["crops_planted"][crop] += 1
                        p_res["daily_planted"][day][crop] += 1
                    elif f_op.startswith("BUILD_"):
                        p_res["structures_built"][f_op] += 1
                    elif f_op == "PASS":
                        p_res["deadweight_turns"] += 1
                else:
                    p_res["deadweight_turns"] += 1
                    
                # Hands actions
                hands_act = act.get("hands", [])
                for h in hands_act:
                    if h and len(h) > 0:
                        h_op = h[0]
                        p_res["hands_action_counts"][h_op] += 1
                        if h_op == "PLANT" and len(h) > 1:
                            crop = h[1]
                            p_res["crops_planted"][crop] += 1
                            p_res["daily_planted"][day][crop] += 1
                            
                # Market actions
                market_act = act.get("market", [])
                for m in market_act:
                    if not m or len(m) == 0:
                        continue
                    m_op = m[0]
                    if m_op == "BUY_SEED" and len(m) > 1:
                        crop = m[1]
                        qty = m[2] if len(m) > 2 else 1
                        p_res["seeds_bought"][crop] += qty
                    elif m_op == "BUY_ANIMAL" and len(m) > 1:
                        animal = m[1]
                        qty = m[2] if len(m) > 2 else 1
                        p_res["animals_bought"][animal] += qty
                    elif m_op == "BUY_PRODUCT" and len(m) > 1:
                        prod = m[1]
                        qty = m[2] if len(m) > 2 else 1
                        p_res["feed_bought"][prod] += qty
                    elif m_op == "SELL" and len(m) > 1:
                        item = m[1]
                        qty = m[2] if len(m) > 2 else 1
                        p_res["market_sales"][item] += qty
                    elif m_op.startswith("DELIVER_") or m_op.startswith("FULFILL_") or m_op == "TOWN_DELIVERY":
                        p_res["town_orders_fulfilled"][m_op] += 1

            if t == num_steps - 1:
                p_res["final_reward"] = reward
                p_res["total_hires"] = sum(p_res["daily_hires"])

    return results

def main():
    manifest = load_manifest()
    downloads = manifest.get("downloads", [])
    
    # Map episode to team rank and info
    ep_to_meta = {}
    for d in downloads:
        ep_to_meta[str(d["ep_id"])] = d
        
    replay_files = glob.glob("replays/top_competitors/*.json")
    valid_files = [f for f in replay_files if not f.endswith("-0.json") and not f.endswith("-1.json") and not "manifest" in f]
    
    print(f"[+] Found {len(valid_files)} replay files to analyze.")
    
    parsed_episodes = []
    for fpath in valid_files:
        try:
            res = parse_replay_forensics(fpath)
            # Find meta
            ep_id = str(res["episode_id"])
            meta = ep_to_meta.get(ep_id, {})
            res["meta"] = meta
            parsed_episodes.append(res)
        except Exception as e:
            print(f"Error parsing {fpath}: {e}")

    print(f"[+] Successfully parsed {len(parsed_episodes)} episodes.")
    
    # Save structured summary JSON for reporting
    with open(".scratch/parsed_top10_replays.json", "w", encoding="utf-8") as f:
        # Convert non-serializable objects
        clean_parsed = []
        for ep in parsed_episodes:
            ep_copy = {
                "episode_id": ep["episode_id"],
                "seed": ep["seed"],
                "total_steps": ep["total_steps"],
                "meta": ep.get("meta", {}),
                "players": {}
            }
            for p_idx, p in ep["players"].items():
                ep_copy["players"][p_idx] = {
                    "index": p["index"],
                    "name": p["name"],
                    "final_reward": p["final_reward"],
                    "peak_cash": p["peak_cash"],
                    "daily_cash": p["daily_cash"],
                    "quadrant_unlocks": p["quadrant_unlocks"],
                    "daily_hires": p["daily_hires"],
                    "total_hires": p["total_hires"],
                    "crops_planted": dict(p["crops_planted"]),
                    "daily_planted": {k: dict(v) for k, v in p["daily_planted"].items()},
                    "seeds_bought": dict(p["seeds_bought"]),
                    "animals_bought": dict(p["animals_bought"]),
                    "feed_bought": dict(p["feed_bought"]),
                    "structures_built": dict(p["structures_built"]),
                    "market_sales": dict(p["market_sales"]),
                    "town_orders_fulfilled": dict(p["town_orders_fulfilled"]),
                    "action_counts": dict(p["action_counts"]),
                    "hands_action_counts": dict(p["hands_action_counts"]),
                    "deadweight_turns": p["deadweight_turns"]
                }
            clean_parsed.append(ep_copy)
        json.dump(clean_parsed, f, indent=2)

    print("[+] Saved structured analysis to .scratch/parsed_top10_replays.json")

if __name__ == "__main__":
    main()
