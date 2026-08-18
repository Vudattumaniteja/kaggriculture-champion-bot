"""
Top 10 Competitors Replay Retrieval and Comprehensive Forensic Engine
"""

import os
import sys
import json
import glob
import subprocess
import time
from collections import defaultdict, Counter
import pandas as pd

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

REPLAYS_DIR = "replays/top_competitors"
os.makedirs(REPLAYS_DIR, exist_ok=True)

def run_cmd(cmd):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command error ({' '.join(cmd)}): {e.stderr}")
        return None

def parse_json_from_cli(text):
    if not text:
        return []
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end+1])
    try:
        return json.loads(text)
    except Exception:
        return []

def fetch_and_download():
    episodes_to_download = []
    team_episode_map = defaultdict(list)
    
    for team in TOP_TEAMS:
        t_id = team["team_id"]
        t_rank = team["rank"]
        t_name = team["name"]
        print(f"\n[+] Fetching submissions for Rank {t_rank}: {t_name} (Team ID: {t_id})")
        out = run_cmd(["kaggle", "competitions", "team-submissions", str(t_id), "--format", "json"])
        if not out:
            continue
        subs = parse_json_from_cli(out)
        print(f"    Found {len(subs)} submissions.")
        
        # Sort submissions by date or score
        # Let's inspect each submission to get recent completed public episodes
        count_for_team = 0
        for sub in subs:
            sub_id = sub["id"]
            sub_score = sub.get("publicScore", "N/A")
            print(f"    Fetching episodes for Sub {sub_id} (Score: {sub_score})...")
            ep_out = run_cmd(["kaggle", "competitions", "episodes", str(sub_id), "--format", "json"])
            if not ep_out:
                continue
            eps = parse_json_from_cli(ep_out)
            # Filter for completed episodes
            completed_eps = [e for e in eps if e.get("state") == "EpisodeState.COMPLETED" and e.get("type") in ["EpisodeType.EPISODE_TYPE_PUBLIC", "EpisodeType.EPISODE_TYPE_VALIDATION"]]
            print(f"      Found {len(completed_eps)} completed episodes.")
            
            for ep in completed_eps:
                ep_id = ep["id"]
                if ep_id not in [e["ep_id"] for e in episodes_to_download]:
                    episodes_to_download.append({
                        "team_rank": t_rank,
                        "team_name": t_name,
                        "team_id": t_id,
                        "sub_id": sub_id,
                        "ep_id": ep_id,
                        "type": ep.get("type")
                    })
                    team_episode_map[t_id].append(ep_id)
                    count_for_team += 1
                if count_for_team >= 3: # 2-3 episodes per team
                    break
            if count_for_team >= 3:
                break

    print(f"\n[+] Total unique episodes targeted: {len(episodes_to_download)}")
    
    # Download episodes
    for item in episodes_to_download:
        ep_id = item["ep_id"]
        json_target = os.path.join(REPLAYS_DIR, f"{ep_id}.json")
        if os.path.exists(json_target) and os.path.getsize(json_target) > 1000:
            print(f"  Episode {ep_id} already exists locally.")
            continue
        print(f"  Downloading replay {ep_id} for Rank {item['team_rank']} ({item['team_name']})...")
        run_cmd(["kaggle", "competitions", "replay", str(ep_id), "-p", REPLAYS_DIR])
        time.sleep(0.5)

    with open(os.path.join(REPLAYS_DIR, "team_episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"teams": TOP_TEAMS, "downloads": episodes_to_download}, f, indent=2)

if __name__ == "__main__":
    fetch_and_download()
