"""
Kaggriculture Top Competitors Replay Retrieval & Forensics Tool

This script automates:
1. Inspecting the Kaggriculture public leaderboard CSV for top teams.
2. Querying team submission IDs via Kaggle CLI / API (`kaggle competitions team-submissions <TEAM_ID>`).
3. Querying match episode IDs (`kaggle competitions episodes <SUBMISSION_ID>`).
4. Downloading match replay JSONs (`kaggle competitions replay <EPISODE_ID> -p replays/top_competitors/`).
5. Parsing and analyzing turn-by-turn plot moves, crop choices, labor allocations, and cash curves.
"""

import os
import sys
import json
import glob
import subprocess
import argparse
from collections import defaultdict, Counter
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_CSV_PATH = "kaggriculture-publicleaderboard-2026-08-17T12_08_50/kaggriculture-publicleaderboard-2026-08-17T12_08_50.csv"
REPLAYS_DIR = "replays/top_competitors"

def check_kaggle_auth():
    """Check if Kaggle credentials exist in environment or ~/.kaggle/."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    key = os.environ.get("KAGGLE_KEY")
    user = os.environ.get("KAGGLE_USERNAME")
    home_kaggle = os.path.expanduser("~/.kaggle")
    
    has_token_file = os.path.exists(os.path.join(home_kaggle, "access_token"))
    has_json_file = os.path.exists(os.path.join(home_kaggle, "kaggle.json"))
    
    if token or (key and user) or has_token_file or has_json_file:
        return True
    return False

def get_top_teams(csv_path=DEFAULT_CSV_PATH, top_n=10):
    """Load top teams from the public leaderboard CSV."""
    if not os.path.exists(csv_path):
        # Try finding any public leaderboard CSV in workspace
        matches = glob.glob("**/*publicleaderboard*.csv", recursive=True)
        if matches:
            csv_path = matches[0]
        else:
            raise FileNotFoundError(f"Could not locate leaderboard CSV at {csv_path}")
            
    df = pd.read_csv(csv_path)
    top_df = df.head(top_n).copy()
    return top_df

def fetch_team_submissions(team_id):
    """Query submission IDs for a team via Kaggle CLI."""
    cmd = ["kaggle", "competitions", "team-submissions", str(team_id), "--format", "json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(res.stdout)
    except Exception as e:
        print(f"Error fetching submissions for Team {team_id}: {e}")
        return []

def fetch_submission_episodes(submission_id):
    """Query match episode IDs for a submission via Kaggle CLI."""
    cmd = ["kaggle", "competitions", "episodes", str(submission_id), "--format", "json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(res.stdout)
    except Exception as e:
        print(f"Error fetching episodes for Submission {submission_id}: {e}")
        return []

def download_replay(episode_id, output_dir=REPLAYS_DIR):
    """Download replay JSON for an episode via Kaggle CLI."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = ["kaggle", "competitions", "replay", str(episode_id), "-p", output_dir]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Downloaded episode {episode_id} to {output_dir}")
        return True
    except Exception as e:
        print(f"Error downloading replay {episode_id}: {e}")
        return False

def analyze_replay_file(replay_path):
    """Deep forensic extraction of turn-by-turn metrics from a replay JSON."""
    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    info = data.get("info", {})
    team_names = info.get("TeamNames") or [a.get("Name") for a in info.get("Agents", [])] or ["Agent 0", "Agent 1"]
    ep_id = info.get("EpisodeId") or data.get("id", "Unknown")
    seed = info.get("seed", "Unknown")
    steps = data.get("steps", [])
    
    summary = {
        "episode_id": ep_id,
        "seed": seed,
        "total_steps": len(steps),
        "agents": team_names,
        "players": {}
    }
    
    for p_idx in range(2):
        summary["players"][p_idx] = {
            "name": team_names[p_idx] if p_idx < len(team_names) else f"Player {p_idx}",
            "final_reward": 0.0,
            "peak_cash": 0.0,
            "quadrant_unlocks": {},
            "crops_planted": Counter(),
            "structures_built": Counter(),
            "daily_hires": [0] * 30,
            "total_hires": 0,
            "market_orders": Counter(),
            "cash_curve": []
        }
        
    prev_unlocked = [{ "NW" }, { "NW" }]
    
    for t, step in enumerate(steps):
        day = t // 24
        hour = t % 24
        
        for p_idx in [0, 1]:
            if p_idx >= len(step):
                continue
            p_data = step[p_idx]
            obs = p_data.get("observation", {})
            act = p_data.get("action", {})
            reward = p_data.get("reward", 0.0)
            
            farms = obs.get("farms", [])
            if len(farms) > p_idx:
                farm = farms[p_idx]
                money = farm.get("money", 0.0)
                unlocked = set(farm.get("unlocked_quadrants", []))
                hires_today = farm.get("hires_today", 0)
                
                summary["players"][p_idx]["cash_curve"].append((t, money))
                summary["players"][p_idx]["peak_cash"] = max(summary["players"][p_idx]["peak_cash"], money)
                if day < 30:
                    summary["players"][p_idx]["daily_hires"][day] = max(summary["players"][p_idx]["daily_hires"][day], hires_today)
                    
                # Track newly unlocked quadrants
                new_quads = unlocked - prev_unlocked[p_idx]
                for q in new_quads:
                    summary["players"][p_idx]["quadrant_unlocks"][q] = {"turn": t, "day": day, "hour": hour}
                prev_unlocked[p_idx] = unlocked
                
                # Parse actions
                if isinstance(act, dict):
                    farmer_act = act.get("farmer", [])
                    if farmer_act:
                        op = farmer_act[0]
                        if op == "PLANT" and len(farmer_act) > 1:
                            summary["players"][p_idx]["crops_planted"][farmer_act[1]] += 1
                        elif op.startswith("BUILD_"):
                            summary["players"][p_idx]["structures_built"][op] += 1
                            
                    hands_act = act.get("hands", [])
                    for h in hands_act:
                        if h and h[0] == "PLANT" and len(h) > 1:
                            summary["players"][p_idx]["crops_planted"][h[1]] += 1
                            
                    market_act = act.get("market", [])
                    for m in market_act:
                        if m:
                            summary["players"][p_idx]["market_orders"][m[0]] += 1
                            if m[0] == "BUY_SEED" and len(m) > 1:
                                summary["players"][p_idx]["market_orders"][f"SEED_{m[1]}"] += (m[2] if len(m) > 2 else 1)
                                
            if t == len(steps) - 1:
                summary["players"][p_idx]["final_reward"] = reward
                summary["players"][p_idx]["total_hires"] = sum(summary["players"][p_idx]["daily_hires"])
                
    return summary

def main():
    parser = argparse.ArgumentParser(description="Kaggriculture Top Competitors Replay Tool")
    parser.add_argument("--top", type=int, default=10, help="Number of top teams to inspect")
    parser.add_argument("--download-all", action="store_true", help="Download replays for top teams (requires auth)")
    parser.add_argument("--episode", type=int, help="Download a specific episode ID")
    parser.add_argument("--analyze", action="store_true", help="Analyze all replays in replays/top_competitors/")
    args = parser.parse_args()
    
    print("================================================================")
    print("   Kaggriculture Leaderboard & Replay Forensics System")
    print("================================================================")
    
    top_df = get_top_teams(top_n=args.top)
    print(f"\n[+] Top {len(top_df)} Leaderboard Teams:")
    for idx, r in top_df.iterrows():
        print(f" Rank {r.Rank:2d} | Team ID: {r.TeamId:8d} | Score: {r.Score:6.1f} | Submissions: {r.SubmissionCount} | Team: {r.TeamName} ({r.TeamMemberUserNames})")
        
    auth_ok = check_kaggle_auth()
    print(f"\n[+] Kaggle Authentication Status: {'AUTHENTICATED' if auth_ok else 'UNAUTHENTICATED (Token/Key Required)'}")
    
    if args.episode:
        if auth_ok:
            print(f"Downloading episode {args.episode}...")
            download_replay(args.episode)
        else:
            print("Cannot download episode: Kaggle credentials not configured.")
            print("Run 'kaggle auth login' or export KAGGLE_API_TOKEN=xxx")
            
    if args.download_all:
        if not auth_ok:
            print("\n[!] Error: Downloading top competitor replays requires active Kaggle CLI authentication.")
            print("    Configure credentials with: 'kaggle auth login' or 'export KAGGLE_API_TOKEN=xxx'")
            print("    Then retry: python scripts/fetch_top_competitors.py --download-all")
        else:
            for idx, r in top_df.iterrows():
                print(f"\nFetching submissions for Rank {r.Rank} (Team {r.TeamId}: {r.TeamName})...")
                subs = fetch_team_submissions(r.TeamId)
                for sub in subs:
                    sub_id = sub.get("id")
                    if sub_id:
                        print(f"  Listing episodes for Submission {sub_id}...")
                        episodes = fetch_submission_episodes(sub_id)
                        for ep in episodes[:3]: # grab latest 3 episodes per sub
                            ep_id = ep.get("id")
                            if ep_id:
                                download_replay(ep_id)
                                
    if args.analyze or not args.download_all:
        print("\n================================================================")
        print("   Available Replays Analysis (replays/top_competitors/)")
        print("================================================================")
        replays = sorted(glob.glob(os.path.join(REPLAYS_DIR, "*.json")))
        valid_replays = [r for r in replays if not r.endswith("-0.json") and not r.endswith("-1.json")]
        if not valid_replays:
            print(f"No replays found in {REPLAYS_DIR}.")
        else:
            for rpath in valid_replays:
                res = analyze_replay_file(rpath)
                print(f"\n--- Episode {res['episode_id']} (Seed {res['seed']}) ---")
                for p_idx in [0, 1]:
                    p = res["players"][p_idx]
                    print(f"  Player {p_idx} [{p['name']}]:")
                    print(f"    Final Cash: ${p['final_reward']:,.2f} | Peak: ${p['peak_cash']:,.2f}")
                    print(f"    Quadrant Unlocks: {p['quadrant_unlocks']}")
                    print(f"    Crops Planted: {dict(p['crops_planted'])}")
                    print(f"    Structures: {dict(p['structures_built'])}")
                    print(f"    Total Farmhand Hires: {p['total_hires']}")
                    print(f"    Daily Hires Schedule: {p['daily_hires']}")

if __name__ == "__main__":
    main()
