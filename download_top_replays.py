import subprocess
import json
import re
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

top_submissions = [
    {"rank": 1, "team_id": 16623716, "name": "tetsuya", "sub_id": "55574890", "score": 3142.0},
    {"rank": 2, "team_id": 16644287, "name": "Ryo Hasegawa", "sub_id": "55614463", "score": 3103.9},
    {"rank": 3, "team_id": 16677252, "name": "Team 16677252", "sub_id": "55540317", "score": 3066.6},
    {"rank": 4, "team_id": 16711376, "name": "Arman Tuganbaev", "sub_id": "55617399", "score": 3043.6},
    {"rank": 5, "team_id": 16703045, "name": "ReCurSiON", "sub_id": "55491717", "score": 2948.5},
    {"rank": 6, "team_id": 16730719, "name": "Xiaowenhao404", "sub_id": "55610626", "score": 2928.2},
    {"rank": 7, "team_id": 16714457, "name": "Crop Dusta", "sub_id": "55623460", "score": 2930.8},
    {"rank": 8, "team_id": 16740010, "name": "Kobe BRYANT", "sub_id": "55621582", "score": 2921.7},
    {"rank": 9, "team_id": 16715733, "name": "peikopon", "sub_id": "55577827", "score": 2914.6},
    {"rank": 10, "team_id": 16709501, "name": "Galaxantic", "sub_id": "55576886", "score": 2911.1},
]

target_dir = Path("grilling model/data/replays")
target_dir.mkdir(parents=True, exist_ok=True)

print("Fetching episodes for top 10 submissions...")
all_episodes_to_download = []

for team in top_submissions:
    sub_id = team["sub_id"]
    res = subprocess.run(["kaggle", "competitions", "episodes", sub_id], capture_output=True, text=True)
    lines = res.stdout.strip().splitlines()
    
    episodes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[0].isdigit() and "COMPLETED" in line:
            episodes.append(parts[0])
    
    print(f"Rank {team['rank']}: {team['name']} (Sub {sub_id}) - Found {len(episodes)} completed episodes")
    # Select up to 10 latest completed episodes per team
    selected = episodes[:10]
    for ep in selected:
        all_episodes_to_download.append({
            "episode_id": ep,
            "rank": team["rank"],
            "team_name": team["name"],
            "score": team["score"]
        })

print(f"\nTotal episodes selected for download: {len(all_episodes_to_download)}")

def download_episode(item):
    ep_id = item["episode_id"]
    dest_file = target_dir / f"episode-{ep_id}-replay.json"
    if dest_file.exists() and dest_file.stat().st_size > 1000:
        return ep_id, True, "Already exists"
    
    res = subprocess.run(
        ["kaggle", "competitions", "replay", ep_id, "-p", str(target_dir)],
        capture_output=True,
        text=True
    )
    if dest_file.exists() and dest_file.stat().st_size > 1000:
        return ep_id, True, "Downloaded"
    else:
        return ep_id, False, res.stderr or res.stdout

print("\nStarting concurrent replay downloads...")
success_count = 0
with ThreadPoolExecutor(max_workers=6) as executor:
    futures = {executor.submit(download_episode, item): item for item in all_episodes_to_download}
    for future in as_completed(futures):
        ep_id, success, msg = future.result()
        item = futures[future]
        if success:
            success_count += 1
            print(f"  [OK] Episode {ep_id} (Rank {item['rank']} {item['team_name']}) - {msg}")
        else:
            print(f"  [FAIL] Episode {ep_id} (Rank {item['rank']} {item['team_name']}) - {msg}")

print(f"\nDownload summary: {success_count}/{len(all_episodes_to_download)} successfully downloaded into {target_dir}")
