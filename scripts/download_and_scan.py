import subprocess
import json
import os
import glob

SUBMISSION_IDS = [55582913, 55582344, 55580706]
DEST_DIR = "replays/latest_frontier"
os.makedirs(DEST_DIR, exist_ok=True)

def get_episodes_for_sub(sub_id):
    cmd = f"kaggle competitions episodes {sub_id}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    episodes = []
    for line in res.stdout.splitlines():
        parts = line.strip().split()
        if len(parts) >= 1 and parts[0].isdigit():
            episodes.append(int(parts[0]))
    return episodes

all_episodes = {}
for sub_id in SUBMISSION_IDS:
    eps = get_episodes_for_sub(sub_id)
    print(f"Submission {sub_id}: found {len(eps)} episodes: {eps}")
    for ep in eps:
        if ep not in all_episodes:
            all_episodes[ep] = sub_id

print(f"\nTotal unique episodes across recent submissions: {len(all_episodes)}")

for ep, sub_id in all_episodes.items():
    dest_file = os.path.join(DEST_DIR, f"episode-{ep}-replay.json")
    if not os.path.exists(dest_file):
        print(f"Downloading episode {ep} (sub {sub_id})...")
        cmd = f"kaggle competitions replay {ep} -p {DEST_DIR}"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        print(f"Episode {ep} already exists.")

print("\nDownload complete. Scanning replays summary...")

summary = []
for file_path in glob.glob(os.path.join(DEST_DIR, "episode-*-replay.json")):
    ep_id = int(os.path.basename(file_path).split("-")[1])
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # In kaggle replays, steps is a list
        steps = data.get("steps", [])
        last_step = steps[-1] if steps else []
        rewards = [agent.get("reward") for agent in last_step]
        
        # Check agents metadata
        agents = data.get("info", {}).get("TeamNames", [])
        
        summary.append({
            "episode_id": ep_id,
            "sub_id": all_episodes.get(ep_id, "unknown"),
            "steps": len(steps),
            "rewards": rewards,
            "team_names": agents,
            "file": file_path
        })
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

summary.sort(key=lambda x: x["episode_id"], reverse=True)
print(f"\nScanned {len(summary)} matches:")
for s in summary:
    print(f"Ep {s['episode_id']} (Sub {s['sub_id']}): Rewards: {s['rewards']}, Teams: {s['team_names']}, Steps: {s['steps']}")
