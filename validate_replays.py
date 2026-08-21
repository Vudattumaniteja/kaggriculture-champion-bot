import os
import sys
import json
import glob
from pathlib import Path

# Add 'grilling model' to sys.path
grilling_model_dir = Path(__file__).parent / "grilling model"
if str(grilling_model_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(grilling_model_dir.resolve()))

from src.awil_dataset import parse_replay_transitions

def validate_all_replays(replay_dir="grilling model/data/replays", min_cash=50000.0):
    replay_files = glob.glob(os.path.join(replay_dir, "episode-*-replay.json"))
    print(f"Total replay files found in {replay_dir}: {len(replay_files)}")
    
    total_transitions = 0
    replay_stats = []
    top_scores = []
    
    for fpath in sorted(replay_files):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            steps = data.get("steps", [])
            rewards = [s.get("reward", 0.0) or 0.0 for s in steps[-1]] if steps else []
            max_reward = max(rewards) if rewards else 0.0
            
            transitions = parse_replay_transitions(data, min_cash=min_cash)
            total_transitions += len(transitions)
            
            p0_reward = rewards[0] if len(rewards) > 0 else 0.0
            p1_reward = rewards[1] if len(rewards) > 1 else 0.0
            
            replay_stats.append({
                "file": os.path.basename(fpath),
                "num_steps": len(steps),
                "rewards": rewards,
                "max_reward": max_reward,
                "parsed_transitions": len(transitions)
            })
            
            for r in rewards:
                if r is not None and r > 0:
                    top_scores.append(float(r))
                    
        except Exception as e:
            print(f"Error parsing {fpath}: {e}")
            
    top_scores.sort(reverse=True)
    
    print("\n" + "="*70)
    print("REPLAY VALIDATION AND TRANSITION PARSE REPORT")
    print("="*70)
    print(f"Total Validated Replay Files : {len(replay_stats)}")
    print(f"Total High-Scoring Transitions: {total_transitions}")
    if top_scores:
        print(f"Highest Individual Game Cash : ${max(top_scores):,.2f}")
        print(f"Top 10 Player Cash Scores    : {[round(s, 2) for s in top_scores[:10]]}")
        print(f"Median Player Cash Score     : ${top_scores[len(top_scores)//2]:,.2f}")
    print("="*70)
    
    return replay_stats, total_transitions

if __name__ == "__main__":
    validate_all_replays()
