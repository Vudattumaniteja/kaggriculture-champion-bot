import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.models.encoder import encode_observation, MACRO_ACTIONS, NUM_MACRO_ACTIONS, PRODUCTS, CROPS
from src.training.dataset import infer_macro_action
from src.models.hrl_150k_network import scalar_to_two_hot, categorical_to_scalar

def test_parse_replay():
    path = "replays/latest_frontier/episode-93981122-replay.json"
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    steps = d["steps"]
    team_names = d.get("info", {}).get("TeamNames", ["P0", "P1"])
    print(f"Replay: {team_names}, Steps: {len(steps)}")
    
    # Test step 0 player 0 (Abracadabra)
    obs0 = steps[0][0]["observation"]
    act1 = steps[1][0]["action"]
    g0, s0 = encode_observation(obs0)
    macro0 = infer_macro_action(act1, obs0.get("day", 0))
    print(f"Step 0 Abracadabra Action: {act1['farmer']}, Macro: {MACRO_ACTIONS[macro0]} ({macro0})")
    print(f"Grid shape: {g0.shape}, Scalar shape: {s0.shape}")
    
    final_r0 = float(steps[-1][0].get("reward", 0))
    two_hot = scalar_to_two_hot(torch.tensor([final_r0]))
    recovered_r0 = categorical_to_scalar(two_hot, is_logits=False).item()
    print(f"Final Reward: ${final_r0:,.1f} -> Two-Hot -> Recovered: ${recovered_r0:,.1f}")
    assert abs(final_r0 - recovered_r0) < 100.0, f"Value mismatch: {final_r0} vs {recovered_r0}"
    print("Value encoding test passed successfully!")

if __name__ == "__main__":
    test_parse_replay()
