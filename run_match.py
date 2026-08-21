"""
Kaggriculture Tournament Match Runner and Forensic Analysis.
Executes matches between the Pretrained Champion Model and Baseline / Competitor bots.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRILLING_MODEL_DIR = os.path.join(BASE_DIR, "grilling model")
if GRILLING_MODEL_DIR not in sys.path:
    sys.path.insert(0, GRILLING_MODEL_DIR)

from kaggle_environments import make
from src.agent import ChampionAgent


def run_tournament_match(
    champion_weights: str = "grilling model/data/pretrained_champion.pt",
    opponent_name: str = "starter",
    champion_player_idx: int = 0,
    episode_steps: int = 720,
    save_html_replay: str = None,
    save_json_replay: str = None,
) -> Dict[str, Any]:
    """
    Executes a head-to-head match in the Kaggle kaggriculture simulation environment.
    """
    weights_path = os.path.join(BASE_DIR, champion_weights) if not os.path.isabs(champion_weights) else champion_weights
    print(f"[*] Initializing Pretrained Champion Agent from: {weights_path}")
    champion_agent = ChampionAgent(model_weights_path=weights_path if os.path.exists(weights_path) else None)

    if champion_player_idx == 0:
        agents = [champion_agent, opponent_name]
        agent_names = ["Pretrained Champion (P0)", f"{opponent_name.capitalize()} (P1)"]
    else:
        agents = [opponent_name, champion_agent]
        agent_names = [f"{opponent_name.capitalize()} (P0)", "Pretrained Champion (P1)"]

    print(f"[*] Starting 720-step simulation: {agent_names[0]} vs {agent_names[1]}...")
    env = make("kaggriculture", configuration={"episodeSteps": episode_steps}, debug=True)

    start_time = time.time()
    steps = env.run(agents)
    elapsed = time.time() - start_time

    p0_final_reward = env.steps[-1][0].reward
    p1_final_reward = env.steps[-1][1].reward

    champ_score = p0_final_reward if champion_player_idx == 0 else p1_final_reward
    opp_score = p1_final_reward if champion_player_idx == 0 else p0_final_reward
    diff = champ_score - opp_score
    margin = ((champ_score - opp_score) / max(1.0, opp_score)) * 100.0

    print(f"[+] Match finished in {elapsed:.2f}s ({episode_steps / elapsed:.1f} steps/s)")
    print("\n" + "=" * 60)
    print("MATCH RESULTS:")
    print(f"  Seat 0 ({agent_names[0]}): ${p0_final_reward:,.2f}")
    print(f"  Seat 1 ({agent_names[1]}): ${p1_final_reward:,.2f}")
    print(f"  Champion Final Bank: ${champ_score:,.2f}")
    print(f"  Opponent Final Bank: ${opp_score:,.2f}")
    print(f"  Net Lead:            +${diff:,.2f} (+{margin:.1f}%)")
    print("=" * 60)

    # Save replay files if requested
    if save_html_replay:
        html_content = env.render(mode="html", width=1000, height=700)
        with open(save_html_replay, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[+] Saved interactive HTML replay to: {save_html_replay}")

    if save_json_replay:
        replay_dict = {
            "configuration": env.configuration,
            "specification": env.specification,
            "steps": env.steps,
        }
        with open(save_json_replay, "w", encoding="utf-8") as f:
            json.dump(replay_dict, f)
        print(f"[+] Saved JSON match replay to: {save_json_replay}")

    return {
        "p0_reward": p0_final_reward,
        "p1_reward": p1_final_reward,
        "champion_score": champ_score,
        "opponent_score": opp_score,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kaggriculture Match")
    parser.add_argument("--weights", type=str, default="grilling model/data/pretrained_champion.pt")
    parser.add_argument("--opponent", type=str, default="starter")
    parser.add_argument("--seat", type=int, default=0, help="0 for Player 1, 1 for Player 2")
    parser.add_argument("--html", type=str, default="match_replay.html")
    args = parser.parse_args()

    run_tournament_match(
        champion_weights=args.weights,
        opponent_name=args.opponent,
        champion_player_idx=args.seat,
        save_html_replay=args.html,
    )
