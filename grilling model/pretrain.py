"""
Standalone CLI pre-training entrypoint for the Kaggriculture Champion Bot.
Loads replay JSONs (or generates high-scoring games if none exist),
applies D4 dihedral augmentation, computes AWIL sample weights, trains the network
via StagedAWILTrainer, and saves the pre-trained checkpoint.
"""

from typing import Any, Dict, List, Optional
import argparse
import glob
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

grilling_model_root = Path(__file__).resolve().parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

from src.encoder import encode_observation
from src.network import ChampionFullNetwork
from src.world_model import TwoScaleHierarchicalWorldModel
from src.trainer import StagedAWILTrainer
from src.agent import ChampionAgent
from src.awil_dataset import (
    AWILDataset,
    Stage1ValueBaseline,
    parse_replay_transitions,
    train_stage1_baseline,
)


def generate_seed_replays(num_matches: int = 4) -> List[Dict[str, Any]]:
    """
    Generates high-scoring seed matches using ChampionAgent against the starter baseline
    if no raw downloaded replay JSONs are found.
    """
    try:
        from kaggle_environments import make
    except ImportError:
        return []

    agent_bot = ChampionAgent()
    replays = []
    print(f"Generating {num_matches} seed matches via ChampionAgent vs starter baseline...")

    for i in range(num_matches):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 42 + i}, debug=False)
        if i % 2 == 0:
            env.run([agent_bot, "starter"])
            p_idx = 0
        else:
            env.run(["starter", agent_bot])
            p_idx = 1

        steps = []
        for step in env.steps:
            step_record = []
            for player_state in step:
                step_record.append({
                    "observation": player_state.observation,
                    "action": player_state.action,
                    "reward": player_state.reward,
                    "status": player_state.status,
                })
            steps.append(step_record)

        r_champ = steps[-1][p_idx].get("reward", 0.0) or 0.0
        r_start = steps[-1][1 - p_idx].get("reward", 0.0) or 0.0
        print(f"  Match {i+1}/{num_matches}: Champion (${r_champ:,.0f}) vs Starter (${r_start:,.0f})")
        replays.append({"steps": steps})

    return replays


def load_replays_from_dir(replays_dir: str) -> List[Dict[str, Any]]:
    """Loads all JSON replay files from the given directory."""
    replay_files = glob.glob(os.path.join(replays_dir, "*.json"))
    replays = []
    for fpath in replay_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "steps" in data:
                    replays.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")
    return replays


def main():
    parser = argparse.ArgumentParser(description="Pre-train Champion Bot with AWIL and World Model")
    parser.add_argument("--replays-dir", type=str, default="grilling model/data/replays", help="Directory with replay JSON files")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="grilling model/data/pretrained_champion.pt", help="Path to save weights")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.replays_dir, exist_ok=True)

    print(f"=== Kaggriculture Champion Bot Pre-training ===")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch size: {args.batch_size}")

    replays = load_replays_from_dir(args.replays_dir)
    if not replays:
        print(f"No replay files found in '{args.replays_dir}'. Generating seed matches from ChampionAgent...")
        replays = generate_seed_replays(num_matches=4)

    print(f"Parsing replay transitions from {len(replays)} matches...")
    all_raw_transitions = []
    for replay in replays:
        parsed = parse_replay_transitions(replay, min_cash=30000.0)
        all_raw_transitions.extend(parsed)

    if not all_raw_transitions:
        print("Error: No qualifying transitions found. Try lowering min_cash or adding more replays.")
        sys.exit(1)

    print(f"Total qualifying transitions parsed: {len(all_raw_transitions):,}")

    # Stage 1 Value baseline & AWIL weights
    print("Fitting Stage 1 value baseline network...")
    initial_dataset = AWILDataset(transitions=all_raw_transitions, augment=False)
    baseline_net = Stage1ValueBaseline()
    baseline_model = train_stage1_baseline(
        model=baseline_net,
        dataset=initial_dataset,
        epochs=3,
        batch_size=args.batch_size,
        device=args.device
    )

    # Wrap in PyTorch Dataset with D4 dihedral augmentation and AWIL weights
    dataset = AWILDataset(all_raw_transitions, augment=True, baseline_model=baseline_model)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset ready: {len(dataset):,} samples ({len(dataloader)} batches per epoch).")

    # Initialize Networks
    network = ChampionFullNetwork().to(args.device)
    world_model = TwoScaleHierarchicalWorldModel().to(args.device)

    trainer = StagedAWILTrainer(
        network=network,
        world_model=world_model,
        lr=args.lr,
        device=args.device,
        total_steps=args.epochs * len(dataloader),
    )

    print("\nStarting Staged AWIL Training...")
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(dataloader)
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] | "
            f"Total Loss: {metrics['total_loss']:.4f} | "
            f"Policy Loss: {metrics['policy_loss']:.4f} | "
            f"Value Loss: {metrics['value_loss']:.4f} | "
            f"Entropy Coeff: {metrics['entropy_coeff']:.5f}"
        )

    # Save checkpoint
    trainer.save_checkpoint(args.save_path)
    print(f"\nPre-training complete! Checkpoint saved to: {args.save_path}")


if __name__ == "__main__":
    main()
