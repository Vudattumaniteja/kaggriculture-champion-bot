"""
Multi-Industry Imitation Trajectory Dataset Generation pipeline for Kaggriculture.
Simulates extensive matches across diverse baseline agents (crops, livestock, arbitrage, hybrid)
and serializes state-action-value transitions to compressed NPZ format.
"""

import os
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from kaggle_environments import make

from src.models.encoder import encode_observation, NUM_MACRO_ACTIONS, MACRO_ACTIONS
from src.agents.balanced_farmer import (
    crop_farmer_portfolio,
    crop_farmer_carrot,
    crop_farmer_melon,
    crop_farmer_strawberry,
    crop_farmer_tomato,
    crop_farmer_wheat,
)
from src.agents.livestock_bot import livestock_agent
from src.agents.arbitrage_bot import arbitrage_agent
from src.agents.hybrid_expert import hybrid_expert_agent


def infer_macro_action(act: Dict[str, Any], day: int) -> int:
    """Infers the high-level macro-action class (0-9) from low-level actions."""
    market = act.get("market", [])
    farmer = act.get("farmer", ["PASS"])
    hands = act.get("hands", [])
    all_unit_ops = [farmer] + hands

    # 1. End game liquidation (Day >= 27)
    if day >= 27:
        return 7  # HARVEST_AND_LIQUIDATE_ALL

    # 2. Check livestock care / feed / fertilizer in unit operations
    for op in all_unit_ops:
        if isinstance(op, list) and len(op) > 0:
            cmd = op[0]
            if cmd in ["FEED", "CARE", "COLLECT_FERTILIZER"]:
                return 9  # LIVESTOCK_CARE_FEED
            elif cmd == "BUILD_COOP" or (cmd == "PLACE" and len(op) > 1 and op[1] == "GOOSE"):
                return 5  # BUILD_GOOSE_COOP
            elif cmd == "BUILD_PASTURE" or (cmd == "PLACE" and len(op) > 1 and op[1] in ["COW", "SHEEP"]):
                return 6  # BUILD_PASTURE_LIVESTOCK

    # 3. Market orders
    for m in market:
        if not isinstance(m, list) or len(m) == 0:
            continue
        op = m[0]
        if op == "BUY_LAND":
            return 3  # BUY_LAND_EXPANSION
        elif op == "HIRE":
            return 4  # HIRE_EXTRA_LABOR
        elif op == "BUY_ANIMAL":
            animal = m[1] if len(m) > 1 else "GOOSE"
            return 5 if animal == "GOOSE" else 6
        elif op == "BUY_PRODUCT":
            return 8  # MARKET_ARBITRAGE_TRADE
        elif op == "BUY_SEED":
            crop = m[1] if len(m) > 1 else "CARROT"
            if crop == "CARROT":
                return 0  # FARM_CARROTS_INTENSIVE
            elif crop == "WHEAT":
                return 1  # FARM_WHEAT_EXPANSION
            else:
                return 2  # FARM_DIVERSIFIED (Tomato, Strawberry, Melon)

    # 4. Crop unit operations (Planting / Fertilizing)
    for op in all_unit_ops:
        if isinstance(op, list) and len(op) > 0:
            cmd = op[0]
            if cmd == "PLANT":
                crop = op[1] if len(op) > 1 else "CARROT"
                if crop == "CARROT":
                    return 0
                elif crop == "WHEAT":
                    return 1
                else:
                    return 2
            elif cmd == "FERTILIZE":
                return 2

    # 5. Default based on active market trades or farming
    for m in market:
        if isinstance(m, list) and len(m) > 0 and m[0] == "SELL":
            return 8

    return 0  # Default active farming


AGENT_POOL: List[Tuple[str, Callable]] = [
    ("CropPortfolio", crop_farmer_portfolio),
    ("CropCarrot", crop_farmer_carrot),
    ("CropMelon", crop_farmer_melon),
    ("CropStrawberry", crop_farmer_strawberry),
    ("CropTomato", crop_farmer_tomato),
    ("CropWheat", crop_farmer_wheat),
    ("LivestockHusbandry", livestock_agent),
    ("TownShopArbitrage", arbitrage_agent),
    ("HybridExpert", hybrid_expert_agent),
]


def generate_multi_industry_dataset(
    num_episodes: int = 50,
    save_path: str = "data/multi_industry_dataset.npz",
    base_seed: int = 2000,
) -> Dict[str, Any]:
    """
    Generates a rich multi-industry trajectory dataset across diverse baseline matchups and seeds.
    Saves compressed NPZ and returns sample statistics.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Diverse matchup pairings
    matchup_pairs = [
        ("CropPortfolio", "LivestockHusbandry"),
        ("CropCarrot", "HybridExpert"),
        ("CropMelon", "TownShopArbitrage"),
        ("CropStrawberry", "LivestockHusbandry"),
        ("CropTomato", "TownShopArbitrage"),
        ("CropWheat", "HybridExpert"),
        ("LivestockHusbandry", "TownShopArbitrage"),
        ("LivestockHusbandry", "HybridExpert"),
        ("TownShopArbitrage", "HybridExpert"),
        ("CropPortfolio", "TownShopArbitrage"),
        ("CropCarrot", "LivestockHusbandry"),
        ("CropMelon", "LivestockHusbandry"),
        ("CropStrawberry", "HybridExpert"),
        ("CropTomato", "LivestockHusbandry"),
        ("CropWheat", "TownShopArbitrage"),
        ("TownShopArbitrage", "CropCarrot"),
        ("HybridExpert", "CropPortfolio"),
    ]

    agent_dict = dict(AGENT_POOL)

    all_grids: List[np.ndarray] = []
    all_scalars: List[np.ndarray] = []
    all_actions: List[int] = []
    all_values: List[float] = []

    # Statistics tracking
    agent_rewards: Dict[str, List[float]] = {name: [] for name, _ in AGENT_POOL}
    matchup_records: List[Dict[str, Any]] = []

    print(f"================================================================================")
    print(f" Starting Batch Multi-Industry Dataset Generation: {num_episodes} Episodes (~72,000 Transitions)")
    print(f" Target File: {save_path}")
    print(f"================================================================================")
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        pair_idx = (ep - 1) % len(matchup_pairs)
        name_a, name_b = matchup_pairs[pair_idx]

        # Alternate P0/P1 positions
        if ep % 2 == 0:
            name0, name1 = name_b, name_a
        else:
            name0, name1 = name_a, name_b

        agent0 = agent_dict[name0]
        agent1 = agent_dict[name1]

        ep_seed = base_seed + ep
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": ep_seed})
        env.reset()

        ep_grids_p0: List[np.ndarray] = []
        ep_scalars_p0: List[np.ndarray] = []
        ep_actions_p0: List[int] = []

        ep_grids_p1: List[np.ndarray] = []
        ep_scalars_p1: List[np.ndarray] = []
        ep_actions_p1: List[int] = []

        ep_t0 = time.time()

        while not env.done:
            obs0 = env.state[0].observation
            obs1 = env.state[1].observation

            g0, s0 = encode_observation(obs0)
            g1, s1 = encode_observation(obs1)

            act0 = agent0(obs0)
            act1 = agent1(obs1)

            macro0 = infer_macro_action(act0, obs0.get("day", 0))
            macro1 = infer_macro_action(act1, obs1.get("day", 0))

            ep_grids_p0.append(g0)
            ep_scalars_p0.append(s0)
            ep_actions_p0.append(macro0)

            ep_grids_p1.append(g1)
            ep_scalars_p1.append(s1)
            ep_actions_p1.append(macro1)

            env.step([act0, act1])

        # Terminal rewards
        r0 = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
        r1 = float(env.state[1].reward if env.state[1].reward is not None else 0.0)

        agent_rewards[name0].append(r0)
        agent_rewards[name1].append(r1)

        # Smooth normalized outcome target in [-1, 1]
        v0 = float(np.tanh((r0 - r1) / 4000.0))
        v1 = float(np.tanh((r1 - r0) / 4000.0))

        all_grids.extend(ep_grids_p0)
        all_scalars.extend(ep_scalars_p0)
        all_actions.extend(ep_actions_p0)
        all_values.extend([v0] * len(ep_actions_p0))

        all_grids.extend(ep_grids_p1)
        all_scalars.extend(ep_scalars_p1)
        all_actions.extend(ep_actions_p1)
        all_values.extend([v1] * len(ep_actions_p1))

        ep_duration = time.time() - ep_t0
        winner_str = name0 if r0 > r1 else (name1 if r1 > r0 else "TIE")

        matchup_records.append({
            "episode": ep,
            "seed": ep_seed,
            "p0_agent": name0,
            "p1_agent": name1,
            "p0_bank": r0,
            "p1_bank": r1,
            "winner": winner_str,
            "duration": ep_duration,
        })

        if ep % 5 == 0 or ep == num_episodes:
            print(
                f"  [Ep {ep:2d}/{num_episodes}] {name0:18s} (${r0:6.0f}) vs {name1:18s} (${r1:6.0f}) -> {winner_str:18s} | Total Samples: {len(all_actions):,} | ({ep_duration:.2f}s)"
            )

    total_time = time.time() - start_time
    grids_arr = np.array(all_grids, dtype=np.float32)
    scalars_arr = np.array(all_scalars, dtype=np.float32)
    actions_arr = np.array(all_actions, dtype=np.int64)
    values_arr = np.array(all_values, dtype=np.float32)

    # Save to compressed NPZ
    np.savez_compressed(
        save_path,
        grids=grids_arr,
        scalars=scalars_arr,
        actions=actions_arr,
        values=values_arr,
    )

    # Compute detailed sample statistics
    total_samples = len(actions_arr)
    action_counts = {MACRO_ACTIONS[i]: int(np.sum(actions_arr == i)) for i in range(NUM_MACRO_ACTIONS)}
    action_dist = {name: (count, count / total_samples * 100.0) for name, count in action_counts.items()}

    agent_mean_banks = {
        name: (float(np.mean(rewards)), len(rewards))
        for name, rewards in agent_rewards.items()
        if len(rewards) > 0
    }

    stats = {
        "total_samples": total_samples,
        "num_episodes": num_episodes,
        "grids_shape": list(grids_arr.shape),
        "scalars_shape": list(scalars_arr.shape),
        "actions_shape": list(actions_arr.shape),
        "values_shape": list(values_arr.shape),
        "action_distribution": action_dist,
        "agent_mean_banks": agent_mean_banks,
        "duration_seconds": total_time,
        "save_path": os.path.abspath(save_path),
        "file_size_mb": os.path.getsize(save_path) / (1024 * 1024),
    }

    print(f"================================================================================")
    print(f" Dataset Generation Complete in {total_time:.1f}s!")
    print(f" Total Transitions: {total_samples:,}")
    print(f" File Saved: {save_path} ({stats['file_size_mb']:.2f} MB)")
    print(f"================================================================================")

    return stats


class ExpertImitationDataset(Dataset):
    def __init__(self, npz_path: str = "data/multi_industry_dataset.npz"):
        data = np.load(npz_path)
        self.grids = torch.tensor(data["grids"], dtype=torch.float32)
        self.scalars = torch.tensor(data["scalars"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.long)
        self.values = torch.tensor(data["values"], dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.grids[idx], self.scalars[idx], self.actions[idx], self.values[idx]
