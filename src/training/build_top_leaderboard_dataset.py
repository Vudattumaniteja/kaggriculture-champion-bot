"""
Build Top-Leaderboard Grandmaster Dataset for Teacher Model Training.
Parses all available high-scoring Kaggle replays from top leaderboard teams
(including Episode 93981122 Abracadabra $133.4k, kawashigi, peikopon, and Thomas Tschinkel)
and fuses them with high-performing multi-industry simulations into a unified dataset.
"""

import glob
import json
import os
import sys
import time
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.models.encoder import encode_observation, MACRO_ACTIONS, NUM_MACRO_ACTIONS, PRODUCTS, CROPS, ANIMALS
from src.training.dataset import infer_macro_action
from src.models.top_leaderboard_teacher import (
    scalar_to_two_hot,
    V_MIN_TEACHER,
    V_MAX_TEACHER,
    NUM_BINS_TEACHER,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def extract_aux_targets_from_obs(obs: Dict[str, Any], player: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts auxiliary targets from observation:
    1. Tile yield map: (2, 10, 10)
    2. Price vector: (9,)
    3. Animal health & fertilizer aux: (4,) [care_today, fertilizer_in_shed, num_cows, num_sheep]
    """
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}

    # 1. Tile yields (2, 10, 10)
    tile_yields = np.zeros((2, 10, 10), dtype=np.float32)
    tiles = my_farm.get("tiles", [])
    num_cows = 0
    num_sheep = 0
    
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    tile_yields[0, r, c] = min(float(t.get("yield_units", 0)) / 6.0, 1.0)
                elif k in ["COOP", "PASTURE"]:
                    tile_yields[1, r, c] = 1.0
                    animal = t.get("animal")
                    if animal == "COW":
                        num_cows += 1
                    elif animal == "SHEEP":
                        num_sheep += 1

    # 2. Prices (9,)
    prices_raw = market.get("prices", {})
    prices = np.zeros(len(PRODUCTS), dtype=np.float32)
    for idx, prod in enumerate(PRODUCTS):
        prices[idx] = min(float(prices_raw.get(prod, 25)) / 250.0, 1.0)

    # 3. Animal aux (4,)
    shed = private.get("shed", {})
    fert_count = float(shed.get("FERTILIZER", 0))
    care_today = float(my_farm.get("hires_today", 0))
    animal_aux = np.array([
        min(care_today / 5.0, 1.0),
        min(fert_count / 50.0, 1.0),
        min(float(num_cows) / 10.0, 1.0),
        min(float(num_sheep) / 10.0, 1.0),
    ], dtype=np.float32)

    return tile_yields, prices, animal_aux


def build_top_leaderboard_dataset(
    output_path: str = "data/top_leaderboard_teacher_dataset.npz",
    min_replay_score: float = 30000.0,
    include_simulated_base: bool = True,
) -> Dict[str, Any]:
    print("=" * 80)
    print(" BUILDING TOP-LEADERBOARD GRANDMASTER TEACHER DATASET")
    print("=" * 80)
    start_time = time.time()

    replay_files = glob.glob("replays/**/*.json", recursive=True)
    print(f"Found {len(replay_files)} total replay JSON files across workspace.")

    all_grids: List[np.ndarray] = []
    all_scalars: List[np.ndarray] = []
    all_actions: List[int] = []
    all_values_raw: List[float] = []
    all_future_scalars: List[np.ndarray] = []
    all_tile_yields: List[np.ndarray] = []
    all_future_prices: List[np.ndarray] = []
    all_animal_aux: List[np.ndarray] = []
    all_weights: List[float] = []

    grandmaster_matches: List[Dict[str, Any]] = []
    parsed_replay_count = 0
    top_replay_transitions = 0

    # Process all Kaggle replays
    for rf in replay_files:
        try:
            with open(rf, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            if "steps" not in d or len(d["steps"]) < 2:
                continue

            names = d.get("info", {}).get("TeamNames", ["P0", "P1"])
            ep_id = d.get("id") or os.path.basename(rf)
            steps = d["steps"]
            last_step = steps[-1]
            r0 = float(last_step[0].get("reward") or 0.0)
            r1 = float(last_step[1].get("reward") or 0.0)

            # Check if match has high-scoring participant
            max_score = max(r0, r1)
            if max_score < min_replay_score:
                continue

            parsed_replay_count += 1
            grandmaster_matches.append({
                "ep_id": ep_id,
                "team0": names[0],
                "team1": names[1],
                "reward0": r0,
                "reward1": r1,
                "file": rf,
            })

            num_steps = len(steps)
            for p_idx in [0, 1]:
                final_reward = r0 if p_idx == 0 else r1
                opp_reward = r1 if p_idx == 0 else r0
                team_name = names[p_idx]

                # Weighting: prioritize high scores ($100k-$155k)
                if final_reward >= 130000.0:
                    weight = 4.0
                elif final_reward >= 90000.0:
                    weight = 2.5
                elif final_reward >= 60000.0:
                    weight = 1.5
                elif final_reward >= 30000.0:
                    weight = 1.0
                else:
                    weight = 0.5

                for t in range(num_steps - 1):
                    obs = steps[t][p_idx].get("observation", {})
                    act = steps[t + 1][p_idx].get("action", {})
                    day = obs.get("day", t // 24)

                    g, s = encode_observation(obs)
                    macro_act = infer_macro_action(act, day)

                    # Future state at t+4
                    t_plus_4 = min(t + 4, num_steps - 1)
                    obs_plus_4 = steps[t_plus_4][p_idx].get("observation", {})
                    _, s_plus_4 = encode_observation(obs_plus_4)

                    # Aux targets
                    y_map, p_curr, a_aux = extract_aux_targets_from_obs(obs, p_idx)

                    # Future prices at t+24
                    t_plus_24 = min(t + 24, num_steps - 1)
                    obs_plus_24 = steps[t_plus_24][p_idx].get("observation", {})
                    _, p_future, _ = extract_aux_targets_from_obs(obs_plus_24, p_idx)

                    all_grids.append(g)
                    all_scalars.append(s)
                    all_actions.append(macro_act)
                    all_values_raw.append(final_reward)
                    all_future_scalars.append(s_plus_4)
                    all_tile_yields.append(y_map)
                    all_future_prices.append(p_future)
                    all_animal_aux.append(a_aux)
                    all_weights.append(weight)
                    top_replay_transitions += 1

        except Exception as e:
            pass

    print(f"Parsed {parsed_replay_count} top Kaggle replay matches -> {top_replay_transitions:,} Grandmaster transitions extracted.")

    # Incorporate existing massive datasets if available for complete coverage
    if include_simulated_base and os.path.exists("data/massive_150k_grandmaster_dataset.npz"):
        print("\nIntegrating diverse simulated Grandmaster dataset (150k+ samples)...")
        sim_data = np.load("data/massive_150k_grandmaster_dataset.npz")
        s_grids = sim_data["grids"]
        s_scalars = sim_data["scalars"]
        s_actions = sim_data["actions"]
        s_future_scalars = sim_data["future_scalars"]
        
        # Subsample high quality transitions
        num_sim = len(s_actions)
        print(f"Loaded {num_sim:,} simulated transitions from massive_150k_grandmaster_dataset.npz")
        
        # Extract or construct aux targets for simulated samples
        # For simulated samples, default weights to 0.75
        sim_weights = np.full(num_sim, 0.75, dtype=np.float32)
        
        # Convert 1001-bin values to continuous raw values
        sim_v1001 = sim_data["values_1001"]
        bin_centers = torch.linspace(V_MIN_TEACHER, V_MAX_TEACHER, NUM_BINS_TEACHER)
        sim_v_symlog = np.sum(sim_v1001 * bin_centers.numpy(), axis=-1)
        sim_v_raw = np.sign(sim_v_symlog) * (np.exp(np.abs(sim_v_symlog)) - 1.0)

        # Tile yields, prices, and animal aux for simulated transitions
        sim_yields = np.zeros((num_sim, 2, 10, 10), dtype=np.float32)
        sim_yields[:, 0, :, :] = s_grids[:, 5, :, :] # crop yield channel in encoder
        sim_yields[:, 1, :, :] = s_grids[:, 6, :, :] # coop/pasture channel
        
        sim_prices = s_scalars[:, 22:31] # prices in scalar vector
        
        sim_animal_aux = np.zeros((num_sim, 4), dtype=np.float32)
        sim_animal_aux[:, 0] = s_scalars[:, 6] # hires
        sim_animal_aux[:, 1] = s_scalars[:, 16] # fert in shed
        
        all_grids.extend(list(s_grids))
        all_scalars.extend(list(s_scalars))
        all_actions.extend(list(s_actions))
        all_values_raw.extend(list(sim_v_raw))
        all_future_scalars.extend(list(s_future_scalars))
        all_tile_yields.extend(list(sim_yields))
        all_future_prices.extend(list(sim_prices))
        all_animal_aux.extend(list(sim_animal_aux))
        all_weights.extend(list(sim_weights))

    # Convert all arrays to NumPy
    print("\nConverting arrays to tensor batches and computing 1001-Bin Two-Hot Symlog targets...")
    grids_arr = np.array(all_grids, dtype=np.float32)
    scalars_arr = np.array(all_scalars, dtype=np.float32)
    actions_arr = np.array(all_actions, dtype=np.int64)
    values_raw_arr = np.array(all_values_raw, dtype=np.float32)
    future_scalars_arr = np.array(all_future_scalars, dtype=np.float32)
    tile_yields_arr = np.array(all_tile_yields, dtype=np.float32)
    future_prices_arr = np.array(all_future_prices, dtype=np.float32)
    animal_aux_arr = np.array(all_animal_aux, dtype=np.float32)
    weights_arr = np.array(all_weights, dtype=np.float32)

    total_samples = len(actions_arr)
    print(f"Total Combined Training Samples: {total_samples:,}")

    # Compute 1001-bin Two-Hot Symlog distributions
    values_tensor = torch.tensor(values_raw_arr, dtype=torch.float32).unsqueeze(1)
    values_1001_arr = scalar_to_two_hot(
        values_tensor, v_min=V_MIN_TEACHER, v_max=V_MAX_TEACHER, num_bins=NUM_BINS_TEACHER
    ).numpy()

    # Save to compressed NPZ
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        grids=grids_arr,
        scalars=scalars_arr,
        actions=actions_arr,
        values_1001=values_1001_arr,
        values_raw=values_raw_arr,
        future_scalars=future_scalars_arr,
        tile_yields=tile_yields_arr,
        future_prices=future_prices_arr,
        animal_aux=animal_aux_arr,
        weights=weights_arr,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    total_time = time.time() - start_time

    # Action distribution
    action_counts = {MACRO_ACTIONS[i]: int(np.sum(actions_arr == i)) for i in range(NUM_MACRO_ACTIONS)}
    action_dist = {name: (count, count / total_samples * 100.0) for name, count in action_counts.items()}

    stats = {
        "total_samples": total_samples,
        "num_top_replays": parsed_replay_count,
        "top_replay_transitions": top_replay_transitions,
        "duration_seconds": total_time,
        "file_size_mb": file_size_mb,
        "output_path": os.path.abspath(output_path),
        "shapes": {
            "grids": list(grids_arr.shape),
            "scalars": list(scalars_arr.shape),
            "actions": list(actions_arr.shape),
            "values_1001": list(values_1001_arr.shape),
            "future_scalars": list(future_scalars_arr.shape),
            "tile_yields": list(tile_yields_arr.shape),
            "future_prices": list(future_prices_arr.shape),
            "animal_aux": list(animal_aux_arr.shape),
            "weights": list(weights_arr.shape),
        },
        "action_distribution": action_dist,
        "score_summary": {
            "min": float(np.min(values_raw_arr)),
            "max": float(np.max(values_raw_arr)),
            "mean": float(np.mean(values_raw_arr)),
            "median": float(np.median(values_raw_arr)),
            "p90": float(np.percentile(values_raw_arr, 90)),
            "pct_ge_100k": float(np.mean(values_raw_arr >= 100000.0) * 100.0),
            "pct_ge_130k": float(np.mean(values_raw_arr >= 130000.0) * 100.0),
        },
        "top_matches": grandmaster_matches[:20],
    }

    stats_json_path = "data/top_leaderboard_teacher_stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print(f" TOP-LEADERBOARD TEACHER DATASET CREATED SUCCESSFULLY ({total_time:.2f}s)")
    print("=" * 80)
    print(f"File Saved      : {output_path} ({file_size_mb:.2f} MB)")
    print(f"Stats JSON      : {stats_json_path}")
    print(f"Total Transitions: {total_samples:,}")
    print(f"Peak Game Score : ${stats['score_summary']['max']:,.1f}")
    print(f"Mean Score      : ${stats['score_summary']['mean']:,.1f}")
    print(f"Pct >= $100k    : {stats['score_summary']['pct_ge_100k']:.2f}%")
    print(f"Pct >= $130k    : {stats['score_summary']['pct_ge_130k']:.2f}%")
    print("=" * 80)

    return stats


if __name__ == "__main__":
    build_top_leaderboard_dataset()
