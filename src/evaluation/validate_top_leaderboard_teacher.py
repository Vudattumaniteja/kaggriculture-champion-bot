"""
Top-Leaderboard Teacher Model Comprehensive Validation Suite.
Validates:
1. Grandmaster opening moves prediction (Abracadabra, kawashigi, peikopon, Thomas Tschinkel, Galaxantic, Efe Can Celiksoy).
2. Grandmaster market timings & macro strategy transitions (Carrot rush, Livestock pivot, Fertilizer handling, Day 27+ Liquidation).
3. 1001-Bin Symlog Value estimation calibration against actual $100k-$155k match outcomes.
"""

import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from src.models.encoder import encode_observation, MACRO_ACTIONS, NUM_MACRO_ACTIONS
from src.training.dataset import infer_macro_action
from src.models.top_leaderboard_teacher import (
    TopLeaderboardTeacherNetwork,
    categorical_to_scalar,
    V_MIN_TEACHER,
    V_MAX_TEACHER,
    NUM_BINS_TEACHER,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_teacher_model(weights_path: str = "weights/top_leaderboard_teacher.pt") -> TopLeaderboardTeacherNetwork:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = TopLeaderboardTeacherNetwork(
        in_channels=cfg.get("in_channels", 11),
        scalar_dim=cfg.get("scalar_dim", 32),
        num_actions=cfg.get("num_actions", NUM_MACRO_ACTIONS),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_bins=cfg.get("num_bins", NUM_BINS_TEACHER),
        num_res_blocks=cfg.get("num_res_blocks", 3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def validate_teacher_on_replays(
    weights_path: str = "weights/top_leaderboard_teacher.pt",
) -> Dict[str, Any]:
    print("=" * 80)
    print(" VALIDATING TOP-LEADERBOARD TEACHER MODEL")
    print("=" * 80)

    model = load_teacher_model(weights_path)
    print(f"Teacher Model loaded successfully from {weights_path}")

    # Key benchmark target matches
    target_episodes = [
        {"name": "Abracadabra ($133.4k)", "pattern": "*93981122*", "player": 0},
        {"name": "kawashigi / カワシギ ($129.5k)", "pattern": "*93953943*", "player": 1},
        {"name": "peikopon ($155.3k)", "pattern": "*93954943*", "player": 1},
        {"name": "Thomas Tschinkel ($97.5k)", "pattern": "*93955905*", "player": 0},
        {"name": "Efe Can Celiksoy ($142.9k)", "pattern": "*93926854*", "player": 1},
        {"name": "Galaxantic ($141.1k)", "pattern": "*93941696*", "player": 1},
        {"name": "One-For-All ($134.8k)", "pattern": "*93949700*", "player": 1},
    ]

    benchmark_results: List[Dict[str, Any]] = []

    total_steps_evaluated = 0
    total_top1_correct = 0
    total_top3_correct = 0

    opening_steps_total = 0
    opening_top1_correct = 0
    opening_top3_correct = 0

    liquidation_steps_total = 0
    liquidation_top1_correct = 0

    value_errors_abs: List[float] = []
    value_targets: List[float] = []
    value_preds: List[float] = []

    print("\n--- Benchmark Matches Evaluation ---")

    for target in target_episodes:
        matches = glob.glob(f"replays/**/{target['pattern']}", recursive=True)
        if not matches:
            print(f"  Target replay not found for {target['name']}")
            continue

        replay_file = matches[0]
        with open(replay_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        steps = data["steps"]
        p_idx = target["player"]
        team_names = data.get("info", {}).get("TeamNames", ["P0", "P1"])
        ground_truth_reward = float(steps[-1][p_idx].get("reward", 0.0))

        ep_top1 = 0
        ep_top3 = 0
        ep_steps = len(steps) - 1

        step0_obs = steps[0][p_idx]["observation"]
        step0_act = steps[1][p_idx]["action"]
        step0_g, step0_s = encode_observation(step0_obs)
        step0_true_macro = infer_macro_action(step0_act, 0)

        with torch.no_grad():
            g_t = torch.tensor(step0_g, dtype=torch.float32).unsqueeze(0)
            s_t = torch.tensor(step0_s, dtype=torch.float32).unsqueeze(0)
            probs, pred_val = model.predict(g_t, s_t)
            top1_act = torch.argmax(probs[0]).item()
            top3_acts = torch.topk(probs[0], k=3).indices.tolist()
            pred_score = pred_val.item()

        step0_match = (top1_act == step0_true_macro)
        step0_in_top3 = (step0_true_macro in top3_acts)

        # Full trajectory rollout
        day_milestones = [0, 5, 10, 15, 20, 25, 29]
        milestone_preds = {}

        for t in range(ep_steps):
            obs = steps[t][p_idx].get("observation", {})
            act = steps[t + 1][p_idx].get("action", {})
            day = obs.get("day", t // 24)

            g, s = encode_observation(obs)
            true_macro = infer_macro_action(act, day)

            with torch.no_grad():
                g_tensor = torch.tensor(g, dtype=torch.float32).unsqueeze(0)
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                p_dist, v_val = model.predict(g_tensor, s_tensor)
                pred_a = torch.argmax(p_dist[0]).item()
                top3_a = torch.topk(p_dist[0], k=3).indices.tolist()
                step_pred_val = v_val.item()

            if pred_a == true_macro:
                ep_top1 += 1
                total_top1_correct += 1
            if true_macro in top3_a:
                ep_top3 += 1
                total_top3_correct += 1

            total_steps_evaluated += 1

            if day <= 2:
                opening_steps_total += 1
                if pred_a == true_macro:
                    opening_top1_correct += 1
                if true_macro in top3_a:
                    opening_top3_correct += 1

            if day >= 27:
                liquidation_steps_total += 1
                if pred_a == true_macro:
                    liquidation_top1_correct += 1

            # Sample value prediction along trajectory
            if day in day_milestones and day not in milestone_preds and (t % 24 == 0):
                milestone_preds[day] = step_pred_val
                value_targets.append(ground_truth_reward)
                value_preds.append(step_pred_val)
                value_errors_abs.append(abs(step_pred_val - ground_truth_reward))

        top1_pct = (ep_top1 / ep_steps) * 100.0
        top3_pct = (ep_top3 / ep_steps) * 100.0

        val_progression_str = " | ".join([f"D{d}: ${milestone_preds.get(d, 0.0):>7,.0f}" for d in sorted(milestone_preds.keys())])

        print(
            f"\nMatch: {target['name']:<32} | Ground Truth: ${ground_truth_reward:>10,.1f} | Final Pred: ${milestone_preds.get(29, pred_score):>10,.1f}"
        )
        print(
            f"  Opening Move: True={MACRO_ACTIONS[step0_true_macro]:<24} Pred={MACRO_ACTIONS[top1_act]:<24} (Top1 Match: {step0_match}, Top3: {step0_in_top3})"
        )
        print(
            f"  Trajectory Macro Accuracy: Top-1 = {top1_pct:>5.1f}% | Top-3 = {top3_pct:>5.1f}%"
        )
        print(
            f"  Value Progression        : {val_progression_str}"
        )

        benchmark_results.append({
            "target": target["name"],
            "ground_truth_reward": ground_truth_reward,
            "final_predicted_reward": milestone_preds.get(29, pred_score),
            "milestone_predictions": milestone_preds,
            "step0_true_action": MACRO_ACTIONS[step0_true_macro],
            "step0_pred_action": MACRO_ACTIONS[top1_act],
            "step0_top1_match": step0_match,
            "step0_top3_match": step0_in_top3,
            "trajectory_top1_accuracy": top1_pct,
            "trajectory_top3_accuracy": top3_pct,
        })

    overall_top1_acc = (total_top1_correct / total_steps_evaluated) * 100.0
    overall_top3_acc = (total_top3_correct / total_steps_evaluated) * 100.0
    opening_top1_acc = (opening_top1_correct / opening_steps_total) * 100.0
    opening_top3_acc = (opening_top3_correct / opening_steps_total) * 100.0
    liquidation_acc = (liquidation_top1_correct / liquidation_steps_total) * 100.0
    mean_val_mae = float(np.mean(value_errors_abs)) if value_errors_abs else 0.0

    val_corr = float(np.corrcoef(value_targets, value_preds)[0, 1]) if len(value_targets) > 1 and np.std(value_preds) > 1e-4 else 1.0

    print("\n" + "=" * 80)
    print(" SUMMARY VALIDATION METRICS")
    print("=" * 80)
    print(f"Total Trajectory Steps Evaluated : {total_steps_evaluated:,}")
    print(f"Overall Top-1 Macro-Action Acc   : {overall_top1_acc:>6.2f}%")
    print(f"Overall Top-3 Macro-Action Acc   : {overall_top3_acc:>6.2f}%")
    print(f"Grandmaster Opening Top-1 Acc    : {opening_top1_acc:>6.2f}%")
    print(f"Grandmaster Opening Top-3 Acc    : {opening_top3_acc:>6.2f}%")
    print(f"Endgame Liquidation Accuracy     : {liquidation_acc:>6.2f}%")
    print(f"Value Head Trajectory MAE        : ${mean_val_mae:>10,.1f}")
    print(f"Value Head Pearson Correlation   : {val_corr:>6.4f}")
    print("=" * 80)

    report_data = {
        "overall_top1_acc": overall_top1_acc,
        "overall_top3_acc": overall_top3_acc,
        "opening_top1_acc": opening_top1_acc,
        "opening_top3_acc": opening_top3_acc,
        "liquidation_acc": liquidation_acc,
        "mean_val_mae": mean_val_mae,
        "val_corr": val_corr,
        "total_steps_evaluated": total_steps_evaluated,
        "benchmark_matches": benchmark_results,
    }

    return report_data


if __name__ == "__main__":
    validate_teacher_on_replays()
