"""
Top-Leaderboard Teacher Model Multi-Task Training Pipeline (Vectorized & Fast).
Trains TopLeaderboardTeacherNetwork on Grandmaster replays & synthetic transitions:
- Policy Head with Weighted Cross-Entropy Loss
- 1001-Bin Two-Hot Symlog Value Head with Two-Hot Symlog Loss
- SSL World Dynamics Head with MSE Loss
- Auxiliary KataGo Heads (Tile yield, Price forecast, Animal health)
"""

import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from src.models.encoder import MACRO_ACTIONS, NUM_MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS
from src.models.top_leaderboard_teacher import (
    TopLeaderboardTeacherNetwork,
    categorical_to_scalar,
    scalar_to_two_hot,
    V_MIN_TEACHER,
    V_MAX_TEACHER,
    NUM_BINS_TEACHER,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Maximize CPU parallelization
torch.set_num_threads(min(8, os.cpu_count() or 4))


def compute_fast_loss(
    model: nn.Module,
    grids: torch.Tensor,
    scalars: torch.Tensor,
    actions: torch.Tensor,
    targets_1001: torch.Tensor,
    target_future_scalars: torch.Tensor,
    target_tile_yields: torch.Tensor,
    target_future_prices: torch.Tensor,
    target_animal_aux: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, float, float, float]:
    # Multi-task forward pass
    pred_policy_logits, pred_value_logits, pred_future_scalars, pred_tile_yield, pred_prices, pred_animal_aux = model(
        grids, scalars, action_indices=actions
    )

    # 1. Policy Loss: Weighted Cross-Entropy
    ce_loss_raw = F.cross_entropy(pred_policy_logits, actions, reduction="none")
    loss_policy = torch.mean(ce_loss_raw * weights)

    # 2. Value Loss: 1001-Bin Two-Hot Symlog Cross-Entropy
    log_probs_value = F.log_softmax(pred_value_logits, dim=-1)
    val_loss_raw = -torch.sum(targets_1001 * log_probs_value, dim=-1)
    loss_value = torch.mean(val_loss_raw * weights)

    # 3. SSL World Dynamics Loss: MSE on s_{t+4}
    loss_dynamics = F.mse_loss(pred_future_scalars, target_future_scalars)

    # 4. KataGo Auxiliary Heads: Tile Yields, Prices, Animal Aux
    loss_yields = F.mse_loss(pred_tile_yield, target_tile_yields)
    loss_prices = F.mse_loss(pred_prices, target_future_prices)
    loss_animal = F.mse_loss(pred_animal_aux, target_animal_aux)

    # Total multi-task loss
    total_loss = (
        loss_policy
        + 0.5 * loss_value
        + 0.2 * loss_dynamics
        + 0.1 * loss_prices
        + 0.1 * loss_yields
        + 0.05 * loss_animal
    )

    with torch.no_grad():
        pred_actions = torch.argmax(pred_policy_logits, dim=-1)
        acc_top1 = (pred_actions == actions).float().mean().item() * 100.0

    return total_loss, loss_policy.item(), loss_value.item(), acc_top1


def train_teacher_model(
    dataset_path: str = "data/top_leaderboard_teacher_dataset.npz",
    save_checkpoint_path: str = "weights/top_leaderboard_teacher.pt",
    num_epochs: int = 10,
    batch_size: int = 512,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    sample_subsample: int = 150000,
) -> Dict[str, Any]:
    print("=" * 80)
    print(" TRAINING TOP-LEADERBOARD TEACHER POLICY-VALUE NETWORK (FAST CPU)")
    print("=" * 80, flush=True)
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load contiguous tensors directly into RAM
    data = np.load(dataset_path)
    grids_all = torch.from_numpy(data["grids"]).float()
    scalars_all = torch.from_numpy(data["scalars"]).float()
    actions_all = torch.from_numpy(data["actions"]).long()
    values_1001_all = torch.from_numpy(data["values_1001"]).float()
    values_raw_all = torch.from_numpy(data["values_raw"]).float().unsqueeze(1)
    future_scalars_all = torch.from_numpy(data["future_scalars"]).float()
    tile_yields_all = torch.from_numpy(data["tile_yields"]).float()
    future_prices_all = torch.from_numpy(data["future_prices"]).float()
    animal_aux_all = torch.from_numpy(data["animal_aux"]).float()
    weights_all = torch.from_numpy(data["weights"]).float()

    total_len = len(actions_all)
    print(f"Full Dataset in NPZ: {total_len:,} samples", flush=True)

    # Subsample stratified: Keep all 80k Grandmaster replay samples (weights >= 1.0) and subsample the rest
    gm_mask = weights_all >= 1.0
    sim_mask = ~gm_mask

    gm_indices = torch.where(gm_mask)[0]
    sim_indices = torch.where(sim_mask)[0]

    num_sim_keep = min(len(sim_indices), max(50000, sample_subsample - len(gm_indices)))
    sim_keep_indices = sim_indices[torch.randperm(len(sim_indices), generator=torch.Generator().manual_seed(42))[:num_sim_keep]]

    selected_indices = torch.cat([gm_indices, sim_keep_indices])
    selected_indices = selected_indices[torch.randperm(len(selected_indices), generator=torch.Generator().manual_seed(42))]

    total_selected = len(selected_indices)
    val_len = min(15000, int(total_selected * 0.1))
    train_len = total_selected - val_len

    train_idx = selected_indices[:train_len]
    val_idx = selected_indices[train_len:]

    print(f"Subsampled Training Dataset: {total_selected:,} samples ({len(gm_indices):,} Grandmaster replay samples, {num_sim_keep:,} diverse simulation samples)", flush=True)
    print(f"Train samples: {train_len:,} | Validation samples: {val_len:,}", flush=True)

    train_grids = grids_all[train_idx]
    train_scalars = scalars_all[train_idx]
    train_actions = actions_all[train_idx]
    train_values_1001 = values_1001_all[train_idx]
    train_future_scalars = future_scalars_all[train_idx]
    train_tile_yields = tile_yields_all[train_idx]
    train_future_prices = future_prices_all[train_idx]
    train_animal_aux = animal_aux_all[train_idx]
    train_weights = weights_all[train_idx]

    val_grids = grids_all[val_idx]
    val_scalars = scalars_all[val_idx]
    val_actions = actions_all[val_idx]
    val_values_1001 = values_1001_all[val_idx]
    val_values_raw = values_raw_all[val_idx]
    val_future_scalars = future_scalars_all[val_idx]
    val_tile_yields = tile_yields_all[val_idx]
    val_future_prices = future_prices_all[val_idx]
    val_animal_aux = animal_aux_all[val_idx]
    val_weights = weights_all[val_idx]

    model = TopLeaderboardTeacherNetwork(
        in_channels=SPATIAL_CHANNELS,
        scalar_dim=SCALAR_DIM,
        num_actions=NUM_MACRO_ACTIONS,
        hidden_dim=128,
        num_bins=NUM_BINS_TEACHER,
        num_res_blocks=3,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Teacher Model Parameters: {param_count:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_acc = 0.0

    os.makedirs(os.path.dirname(os.path.abspath(save_checkpoint_path)), exist_ok=True)
    num_train_batches = (train_len + batch_size - 1) // batch_size
    num_val_batches = (val_len + batch_size - 1) // batch_size

    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_pol_loss_sum = 0.0
        train_val_loss_sum = 0.0
        train_acc_sum = 0.0

        epoch_perm = torch.randperm(train_len)

        for b in range(num_train_batches):
            start_i = b * batch_size
            end_i = min(start_i + batch_size, train_len)
            batch_idx = epoch_perm[start_i:end_i]

            b_grids = train_grids[batch_idx].to(device)
            b_scalars = train_scalars[batch_idx].to(device)
            b_actions = train_actions[batch_idx].to(device)
            b_values_1001 = train_values_1001[batch_idx].to(device)
            b_future_scalars = train_future_scalars[batch_idx].to(device)
            b_tile_yields = train_tile_yields[batch_idx].to(device)
            b_future_prices = train_future_prices[batch_idx].to(device)
            b_animal_aux = train_animal_aux[batch_idx].to(device)
            b_weights = train_weights[batch_idx].to(device)

            optimizer.zero_grad()
            loss, p_l, v_l, acc = compute_fast_loss(
                model,
                b_grids,
                b_scalars,
                b_actions,
                b_values_1001,
                b_future_scalars,
                b_tile_yields,
                b_future_prices,
                b_animal_aux,
                b_weights,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_pol_loss_sum += p_l
            train_val_loss_sum += v_l
            train_acc_sum += acc

            if (b + 1) % 50 == 0 or (b + 1) == num_train_batches:
                print(
                    f"  Epoch {epoch:2d}/{num_epochs:2d} | Batch [{b+1:3d}/{num_train_batches:3d}] | Loss: {loss.item():.4f} (Pol Acc: {acc:5.1f}%, Val Loss: {v_l:.4f})",
                    flush=True,
                )

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        train_avg_loss = train_loss_sum / num_train_batches
        train_avg_pol = train_pol_loss_sum / num_train_batches
        train_avg_val = train_val_loss_sum / num_train_batches
        train_avg_acc = train_acc_sum / num_train_batches

        # Validation pass
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_top3_sum = 0.0
        val_val_loss_sum = 0.0

        with torch.no_grad():
            for b in range(num_val_batches):
                start_i = b * batch_size
                end_i = min(start_i + batch_size, val_len)

                v_grids = val_grids[start_i:end_i].to(device)
                v_scalars = val_scalars[start_i:end_i].to(device)
                v_actions = val_actions[start_i:end_i].to(device)
                v_values_1001 = val_values_1001[start_i:end_i].to(device)
                v_future_scalars = val_future_scalars[start_i:end_i].to(device)
                v_tile_yields = val_tile_yields[start_i:end_i].to(device)
                v_future_prices = val_future_prices[start_i:end_i].to(device)
                v_animal_aux = val_animal_aux[start_i:end_i].to(device)
                v_weights = val_weights[start_i:end_i].to(device)

                loss, p_l, v_l, acc = compute_fast_loss(
                    model,
                    v_grids,
                    v_scalars,
                    v_actions,
                    v_values_1001,
                    v_future_scalars,
                    v_tile_yields,
                    v_future_prices,
                    v_animal_aux,
                    v_weights,
                )
                val_loss_sum += loss.item()
                val_acc_sum += acc
                val_val_loss_sum += v_l

                # Top 3 accuracy
                logits, _, _, _, _, _ = model(v_grids, v_scalars)
                top3 = torch.topk(logits, k=min(3, NUM_MACRO_ACTIONS), dim=-1).indices
                top3_match = (top3 == v_actions.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
                val_top3_sum += top3_match

        val_avg_loss = val_loss_sum / num_val_batches
        val_avg_acc = val_acc_sum / num_val_batches
        val_avg_top3 = val_top3_sum / num_val_batches
        val_avg_val_loss = val_val_loss_sum / num_val_batches

        ep_duration = time.time() - ep_start

        epoch_record = {
            "epoch": epoch,
            "lr": curr_lr,
            "duration": ep_duration,
            "train_loss": train_avg_loss,
            "train_acc": train_avg_acc,
            "val_loss": val_avg_loss,
            "val_acc_top1": val_avg_acc,
            "val_acc_top3": val_avg_top3,
            "val_val_loss": val_avg_val_loss,
        }
        history.append(epoch_record)

        print(
            f"--> Epoch {epoch:2d}/{num_epochs:2d} Summary ({ep_duration:4.1f}s) | "
            f"Train Loss: {train_avg_loss:.4f} (Acc: {train_avg_acc:5.1f}%) | "
            f"Val Loss: {val_avg_loss:.4f} (Top-1: {val_avg_acc:5.1f}%, Top-3: {val_avg_top3:5.1f}%, ValLoss: {val_avg_val_loss:.4f})",
            flush=True,
        )

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_acc = val_avg_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_avg_loss,
                    "val_acc_top1": val_avg_acc,
                    "val_acc_top3": val_avg_top3,
                    "config": {
                        "in_channels": SPATIAL_CHANNELS,
                        "scalar_dim": SCALAR_DIM,
                        "num_actions": NUM_MACRO_ACTIONS,
                        "hidden_dim": 128,
                        "num_bins": NUM_BINS_TEACHER,
                        "num_res_blocks": 3,
                        "v_min": V_MIN_TEACHER,
                        "v_max": V_MAX_TEACHER,
                    },
                },
                save_checkpoint_path,
            )

    total_training_time = time.time() - start_time
    print("=" * 80, flush=True)
    print(f" TEACHER MODEL TRAINING COMPLETE ({total_training_time:.1f}s)", flush=True)
    print(f" Best Validation Loss: {best_val_loss:.4f} (Best Policy Top-1 Acc: {best_acc:.2f}%)", flush=True)
    print(f" Best Checkpoint Saved: {save_checkpoint_path} ({os.path.getsize(save_checkpoint_path)/(1024*1024):.2f} MB)", flush=True)
    print("=" * 80, flush=True)

    # Save training log JSON
    log_path = "data/top_leaderboard_teacher_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "training_time_seconds": total_training_time,
                "epochs": num_epochs,
                "best_val_loss": best_val_loss,
                "best_val_acc_top1": best_acc,
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "best_val_loss": best_val_loss,
        "best_val_acc_top1": best_acc,
        "training_time_seconds": total_training_time,
        "history": history,
    }


if __name__ == "__main__":
    train_teacher_model(num_epochs=10, batch_size=512, lr=2e-3, sample_subsample=130000)
