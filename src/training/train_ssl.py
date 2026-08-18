"""
Grandmaster Self-Supervised Learning (SSL) & Policy-Value Training Loop for Kaggriculture.
Trains SSLPolicyValueNet on curated multi-industry trajectories with elite filtering & sample prioritization.
Jointly optimizes:
1. Macro-Action Policy Cross-Entropy Loss
2. Value MSE Loss (Game Outcome Prediction)
3. Self-Supervised World-Dynamics Future Scalar MSE Loss
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.ssl_network import SSLPolicyValueNet
from src.models.encoder import NUM_MACRO_ACTIONS, MACRO_ACTIONS


class SSLMultiIndustryDataset(Dataset):
    """
    Dataset loader for massive multi-industry trajectories with SSL future-state pairs.
    Supports precomputed future_scalars or dynamic horizon calculation.
    """
    def __init__(self, npz_path: str = "data/massive_ssl_dataset.npz", future_horizon: int = 4):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset not found at {npz_path}")

        data = np.load(npz_path)
        raw_grids = data["grids"]
        raw_scalars = data["scalars"]
        raw_actions = data["actions"]
        raw_values = data["values"]

        n_samples = len(raw_actions)

        # Use precomputed future_scalars if present in NPZ, otherwise generate with episode boundaries
        if "future_scalars" in data:
            raw_future_scalars = data["future_scalars"]
        else:
            raw_future_scalars = np.zeros_like(raw_scalars)
            for i in range(n_samples):
                ep_offset = i % 719
                target_ep_offset = min(718, ep_offset + future_horizon)
                target_idx = min(n_samples - 1, (i - ep_offset) + target_ep_offset)
                raw_future_scalars[i] = raw_scalars[target_idx]

        self.grids = torch.tensor(raw_grids, dtype=torch.float32)
        self.scalars = torch.tensor(raw_scalars, dtype=torch.float32)
        self.actions = torch.tensor(raw_actions, dtype=torch.long)
        self.values = torch.tensor(raw_values, dtype=torch.float32).unsqueeze(1)
        self.future_scalars = torch.tensor(raw_future_scalars, dtype=torch.float32)

        self.raw_scalars_np = raw_scalars
        self.raw_values_np = raw_values

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.grids[idx],
            self.scalars[idx],
            self.actions[idx],
            self.values[idx],
            self.future_scalars[idx],
        )


def train_ssl_model(
    dataset_path: str = "data/massive_ssl_dataset.npz",
    save_weights_path: str = "weights/grandmaster_ssl.pt",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_split: float = 0.15,
    value_weight: float = 1.0,
    ssl_weight: float = 0.5,
    elite_multiplier: float = 3.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains SSLPolicyValueNet with high-score sample prioritization / elite filtering.
    Saves best model checkpoint and returns convergence metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_dir = os.path.dirname(os.path.abspath(save_weights_path))
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print(f" Grandmaster SSL Policy-Value & Dynamics Neural Training")
    print(f" Dataset Path: {dataset_path}")
    print(f" Target Weights Path: {save_weights_path}")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr} (Cosine Annealed)")
    print(f" Loss Weights -> Policy: 1.0, Value MSE: {value_weight}, SSL Dynamics MSE: {ssl_weight}")
    print(f" Elite Prioritization Multiplier: {elite_multiplier}x")
    print("=" * 80)

    dataset = SSLMultiIndustryDataset(dataset_path)
    total_samples = len(dataset)
    ep_len = 719
    n_runs = total_samples // ep_len

    # Trajectory-level run analysis
    run_final_cash = dataset.raw_scalars_np[ep_len - 1 :: ep_len, 0]  # min(money/10000, 1.0)
    run_vals = dataset.raw_values_np[::ep_len]  # tanh outcome

    elite_run_mask = (run_final_cash >= 1.0)
    elite_runs = np.where(elite_run_mask)[0]
    non_elite_runs = np.where(~elite_run_mask)[0]

    print(
        f"Total Samples: {total_samples:,} across {n_runs} agent trajectories | "
        f"Elite Runs ($10k-$23.7k): {len(elite_runs)} ({len(elite_runs)/n_runs*100:.1f}%)"
    )

    # Stratified split by trajectory
    rng = np.random.RandomState(seed)
    shuffled_elite = rng.permutation(elite_runs)
    shuffled_non_elite = rng.permutation(non_elite_runs)

    n_val_elite = max(1, int(len(elite_runs) * val_split))
    n_val_non_elite = max(1, int(len(non_elite_runs) * val_split))

    val_run_indices = np.concatenate([shuffled_elite[:n_val_elite], shuffled_non_elite[:n_val_non_elite]])
    train_run_indices = np.concatenate([shuffled_elite[n_val_elite:], shuffled_non_elite[n_val_non_elite:]])

    train_indices: List[int] = []
    train_sample_weights: List[float] = []

    for r_idx in train_run_indices:
        start_i = r_idx * ep_len
        end_i = start_i + ep_len
        train_indices.extend(range(start_i, end_i))

        # Elite sample prioritization
        cash = run_final_cash[r_idx]
        v = run_vals[r_idx]
        if cash >= 1.0:  # Elite trajectory ($10,000 - $23,759 scores)
            w = elite_multiplier
        elif v > 0 and cash >= 0.7:  # High-profit winning game
            w = 2.0
        elif cash >= 0.5:  # Above average profit
            w = 1.0
        else:  # Suboptimal / losing run
            w = 0.35

        train_sample_weights.extend([w] * ep_len)

    val_indices: List[int] = []
    for r_idx in val_run_indices:
        start_i = r_idx * ep_len
        end_i = start_i + ep_len
        val_indices.extend(range(start_i, end_i))

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Weighted random sampler for training to prioritize high-score / elite transitions
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | Train Samples: {len(train_indices):,} ({len(train_run_indices)} runs) | "
        f"Val Samples: {len(val_indices):,} ({len(val_run_indices)} runs)"
    )

    # Initialize SSL Network
    model = SSLPolicyValueNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_metrics = {}
    history = []

    print("\n" + "=" * 105)
    print(
        f"{'Epoch':^7} | {'Train Loss':^10} | {'Train Pol Acc':^13} | {'Train V-MSE':^11} | {'Train SSL-MSE':^13} | "
        f"{'Val Loss':^10} | {'Val Pol Acc':^11} | {'Val V-MSE':^9} | {'Val SSL-MSE':^11} | {'Time':^7}"
    )
    print("=" * 105)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        train_total_loss = 0.0
        train_pol_loss = 0.0
        train_val_loss = 0.0
        train_ssl_loss = 0.0
        train_correct = 0
        train_count = 0

        for grids, scalars, actions, values, future_scalars in train_loader:
            grids = grids.to(device)
            scalars = scalars.to(device)
            actions = actions.to(device)
            values = values.to(device)
            future_scalars = future_scalars.to(device)

            optimizer.zero_grad()

            pol_logits, pred_val, pred_future = model(grids, scalars, action_indices=actions)

            loss_p = ce_loss_fn(pol_logits, actions)
            loss_v = mse_loss_fn(pred_val, values)
            loss_ssl = mse_loss_fn(pred_future, future_scalars)

            batch_loss = loss_p + value_weight * loss_v + ssl_weight * loss_ssl

            batch_loss.backward()
            optimizer.step()

            bs = grids.size(0)
            train_total_loss += batch_loss.item() * bs
            train_pol_loss += loss_p.item() * bs
            train_val_loss += loss_v.item() * bs
            train_ssl_loss += loss_ssl.item() * bs

            preds = pol_logits.argmax(dim=-1)
            train_correct += (preds == actions).sum().item()
            train_count += bs

        scheduler.step()

        avg_train_loss = train_total_loss / train_count
        avg_train_pol_acc = (train_correct / train_count) * 100.0
        avg_train_val_mse = train_val_loss / train_count
        avg_train_ssl_mse = train_ssl_loss / train_count

        # Validation Phase
        model.eval()
        val_total_loss = 0.0
        val_pol_loss = 0.0
        val_val_loss = 0.0
        val_ssl_loss = 0.0
        val_correct = 0
        val_count = 0

        with torch.no_grad():
            for grids, scalars, actions, values, future_scalars in val_loader:
                grids = grids.to(device)
                scalars = scalars.to(device)
                actions = actions.to(device)
                values = values.to(device)
                future_scalars = future_scalars.to(device)

                pol_logits, pred_val, pred_future = model(grids, scalars, action_indices=actions)

                loss_p = ce_loss_fn(pol_logits, actions)
                loss_v = mse_loss_fn(pred_val, values)
                loss_ssl = mse_loss_fn(pred_future, future_scalars)

                batch_loss = loss_p + value_weight * loss_v + ssl_weight * loss_ssl

                bs = grids.size(0)
                val_total_loss += batch_loss.item() * bs
                val_pol_loss += loss_p.item() * bs
                val_val_loss += loss_v.item() * bs
                val_ssl_loss += loss_ssl.item() * bs

                preds = pol_logits.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()
                val_count += bs

        avg_val_loss = val_total_loss / val_count if val_count > 0 else 0.0
        avg_val_pol_acc = (val_correct / val_count) * 100.0 if val_count > 0 else 0.0
        avg_val_val_mse = val_val_loss / val_count if val_count > 0 else 0.0
        avg_val_ssl_mse = val_ssl_loss / val_count if val_count > 0 else 0.0

        epoch_duration = time.time() - epoch_start

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_pol_acc": avg_train_pol_acc,
            "train_val_mse": avg_train_val_mse,
            "train_ssl_mse": avg_train_ssl_mse,
            "val_loss": avg_val_loss,
            "val_pol_acc": avg_val_pol_acc,
            "val_val_mse": avg_val_val_mse,
            "val_ssl_mse": avg_val_ssl_mse,
            "epoch_duration": epoch_duration,
        }
        history.append(epoch_metrics)

        saved_marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_pol_acc
            best_metrics = epoch_metrics.copy()
            torch.save(model.state_dict(), save_weights_path)
            saved_marker = " *"

        print(
            f"{epoch:3d}/{epochs:2d}  | "
            f"{avg_train_loss:10.4f} | "
            f"{avg_train_pol_acc:12.2f}% | "
            f"{avg_train_val_mse:11.4f} | "
            f"{avg_train_ssl_mse:13.5f} | "
            f"{avg_val_loss:10.4f} | "
            f"{avg_val_pol_acc:10.2f}% | "
            f"{avg_val_val_mse:9.4f} | "
            f"{avg_val_ssl_mse:11.5f} | "
            f"{epoch_duration:5.1f}s{saved_marker}"
        )

    print("=" * 105)
    print(f"\n Training Complete! Best Grandmaster Checkpoint Saved: {save_weights_path}")
    print(
        f" Best Validation Metrics (Epoch {best_metrics.get('epoch', epochs)}):\n"
        f"   - Val Total Loss        : {best_metrics.get('val_loss', 0.0):.4f}\n"
        f"   - Policy Accuracy       : {best_metrics.get('val_pol_acc', 0.0):.2f}%\n"
        f"   - Value MSE Loss        : {best_metrics.get('val_val_mse', 0.0):.4f}\n"
        f"   - SSL Dynamics MSE Loss : {best_metrics.get('val_ssl_mse', 0.0):.5f}"
    )

    return {
        "best_metrics": best_metrics,
        "history": history,
        "model": model,
        "save_path": os.path.abspath(save_weights_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Grandmaster SSL Policy-Value & Dynamics Model.")
    parser.add_argument("--dataset", type=str, default="data/massive_ssl_dataset.npz", help="Dataset NPZ path")
    parser.add_argument("--save-weights", type=str, default="weights/grandmaster_ssl.pt", help="Checkpoint save path")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--elite-mult", type=float, default=3.5, help="Sampling weight multiplier for elite runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_ssl_model(
        dataset_path=args.dataset,
        save_weights_path=args.save_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        elite_multiplier=args.elite_mult,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
