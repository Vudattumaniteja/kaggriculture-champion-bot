"""
Supervised Pretraining loop for AlphaZero Policy-Value Network.
Trains the network on expert imitation datasets with joint Policy CrossEntropy + Value MSE loss.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.models.network import PolicyValueNet
from src.training.dataset import ExpertImitationDataset, generate_imitation_dataset


def train_policy_value_net(
    dataset_path: str = "data/imitation_dataset.npz",
    save_weights_path: str = "weights/pretrained_alphazero.pt",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    value_loss_weight: float = 1.0,
    generate_if_missing: bool = True,
    num_episodes_to_generate: int = 15,
) -> PolicyValueNet:
    os.makedirs(os.path.dirname(os.path.abspath(save_weights_path)), exist_ok=True)

    if not os.path.exists(dataset_path) and generate_if_missing:
        print(f"Dataset not found at {dataset_path}. Generating from expert simulations...")
        generate_imitation_dataset(num_episodes=num_episodes_to_generate, save_path=dataset_path)

    full_dataset = ExpertImitationDataset(dataset_path)
    total_samples = len(full_dataset)

    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"=== Starting Training ({total_samples:,} samples: {train_size:,} train, {val_size:,} val) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = PolicyValueNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_policy_loss = 0.0
        train_value_loss = 0.0
        correct_policy = 0
        total_train_samples = 0

        t0 = time.time()
        for grids, scalars, actions, values in train_loader:
            grids = grids.to(device)
            scalars = scalars.to(device)
            actions = actions.to(device)
            values = values.to(device)

            optimizer.zero_grad()
            policy_logits, pred_values = model(grids, scalars)

            loss_p = ce_criterion(policy_logits, actions)
            loss_v = mse_criterion(pred_values, values)
            loss = loss_p + value_loss_weight * loss_v

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * grids.size(0)
            train_policy_loss += loss_p.item() * grids.size(0)
            train_value_loss += loss_v.item() * grids.size(0)

            preds = policy_logits.argmax(dim=-1)
            correct_policy += (preds == actions).sum().item()
            total_train_samples += grids.size(0)

        scheduler.step()
        train_acc = (correct_policy / total_train_samples) * 100

        # Validation
        model.eval()
        val_loss = 0.0
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for grids, scalars, actions, values in val_loader:
                grids = grids.to(device)
                scalars = scalars.to(device)
                actions = actions.to(device)
                values = values.to(device)

                policy_logits, pred_values = model(grids, scalars)
                loss_p = ce_criterion(policy_logits, actions)
                loss_v = mse_criterion(pred_values, values)
                loss = loss_p + value_loss_weight * loss_v

                val_loss += loss.item() * grids.size(0)
                val_policy_loss += loss_p.item() * grids.size(0)
                val_value_loss += loss_v.item() * grids.size(0)

                preds = policy_logits.argmax(dim=-1)
                val_correct += (preds == actions).sum().item()
                total_val_samples += grids.size(0)

        val_acc = (val_correct / total_val_samples) * 100 if total_val_samples > 0 else 0.0
        epoch_time = time.time() - t0

        avg_train_loss = train_loss / total_train_samples
        avg_val_loss = val_loss / total_val_samples

        print(
            f"Epoch {epoch:2d}/{epochs} [{epoch_time:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} (Pol Acc: {train_acc:.1f}%) | "
            f"Val Loss: {avg_val_loss:.4f} (Val Acc: {val_acc:.1f}%)"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_weights_path)

    print(f"=== Training Complete: Best Checkpoint Saved to {save_weights_path} ===")
    return model


if __name__ == "__main__":
    train_policy_value_net(epochs=10, num_episodes_to_generate=15)
