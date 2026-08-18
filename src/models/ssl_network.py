"""
Self-Supervised Policy-Value & Dynamics Neural Network (SSLPolicyValueNet).
Integrates Policy Head, Value Head, and Self-Supervised World-Dynamics Predictor Head.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import NUM_MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS


class SSLPolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = SPATIAL_CHANNELS,
        scalar_dim: int = SCALAR_DIM,
        num_actions: int = NUM_MACRO_ACTIONS,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_actions = num_actions

        # 1. Spatial CNN Trunk (Processes 10x10 farm grid)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.spatial_fc = nn.Linear(64 * 10 * 10, hidden_dim)

        # 2. Economic Scalar Trunk (Processes 32 global features)
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 3. Latent Fusion Trunk
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Head 1: Policy Head (Macro-action distribution)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        # Head 2: Value Head (Predicted match outcome in [-1, 1])
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        # Head 3: SELF-SUPERVISED DYNAMICS HEAD (World Model Predictor)
        # Predicts future economic scalars (prices, cash, inventory) given current state + action
        self.ssl_dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim + num_actions, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, scalar_dim),
        )

    def extract_features(self, grid: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Extracts the shared 128-dim latent representation."""
        x_grid = F.relu(self.bn1(self.conv1(grid)))
        x_grid = F.relu(self.bn2(self.conv2(x_grid)))
        x_grid = F.relu(self.bn3(self.conv3(x_grid)))
        x_spatial = F.relu(self.spatial_fc(x_grid.view(x_grid.size(0), -1)))

        x_scalar = self.scalar_fc(scalars)
        return self.fusion(torch.cat([x_spatial, x_scalar], dim=1))

    def forward(
        self,
        grid: torch.Tensor,
        scalars: torch.Tensor,
        action_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            grid: (B, 11, 10, 10)
            scalars: (B, 32)
            action_indices: Optional (B,) long tensor of macro action taken
        Returns:
            policy_logits: (B, 10)
            value: (B, 1)
            pred_future_scalars: Optional (B, 32) future economic state prediction
        """
        latent = self.extract_features(grid, scalars)

        policy_logits = self.policy_head(latent)
        value = self.value_head(latent)

        pred_future_scalars = None
        if action_indices is not None:
            # One-hot encode action
            action_one_hot = F.one_hot(action_indices, num_classes=self.num_actions).float()
            dynamics_input = torch.cat([latent, action_one_hot], dim=1)
            pred_future_scalars = self.ssl_dynamics_head(dynamics_input)

        return policy_logits, value, pred_future_scalars

    @torch.no_grad()
    def predict(
        self, grid: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast inference returning softmax probabilities and value."""
        self.eval()
        latent = self.extract_features(grid, scalars)
        logits = self.policy_head(latent)
        value = self.value_head(latent)
        probs = F.softmax(logits, dim=-1)
        return probs, value
