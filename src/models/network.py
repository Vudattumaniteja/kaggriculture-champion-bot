"""
Dual-Head Policy-Value Neural Network Architecture for AlphaZero / MCTS.
Optimized for high representational capacity and sub-5ms CPU inference latency.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import NUM_MACRO_ACTIONS, SCALAR_DIM, SPATIAL_CHANNELS


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = SPATIAL_CHANNELS,
        scalar_dim: int = SCALAR_DIM,
        num_actions: int = NUM_MACRO_ACTIONS,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Spatial CNN Trunk (Processes 10x10 farm grid)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Spatial pooling: 64 x 10 x 10 -> 64 x 5 x 5 (1600 dims) -> 128 dims
        self.spatial_fc = nn.Linear(64 * 10 * 10, hidden_dim)

        # Economic Scalar Trunk (Processes 32 global features)
        self.scalar_fc = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fusion Trunk (Combines spatial + scalar features)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 1. Policy Head: Outputs logits for macro-actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

        # 2. Value Head: Outputs scalar expected outcome v(s) in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self, grid: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            grid: (B, 11, 10, 10) tensor
            scalars: (B, 32) tensor
        Returns:
            policy_logits: (B, NUM_MACRO_ACTIONS)
            value: (B, 1) in [-1.0, 1.0]
        """
        # Spatial conv stream
        x_grid = F.relu(self.bn1(self.conv1(grid)))
        x_grid = F.relu(self.bn2(self.conv2(x_grid)))
        x_grid = F.relu(self.bn3(self.conv3(x_grid)))
        x_spatial = F.relu(self.spatial_fc(x_grid.view(x_grid.size(0), -1)))

        # Scalar stream
        x_scalar = self.scalar_fc(scalars)

        # Fusion
        fused = self.fusion(torch.cat([x_spatial, x_scalar], dim=1))

        # Heads
        logits = self.policy_head(fused)
        value = self.value_head(fused)

        return logits, value

    @torch.no_grad()
    def predict(
        self, grid: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference mode returning softmax probabilities and value."""
        self.eval()
        logits, value = self.forward(grid, scalars)
        probs = F.softmax(logits, dim=-1)
        return probs, value
