Type: prototype
Status: resolved
Blocked by: 04

## Question

What is the optimal dual-head Policy-Value neural network architecture ($\pi(a|s), v(s)$) that balances high representation capacity with CPU inference speed ($< 15\text{ms}$ per evaluation)?

## Answer

Implemented `PolicyValueNet` in `src/models/network.py`:

1. **Architecture Specifications**:
   - **Spatial CNN Stream**: 3 Conv2d layers ($32 \to 64 \to 64$ filters with BatchNorm + ReLU) processing the $(11, 10, 10)$ grid $\to 128$-dim dense projection.
   - **Economic Scalar Stream**: 2-layer MLP ($32 \to 64 \to 64$) processing global market/inventory features.
   - **Fusion Trunk**: LayerNorm + 2-layer dense trunk ($192 \to 128 \to 128$) merging spatial and economic streams.
   - **Policy Head**: Outputs logits for 9 macro-actions.
   - **Value Head**: $128 \to 64 \to 1$ with `Tanh` activation producing scalar value $v(s) \in [-1, 1]$.
2. **Benchmarking & Latency Verification**:
   - Total Parameters: 943,178 (~3.60 MB total model weight size).
   - CPU Inference Latency: **0.86 ms per evaluation** (enabling 100+ MCTS rollouts per turn well within Kaggle's 1.0s limit).
