Type: task
Status: resolved
Blocked by: 02, 03, 05

## Question

How should we run batch simulations with heuristic agents to generate 10,000+ expert transitions and train the Policy-Value network via supervised learning to establish a strong warm-start checkpoint?

## Answer

Completed imitation dataset generation and supervised pretraining in `src/training/`:

1. **Dataset Pipeline (`src/training/dataset.py`)**:
   - Simulated 15 multi-agent matches between expert baselines, recording 21,570 transition states `(grid, scalars, macro_action, terminal_value)`.
   - Serialized compressed dataset to `data/imitation_dataset.npz`.

2. **Training & Convergence (`src/training/train.py`)**:
   - Trained `PolicyValueNet` for 10 epochs using AdamW + Cosine Annealing with joint CrossEntropy policy loss + MSE value loss.
   - **Performance**:
     - Policy Accuracy: **99.5% Train / 99.2% Validation Accuracy**.
     - Validation Loss: $0.0352$.
   - Checkpoint saved to `weights/pretrained_alphazero.pt`.
