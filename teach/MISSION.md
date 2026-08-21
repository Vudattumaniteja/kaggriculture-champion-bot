# Mission: Mastering Reinforcement Learning Architectures & Reward Design for Kaggriculture

## Why
Understand why the current RL training pipeline produces a bot earning less cash than the supervised pretrained checkpoint, master the architectural paradigms of game-playing systems (AlphaGo, AlphaZero, MuZero, AWAC), and determine the optimal RL algorithm and reward structure to maximize terminal cash in Kaggriculture.

## Success looks like
- Pinpointing exactly why the current RL reward signals and training loop degrade pretrained weights.
- Explaining the differences between AlphaGo, AlphaZero, MuZero, AWIL, and PPO with respect to large action spaces and hierarchical control.
- Selecting and implementing the optimal RL training regime (objective, value target, policy regularization, and micro-solver integration) that reliably scales beyond $35,000 terminal cash.

## Constraints
- Kaggriculture environment: 720 turns, 10x10 grid, 4 quadrants, combinatorial action space (spatial planting, hiring, expansion, market trading, livestock, micro chore routing).
- High execution speed requirements for Kaggle submissions (under 1 second per step).

## Out of scope
- Vision-only RL (Atari pixel inputs) or continuous physics robotics engines (MuJoCo). Focus strictly on turn-based grid and discrete/hierarchical economic strategy environments.
