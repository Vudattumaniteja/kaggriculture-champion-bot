# Kaggriculture RL & Game-Playing Systems Resources

## Knowledge

- [Paper: "Mastering the game of Go with deep neural networks and tree search" — Silver et al. (2016, Nature)](https://www.nature.com/articles/nature16961)
  AlphaGo seminal paper. Combines expert supervised learning (SL) policy, fast rollout policy, self-play reinforcement learning (RL) policy, value network, and Monte Carlo Tree Search (MCTS). Use for: understanding why SL pretraining + RL fine-tuning + search is the gold standard in board/grid games.

- [Paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" — Silver et al. (2017 / 2018, Science)](https://arxiv.org/abs/1712.01815)
  AlphaZero. Replaces human expert pretraining with pure self-play RL using a single neural network (policy head + value head) guided by MCTS with Dirichlet exploration noise and PUCT action selection. Use for: understanding self-play dynamics, policy-value synchronization, and exploration noise.

- [Paper: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" — Schrittwieser et al. (2020, Nature)](https://www.nature.com/articles/s41586-020-03051-4)
  MuZero. Learns a latent dynamics world model (representation, dynamics, prediction functions) to plan without knowing the environment simulator. Use for: understanding latent world models and value distribution representation.

- [Paper: "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping" — Ng, Harada, & Russell (1999, ICML)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
  The mathematical proof that only Potential-Based Reward Shaping ($F(s, a, s') = \gamma \Phi(s') - \Phi(s)$) preserves the optimal policy. Use for: designing intermediate reward signals (crops, assets, cows) without distorting terminal cash optimization.

- [Paper: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" — Schulman et al. (2015, ICLR)](https://arxiv.org/abs/1506.02438)
  GAE ($\gamma, \lambda$) formulation. Shows how to balance bias and variance in multi-step temporal credit assignment. Use for: propagating rewards across 720 game steps.

- [Paper: "Accelerating Online Reinforcement Learning with Offline Datasets" — Nair et al. (2020, NeurIPS)](https://arxiv.org/abs/2006.09359)
  Advantage-Weighted Actor-Critic (AWAC). Prevents policy collapse when fine-tuning an offline pretrained model with online RL rollouts by penalizing out-of-distribution actions. Use for: fixing our RL policy degradation over the pretrained checkpoint.

- [Paper: "Grandmaster-Level in StarCraft II using Multi-Agent Reinforcement Learning" — Vinyals et al. (2019, Nature)](https://www.nature.com/articles/s41586-019-1724-2)
  AlphaStar. Handles combinatorial action spaces and long horizons using hierarchical macro-micro separation, prioritized league self-play, and pointer networks. Use for: separating high-level economic decisions from low-level worker chore routing.

## Wisdom (Communities)

- [Kaggle Kaggriculture Competition Discussion](https://www.kaggle.com/competitions/kaggriculture/discussion)
  Primary tournament forum. Use for: tracking environment mechanics quirks, opponent distribution shifts, and benchmark scores.
- [RL Discord / Gymnasium Community](https://discord.gg/farama)
  Active RL practitioner community. Use for: debugging GAE implementations, value loss spikes, and off-policy actor-critic stabilization.
