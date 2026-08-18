## Destination

A high-performing, submission-ready AlphaZero / MCTS + Supervised Pretrained agent (`submission.py`) for the Kaggle Kaggriculture simulation competition, supported by a fast local tournament evaluation harness, baseline bots, and a lightweight neural forward model.

## Notes

- Domain: Agricultural & Economic Multi-Agent Turn-Based Simulation (Kaggle Kaggriculture).
- Skills: `/grilling`, `/domain-modeling`, `/research`, `/prototype`.
- Standing Preferences:
  - AlphaZero-style architecture (MCTS + Policy/Value Network).
  - Supervised warm-start / imitation learning from heuristic baselines to ensure early economic viability.
  - Hierarchical action decomposition (macro-economic allocations down to low-level deterministic chore queues).
  - Standalone, CPU-optimized submission ($< 1$s per step).

## Decisions so far

<!-- the index — one line per closed ticket -->

- [01-research-environment-rules-and-mechanics](./issues/01-research-environment-rules-and-mechanics.md) — Exhaustive investigation of the 720-turn Kaggriculture engine: 4 quadrants, 5 crop cycles, livestock husbandry & care multipliers, dynamic commodity price elasticity, Fibonacci labor costs, and 1.0s turn limits.
- [02-local-simulation-and-evaluation-harness](./issues/02-local-simulation-and-evaluation-harness.md) — Implemented head-to-head match runner and multi-game tournament evaluator in `src/evaluation/runner.py` with CLI `evaluate.py` and HTML/JSON replay export.
- [03-heuristic-baseline-agents](./issues/03-heuristic-baseline-agents.md) — Built `MultiTileCropAgent`, `LivestockHusbandryAgent`, and `HybridExpertAgent`. `MultiTileCropAgent` achieved 100% win rate (10-0) against `starter` with $7,268 mean score (max $12,034).
- [04-action-abstraction-and-state-encoder](./issues/04-action-abstraction-and-state-encoder.md) — Implemented observation encoder producing (11, 10, 10) spatial grid tensors + (32,) economic vectors, and defined the 9-action macro-economic action vocabulary.
- [05-policy-value-network-architecture](./issues/05-policy-value-network-architecture.md) — Implemented `PolicyValueNet` with 943k parameters (3.6 MB), dual policy & value heads, and sub-millisecond (0.86ms) CPU inference speed.
- [06-fast-forward-model-for-mcts](./issues/06-fast-forward-model-for-mcts.md) — Designed lightweight PUCT search engine evaluated by PolicyValueNet with zero external simulation overhead.
- [07-imitation-dataset-generation-and-pretraining](./issues/07-imitation-dataset-generation-and-pretraining.md) — Generated 21,570 expert transitions and trained `PolicyValueNet` achieving 99.5% policy accuracy, saving weights to `weights/pretrained_alphazero.pt`.
- [08-alphazero-mcts-self-play-fine-tuning](./issues/08-alphazero-mcts-self-play-fine-tuning.md) — Integrated MCTS search with PolicyValueNet, achieving 80% win rate against `starter` in ~4.1ms/turn.
- [09-submission-packaging-and-validation](./issues/09-submission-packaging-and-validation.md) — Created standalone `submission.py`, achieving a 100% win rate (5-0) against `starter` with $9,890 mean score (peak $14,496) at ~2.0ms/turn.

## Not yet specified

- Multi-agent market game theory modeling (predicting competitor dumping vs. price holding).
- Dynamic MCTS rollout budget allocation (spending more search budget on crucial investment turns like land purchases).
- Weight quantization scheme if submission exceeds 10MB file threshold.

## Out of scope

- Direct visual pixel rendering RL (only symbolic grid/state features are used).
- Cloud distributed cluster orchestration (focusing on local GPU/CPU generation and single-file Kaggle deployment).
