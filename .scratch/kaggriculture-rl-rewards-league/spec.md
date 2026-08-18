# Feature Specification: Kaggriculture RL Rewards, Maturity-Aware PBRS, GAE Credit Assignment & Multi-Agent League Ecosystem

## Problem Statement

Previous deep reinforcement learning attempts in the Kaggriculture environment suffered from severe structural and game-theoretic failure modes that prevented agents from reaching competitive Grandmaster performance:
1. **Day 1 Capital Paralysis ($3,000 Bank Kink)**: Piecewise capital loss penalties around the initial $3,000 cash balance created a steep artificial gradient cliff, discouraging early land expansion and crop planting during the critical early compound growth window.
2. **Monte Carlo Credit Blurring**: Trajectory-level flat broadcast returns assigned identical credit to brilliant Day 10 commodity holding decisions and disastrous Day 28 deadweight over-purchases, blinding the policy to multi-day price arbitrage dynamics.
3. **Severe End-Game Deadweight Accumulation**: Standard step rewards rewarded asset accumulation without accounting for the remaining horizon, resulting in bots planting slow 12-day Melons or buying seeds on Day 28 that could never mature or liquidate before Turn 719.
4. **Strategy Cycling & Meta Forgetting**: Training solely via mirror self-play caused the policy to collapse into narrow local optima (e.g. Carrot-only monocultures), leaving it vulnerable to diverse heuristic specialists like fast Melon rushers, cow dairies, or aggressive market price crashers.

## Solution

A mathematically sound, high-throughput reinforcement learning pipeline featuring:
1. **Maturity-Aware Potential-Based Reward Shaping (PBRS)**: An Ng et al. (1999) policy-invariant potential function $\Phi(s, t) = \text{Cash}_t + \sum \omega_k(t) V_k(s_t)$ that dynamically discounts immature crops, livestock break-even, unplanted seeds, and shed inventory as the episode approaches Turn 719, combined with a terminal deadweight surcharge.
2. **Generalized Advantage Estimation (GAE)**: Fine-grained credit assignment ($\gamma=0.995, \lambda=0.95$) that assigns positive advantages to trades executed during price spikes and negative advantages to holding past market peaks.
3. **1001-Bin Symlog Categorical Two-Hot Value Head**: Bounded unit-norm gradient optimization over $[-15.0, +15.0]$ symlog cash scale ($[-\$3.26\text{M}, +\$3.26\text{M}]$).
4. **Prioritized Fictitious Self-Play (PFSP) Single-Champion League**: 100% of gradient updates dedicated to the primary Champion bot, matchmaking dynamically against 40% historical checkpoints ($P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$), 40% self-play mirror, and 20% fixed heuristic specialists.
5. **80/20 Curriculum Fast-Forward Warmup**: 20% of training matches fast-forwarded in $<150\text{ms}$ via heuristic matchups into Midgame ($t \in [288, 432]$) and Endgame ($t \in [528, 648]$) states for dense, early liquidation credit assignment.
6. **Mask-Safe Exploration & Entropy Regularization**: Dirichlet root noise ($\alpha=0.3, \epsilon=0.25$) restricted strictly to legal action masks, paired with cosine-annealed policy entropy loss ($0.05 \to 0.002$).

## User Stories

1. As a competition bot designer, I want the agent to aggressively deploy starting capital on Days 1–15, so that the farm achieves maximal exponential compounding without artificial bank penalty cliffs.
2. As a competition bot designer, I want unplanted seeds in inventory to have zero potential value when $(T - t) < \text{GrowTurns} + 24$, so that the agent never buys deadweight seeds on Day 28.
3. As a competition bot designer, I want planted crops to dynamically decay in potential value if remaining turns are less than their growth duration, so that the agent transitions smoothly into fast-cycling crops near the end of the season.
4. As a competition bot designer, I want livestock potential to decay linearly over a 10-day break-even horizon, so that pastures and coops are built exclusively during early game when positive ROI is mathematically guaranteed.
5. As a competition bot designer, I want shed goods potential to decay to zero at $t \ge 718$, so that the agent is forced to fully liquidate inventory into cash before the match ends.
6. As a competition bot designer, I want step-level shaped rewards to satisfy Ng et al. (1999) potential telescoping, so that intermediate rewards never alter the optimal policy relative to final net cash margin.
7. As a competition bot designer, I want GAE temporal credit assignment ($\gamma=0.995, \lambda=0.95$), so that 4-day market price arbitrage holding cycles and 12-day Melon growth receive strong, accurate gradient signals.
8. As a competition bot designer, I want the Critic value head to use a 1001-bin two-hot symlog categorical representation, so that cross-entropy value loss prevents MSE gradient explosions across multi-million dollar cash swings.
9. As a competition bot designer, I want a 500,000-transition Prioritized Experience Replay buffer, so that surprising transitions and high TD-error strategic pivots are replayed with proportional priority.
10. As a competition bot designer, I want actor transitions to log verified causal macro actions executed by the Hungarian dispatcher, so that unexecutable target intents do not contaminate the replay buffer.
11. As a competition bot designer, I want Prioritized Fictitious Self-Play (PFSP) matchmaking, so that the Champion bot over-samples past checkpoints and heuristic opponents that cause the highest loss rates.
12. As a competition bot designer, I want 100% of gradient updates allocated to the single Champion network, so that compute efficiency is maximized during time-constrained training windows.
13. As a competition bot designer, I want an 80/20 Curriculum Fast-Forward Warmup mechanism, so that the agent receives high-density exposure to Day 28 liquidation crunches from the very first hour of training.
14. As a competition bot designer, I want heuristic warmup simulation to execute in $<150\text{ms}$ on CPU, so that curriculum matches incur near-zero overhead compared to full matches.
15. As a competition bot designer, I want sub-trajectory GAE advantages to recurse cleanly from Turn 718 down to $t_{\text{start}}$, so that heuristic warmup turns do not pollute the neural training dataset.
16. As a competition bot designer, I want Dirichlet exploration noise ($\alpha=0.3, \epsilon=0.25$) to be masked to valid actions, so that exploration mass is never wasted on bankrupting or illegal moves.
17. As a competition bot designer, I want policy entropy regularization to anneal from $0.05 \to 0.002$ via a cosine schedule, so that early macro diversity transitions into sharp, deterministic tournament execution.
18. As a competition bot designer, I want auto-checkpointing every 5–15 minutes, so that training progress is resilient to sudden interruptions and always ready for immediate packaging.
19. As a competition bot designer, I want an automated packaging routine that compiles weights into a standalone single-file `submission.py`, so that final tournament bots can be submitted to Kaggle with a single CLI command.

## Implementation Decisions

### Decision 1: Relative Margin Objective & PBRS Formulation
The foundational objective is symmetric relative margin at Turn 719: $M(s_{719}) = \text{Cash}_{\text{champ}} - \text{Cash}_{\text{opp}}$. Step rewards are shaped using relative potential difference $\Phi(s, t) = \Phi_{\text{champ}}(s, t) - \Phi_{\text{opp}}(s, t)$:
$$r'_t = r_t + \gamma \Phi(s_{t+1}, t+1) - \Phi(s_t, t)$$
The potential function dynamically discounts crops, livestock, seeds, and shed goods based on remaining turns $(T - t)$, ensuring that $\Phi(s_{719}, 719) = \text{Cash}_{719}$. A terminal deadweight penalty of $2.0 \times \text{Deadweight}$ is applied at $t = 718$.

### Decision 2: Pathology Purge
- Completely remove the piecewise $3,000 cash growth bonus and capital loss penalty terms to restore linear, uninhibited early capital deployment.
- Completely remove flat Monte Carlo trajectory broadcast; replace with step-wise GAE advantage targets.

### Decision 3: GAE Horizon & Discounting
- Set discount factor $\gamma = 0.995$ (effective horizon $\approx 200$ turns / 8.3 days) and GAE parameter $\lambda = 0.95$.
- Utilize the auxiliary 24-turn commodity price forecaster head to guide market holding advantages.

### Decision 4: Distributional Symlog Two-Hot Value Head
- Value head produces logits over 1001 categorical bins uniformly spaced in $[-15.0, +15.0]$ symlog space ($h(x) = \text{sign}(x) \ln(|x| + 1)$).
- Supervised via categorical cross-entropy against two-hot soft target distributions.

### Decision 5: Single-Champion PFSP League Matchmaker
- 100% of gradient updates optimize the active Champion bot.
- Matchmaking samples 40% from a rolling FIFO pool of up to 30 past Champion checkpoints weighted by $P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$, 40% self-play mirror with Dirichlet noise, and 20% fixed Mega League heuristic specialists.

### Decision 6: Heuristic Fast-Forward Curriculum Ingestion
- 80% of matches initialize at $t = 0$.
- 10% initialize in Midgame ($D_{\text{start}} \sim \text{Uniform}(12, 18)$) and 10% in Endgame ($D_{\text{start}} \sim \text{Uniform}(22, 27)$).
- Warmup turns are rapidly stepped ($<150\text{ms}$) using randomly paired heuristic agents. Neural MCTS takes control at $t_{\text{start}}$, and sub-trajectory transitions $[t_{\text{start}}, 718]$ are recorded with GAE backward recursion.

### Decision 7: Mask-Safe Exploration & Cosine Entropy Decay
- Root Dirichlet noise $\text{Dir}(\alpha=0.3, \epsilon=0.25)$ added to policy priors strictly over valid action masks.
- Policy entropy loss coefficient $c_{\text{entropy}}$ annealed via cosine decay from $0.05 \to 0.002$.
- Evaluation Gumbel temperature $\tau = 1.0$ for turns $0 \le t < 48$, transitioning to $\tau = 0.0$ for $t \ge 48$.

## Testing Decisions

### Testing Principles
- Tests must verify external economic and game-theoretic behavior across full 720-step matches rather than internal neural activations.
- PBRS telescoping must be mathematically verified over arbitrary trajectory lengths ($T=720$ and truncated sub-trajectories).
- Zero-deadweight compliance must be tested by measuring final unliquidated assets at Turn 719.

### Seams to Test
1. **Primary Agent Seam**: `agent(obs, config=None)` evaluated on `kaggle_environments.make("kaggriculture")` against standard `"starter"` baseline.
2. **Trajectory Processor Seam**: `TrajectoryGAEProcessor.process_match(observations, value_estimates)` verifying GAE advantage bounds, PBRS reward telescoping, and symlog two-hot reconstruction fidelity.
3. **League Matchmaker Seam**: `AlphaGoatMegaLeague.sample_opponent()` verifying PFSP priority probability weighting and Elo tracking updates.
4. **Fast-Forward Curriculum Seam**: Match simulation worker starting at $t_{\text{start}} > 0$ verifying warmup execution speed ($<200\text{ms}$) and valid sub-trajectory length.

### Prior Art
- `src/training/overnight_rl_pipeline.py` multi-worker parallel simulation harness.
- `src/training/mega_league.py` 30-personality tournament evaluator.

## Out of Scope
- Architectural changes to the 2-Tier Hierarchical micro assignment engine or Hungarian bipartite matching logic.
- Adding additional macro action categories beyond the established 10 discrete macro actions.
- Multi-network concurrent optimization leagues (e.g. 3-tier AlphaStar exploiter architectures) due to compute budget constraints.

## Further Notes
- Supports flexible training durations via the `--target_hours` parameter (e.g., 0.5h fast sprint, 3.0h standard, 6.5h overnight).
- Standalone weights compilation automatically packages half-precision FP16 weights into base85 zlib-compressed strings inside `submission.py`.
