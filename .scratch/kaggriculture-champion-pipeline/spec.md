# Feature Specification: Kaggriculture Unified Champion Bot End-to-End Pipeline

**Status**: `ready-for-agent`  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Tracked Issues**: 
- GitHub Issue #1: [Hierarchical ML/SSL/RL Bot Architecture & I/O Pipeline](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/1)
- GitHub Issue #2: [Advantage-Weighted Imitation Learning (AWIL) & Two-Scale Hierarchical World Model Pre-Training Pipeline](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/2)
- GitHub Issue #3: [RL Rewards, Potential-Based Shaping (PBRS), GAE Credit Assignment & League Ecosystem](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/3) (Frontier 7 Resolved)

---

## Problem Statement

Developing a Grandmaster-tier competitive agent for Kaggriculture presents four fundamental bottlenecks across representation, imitation pre-training, credit assignment, and league generalization:

1. **Representation & Coordination Bottleneck**:
   - Monolithic end-to-end reinforcement learning fails to coordinate up to 12 farmhands on a $10 \times 10$ grid. Squashing multiple worker actions into a single discrete label creates severe label collapse ($42.4\%$ of steps collapsed onto feeding cows, while land acquisition was squashed to $0.25\%$).
   - Standard spatial convolutions are blind to dynamic market elasticity, commodity spot price crashes, and town shop demand schedules, causing agents to dump high-value harvests into saturated markets.

2. **Imitation & Dynamics Bottleneck**:
   - Uniform behavioral cloning treats all turns in a winning replay as equally optimal, memorizing blunders while discarding brilliant opening crop rotations from matches that collapsed late.
   - Traditional world models generate "Frankenstein states" where scalar bank balances are predicted into the future while spatial grid images remain frozen in time, blinding lookahead search to multi-day crop maturation and animal break-even timelines.

3. **Reward & Credit Assignment Pathology**:
   - Evaluating matches purely on terminal cash at Turn 719 creates extreme credit diffusion across 720 turns, while naive step-by-step net worth rewards incentivize disastrous Day 28 deadweight asset hoarding (e.g. planting slow 12-day Melons or purchasing unplantable seeds near the end of the season).
   - Artificial piecewise reward penalties around starting capital create discontinuous gradient cliffs that paralyze early-game compounding investments, while flat Monte Carlo trajectory broadcast destroys temporal credit differentiation.

4. **Strategy Cycling & Meta Forgetting**:
   - Training solely against self-play mirrors causes the policy to collapse into narrow local optima (such as Carrot monocultures), leaving the agent vulnerable to aggressive heuristic specialists (Melon rushers, dairy syndicates, or market price crashers).

---

## Solution

A unified, mathematically rigorous, high-throughput championship training and execution pipeline comprising three tightly integrated tiers:

1. **Hierarchical 2-Tier Architecture & I/O Pipeline (Tier 1)**:
   - **Observation Encoder**: $24 \times 10 \times 10$ spatial feature tensor capturing crop identities, growth stages, moisture, fertilizer flags, livestock pens, shed locations, farmhand densities, and static economic maps (`Quadrant_Cost_Map`, `Distance_to_Shed`, `Shop_Vectors`), coupled with a 72-dimensional scalar vector of market prices, elasticity metrics, town shop demands, debt liabilities, and opponent public profiles.
   - **FiLM-Modulated Convolutional Backbone**: Global economic embeddings dynamically modulate convolutional feature maps via Feature-wise Linear Modulation ($\gamma, \beta$), ensuring spatial perception is conditioned on market liquidity and financial pressure.
   - **Decoupled Multi-Head Policy**: Independent factorized output heads for Spatial Crop Allocation ($5 \times 10 \times 10$), Livestock Target Herd Size (3 dims), Workforce Recruiting ($0..12$), Land Acquisition ($[0, 1]$), Autonomous Seed Replenishment (5 dims), and Continuous Market Liquidation ($9 \times [0, 1]$), safeguarded by pre-softmax analytical action masks.
   - **Critic Distributional Value & Win-Probability Heads**: 1001-bin two-hot symlog categorical distribution over $[-15.0, +15.0]$ symlog cash space ($[-\$3.26\text{M}, +\$3.26\text{M}]$) trained via cross-entropy loss alongside an auxiliary win-probability head.
   - **Two-Stage Market Guardrails & Neural-Weighted Hungarian Micro Solver**: Continuous market liquidation fractions pass through deterministic cow feed reservation floors and midnight shed overflow auto-monetization filters, while farmhand pathfinding and chore execution are solved via global linear sum assignment with biological urgency penalties.
   - **Single-File Kaggle Deployment**: Self-contained callable `agent(obs, config=None)` with base85-compressed FP16 neural weights for single-command CLI submission.

2. **Advantage-Weighted Imitation Learning (AWIL) & Two-Scale Hierarchical World Model (Tier 2)**:
   - **Quality Replay Curation**: High-volume parsing of 440,000 Grandmaster transitions filtered with a $\$50\text{k}$ terminal return floor, duplicate idle PASS deduplication, dual-player trajectory ingestion, and per-player representation caps.
   - **Equivariant D4 Spatial Augmentation**: Synchronized $90^\circ$ rotations and reflections applied to the 24-channel grid and $(r, c) \to (c, 9-r)$ spatial action targets while keeping scalar economic features and market fractions invariant.
   - **Two-Stage AWIL Advantage Weighting**: Value baseline $V_\phi(s)$ trained in Stage 1 to compute standardized advantages $\hat{A}_t$ within 5 seasonal phase buckets (Days 0–5, 6–12, 13–18, 19–24, 25–29), weighting transition losses via $w_t = g_i \cdot \operatorname{clip}(e^{\hat{A}_t / \tau}, 0.2, 5.0)$.
   - **Two-Scale Hierarchical World Model**: Latent micro-dynamics ($g_{\text{micro}}: z_t \times \mathbf{a}_{\text{strat}} \to z_{t+1}$) for within-day unit steps and macro day-skip dynamics ($g_{\text{day}}: z_{d, 23} \times \mathbf{a}_{\text{day}} \to z_{d+1, 0}$) supervised by auxiliary next-day yield ($\hat{Y}_{d+1}$), price ($\hat{P}_{d+1}$), and town shop demand ($\hat{D}_{d+1}$) decoders.
   - **Two-Group Staged Loss Balancing**: Strategic policy heads balanced dynamically via GradNorm while locking distributional value ($\lambda_{\text{val}} = 1.0$) and auxiliary foresight decoders ($\lambda_{\text{aux}} = 0.1$) to fixed manual weights.

3. **Maturity-Aware PBRS, GAE Credit Assignment & League Ecosystem (Tier 3)**:
   - **Maturity-Aware Potential-Based Reward Shaping (PBRS)**: Policy-invariant potential function $\Phi(s, t) = \text{Cash}_t + \sum \omega_k(t) V_k(s_t)$ that dynamically discounts immature crops, livestock 10-day break-even, unplanted seeds, and shed inventory as $t \to 719$, telescoping cleanly to terminal cash while applying a $2.0 \times \text{Deadweight}$ penalty at Turn 718.
   - **Generalized Advantage Estimation (GAE)**: $\gamma=0.995, \lambda=0.95$ temporal credit assignment distinguishing 4-day market price arbitrage holding windows from premature dumping or overdue holding.
   - **500k Verified Prioritized Experience Replay (PER)**: Binary SumTree sampling ($\alpha=0.6, \beta: 0.4 \to 1.0$) storing verified causal macro actions executed by the Hungarian dispatcher.
   - **Prioritized Fictitious Self-Play (PFSP) Single-Champion League**: 100% of gradient updates dedicated to the primary Champion bot, matchmaking dynamically across 40% historical checkpoints ($P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$), 40% mirror self-play, and 20% fixed Mega League heuristic specialists.
   - **80/20 Curriculum Fast-Forward Warmup**: 20% of training matches fast-forwarded in $<150\text{ms}$ on CPU using heuristic matchups into Midgame ($t \in [288, 432]$) and Endgame ($t \in [528, 648]$) states for dense, early liquidation credit assignment with sub-trajectory GAE recursion.
   - **Mask-Safe Dirichlet Exploration & Entropy Decay**: Dirichlet root noise ($\alpha=0.3, \epsilon=0.25$) restricted strictly to legal action masks, combined with cosine-annealed policy entropy loss ($0.05 \to 0.002$) and a two-stage Gumbel evaluation temperature schedule.

---

## User Stories

### Tier 1: State Representation, Neural Trunk & Decoupled Policy
1. As a competitive player, I want the agent to encode the 10×10 farm grid into a 24-channel spatial tensor, so that crop growth, soil moisture, animal health, worker positions, and economic distance fields are cleanly separated.
2. As a competitive player, I want the agent to track spot prices, baseline price ratios, and market crash indicators across all 9 commodities, so that harvested crops are never liquidated into collapsed markets.
3. As a competitive player, I want the agent to track town shop demand across all local establishments, so that high-margin downstream purchasing opportunities are prioritized.
4. As a competitive player, I want the agent to extract the public opponent profile (cash balance, unlocked quadrants, workforce headcount, tile development), so that the agent dynamically adapts to competitor expansion speed.
5. As a competitive player, I want the neural network to fuse spatial and scalar economic features using FiLM conditioning, so that financial pressure directly modulates spatial visual filters.
6. As a competitive player, I want the policy to output a decoupled $5 \times 10 \times 10$ spatial crop priority heatmap, so that planting decisions across different quadrants do not conflict with each other.
7. As a competitive player, I want the policy to output dedicated livestock target headcounts for Cows, Sheep, and Geese, so that herd expansion is balanced against daily feed supplies.
8. As a competitive player, I want the policy to output a dedicated workforce hiring head ($0..12$), so that farmhand recruitment scales dynamically with seasonal labor demand.
9. As a competitive player, I want the policy to output a quadrant acquisition probability, so that land expansion occurs at financially optimal turns.
10. As a competitive player, I want the policy to output autonomous seed replenishment targets, so that required seeds are maintained without over-purchasing.
11. As a competitive player, I want the policy to output continuous market liquidation fractions ($[0, 1]$) per commodity, so that the agent can execute fine-grained inventory sell-offs.
12. As a competitive player, I want a Stage 2 cow feed reservation filter, so that wheat required for daily cow sustenance is never accidentally liquidated at market.
13. As a competitive player, I want a Stage 2 midnight shed overflow auto-liquidation guardrail, so that excess items above the 100-item shed capacity are monetized instead of trashed at day end.
14. As a competitive player, I want pre-softmax analytical action masking on hiring and land acquisition heads, so that illegal or bankrupting moves are eliminated before policy sampling.
15. As a competitive player, I want the Critic to output a 1001-bin distributional two-hot symlog cash distribution, so that value gradients remain stable across multi-million dollar cash scales.
16. As a competitive player, I want the Critic to output an auxiliary win-probability head, so that head-to-head match dominance is explicitly modeled.
17. As a competitive player, I want the Micro Assignment Engine to solve worker dispatch via Hungarian linear sum assignment, so that worker travel time is minimized with zero path collisions.
18. As a competitive player, I want biological urgency penalties integrated into the Hungarian cost matrix, so that starving animals, dry crops, and ripe harvests are serviced before low-priority tasks.
19. As a competitive player, I want the complete agent packaged into a single-file callable contract `agent(obs, config=None)`, so that it can be submitted directly to Kaggle and benchmarked locally.

### Tier 2: AWIL Pre-Training & Hierarchical World Modeling
20. As a machine learning engineer, I want the replay parser to extract $5 \times 10 \times 10$ rolling crop allocation heatmaps from replay farm tiles, so that the crop head receives dense spatial supervisory gradients on every turn.
21. As a machine learning engineer, I want the replay parser to generate an exponential lead-in ramp $y_t = \exp(-\Delta t / 12)$ for the 12 turns prior to land acquisition, so that the policy learns to anticipate capital allocation before expansion.
22. As a machine learning engineer, I want the replay parser to compute continuous market liquidation fractions and mask empty inventory items, so that trade supervision reflects proportional inventory strategy.
23. As a machine learning engineer, I want the dataset generator to prune matches with final score $< \$50,000$, so that low-quality random baselines do not pollute the imitation pool.
24. As a machine learning engineer, I want the dataset generator to prune identical consecutive PASS transitions, so that zero-action idle steps do not cause policy apathy.
25. As a machine learning engineer, I want the dataset generator to include both Player 0 and Player 1 transitions if both scored $\ge \$50,000$, so that competitive dual-player dynamics are captured.
26. As a machine learning engineer, I want the dataset generator to cap maximum transitions contributed by any single player/match, so that the model does not overfit to one specific playstyle.
27. As a machine learning engineer, I want to train a value baseline network $V_\phi(s)$ in Stage 1, so that state-value expectations are established before policy weighting.
28. As a machine learning engineer, I want advantage estimates $\hat{A}_t = R_t - V_\phi(s_t)$ normalized independently within 5 seasonal phase buckets, so that early-game opening moves are not overshadowed by late-game dollar scales.
29. As a machine learning engineer, I want transition sample weights computed via $w_t = g_i \cdot \operatorname{clip}(e^{\hat{A}_t / \tau}, 0.2, 5.0)$, so that brilliant decisions are strongly imitated while mistakes are suppressed.
30. As a machine learning engineer, I want the observation encoder to embed quadrant unlock costs, shed distance fields, and shop direction vectors into $10 \times 10$ spatial feature channels, so that spatial filters learn invariant economic geometry.
31. As a machine learning engineer, I want data augmentation to apply synchronized D4 rotations and reflections to the $24 \times 10 \times 10$ spatial tensor and action target coordinates $(r, c) \to (c, 9-r)$, so that effective training data volume is quadrupled without breaking spatial validity.
32. As a machine learning engineer, I want non-spatial market orders and global economic scalars left invariant during D4 augmentation, so that commodity pricing is not corrupted.
33. As a machine learning engineer, I want the Two-Scale Hierarchical World Model to train micro-step dynamics $g_{\text{micro}}(z_t, \mathbf{a}_{\text{strat}}) \to z_{t+1}$, so that within-day unit movements and action point expenditures are modeled in latent space.
34. As a machine learning engineer, I want the Two-Scale Hierarchical World Model to train macro day-skip dynamics $g_{\text{day}}(z_{d, 23}, \mathbf{a}_{\text{day}}) \to z_{d+1, 0}$, so that long-horizon seasonal MCTS can plan in 3–5 macro steps.
35. As a machine learning engineer, I want auxiliary next-morning decoders predicting crop yield grids ($\hat{Y}_{d+1}$), commodity spot prices ($\hat{P}_{d+1}$), and town shop demand ($\hat{D}_{d+1}$), so that the latent embedding $z$ is physically grounded and protected against representation collapse.
36. As a machine learning engineer, I want policy action heads balanced dynamically using GradNorm, so that gradients from the spatial crop head do not overpower workforce hiring or land acquisition.
37. As a machine learning engineer, I want the distributional value head and auxiliary decoders locked to fixed manual loss weights ($\lambda_{\text{val}}=1.0, \lambda_{\text{aux}}=0.1$), so that value calibration remains unconditionally stable during policy optimization.

### Tier 3: RL Rewards, GAE Credit Assignment & League Ecosystem
38. As a competition bot designer, I want the agent to aggressively deploy starting capital on Days 1–15, so that the farm achieves maximal exponential compounding without artificial bank penalty cliffs.
39. As a competition bot designer, I want unplanted seeds in inventory to have zero potential value when $(T - t) < \text{GrowTurns} + 24$, so that the agent never buys deadweight seeds on Day 28.
40. As a competition bot designer, I want planted crops to dynamically decay in potential value if remaining turns are less than their growth duration, so that the agent transitions smoothly into fast-cycling crops near the end of the season.
41. As a competition bot designer, I want livestock potential to decay linearly over a 10-day break-even horizon, so that pastures and coops are built exclusively during early game when positive ROI is mathematically guaranteed.
42. As a competition bot designer, I want shed goods potential to decay to zero at $t \ge 718$, so that the agent is forced to fully liquidate inventory into cash before the match ends.
43. As a competition bot designer, I want step-level shaped rewards to satisfy Ng et al. (1999) potential telescoping, so that intermediate rewards never alter the optimal policy relative to final net cash margin.
44. As a competition bot designer, I want GAE temporal credit assignment ($\gamma=0.995, \lambda=0.95$), so that 4-day market price arbitrage holding cycles and 12-day Melon growth receive strong, accurate gradient signals.
45. As a competition bot designer, I want a 500,000-transition Prioritized Experience Replay buffer, so that surprising transitions and high TD-error strategic pivots are replayed with proportional priority.
46. As a competition bot designer, I want actor transitions to log verified causal macro actions executed by the Hungarian dispatcher, so that unexecutable target intents do not contaminate the replay buffer.
47. As a competition bot designer, I want Prioritized Fictitious Self-Play (PFSP) matchmaking, so that the Champion bot over-samples past checkpoints and heuristic opponents that cause the highest loss rates.
48. As a competition bot designer, I want 100% of gradient updates allocated to the single Champion network, so that compute efficiency is maximized during time-constrained training windows.
49. As a competition bot designer, I want an 80/20 Curriculum Fast-Forward Warmup mechanism, so that the agent receives high-density exposure to Day 28 liquidation crunches from the very first hour of training.
50. As a competition bot designer, I want heuristic warmup simulation to execute in $<150\text{ms}$ on CPU, so that curriculum matches incur near-zero overhead compared to full matches.
51. As a competition bot designer, I want sub-trajectory GAE advantages to recurse cleanly from Turn 718 down to $t_{\text{start}}$, so that heuristic warmup turns do not pollute the neural training dataset.
52. As a competition bot designer, I want Dirichlet exploration noise ($\alpha=0.3, \epsilon=0.25$) to be masked to valid actions, so that exploration mass is never wasted on bankrupting or illegal moves.
53. As a competition bot designer, I want policy entropy regularization to anneal from $0.05 \to 0.002$ via a cosine schedule, so that early macro diversity transitions into sharp, deterministic tournament execution.
54. As a competition bot designer, I want auto-checkpointing every 5–15 minutes, so that training progress is resilient to sudden interruptions and always ready for immediate packaging.
55. As a competition bot designer, I want an automated packaging routine that compiles weights into a standalone single-file `submission.py`, so that final tournament bots can be submitted to Kaggle with a single CLI command.

---

## Implementation Decisions

### Architectural Decision 1: Hierarchical 2-Tier I/O Contract & Feature Formulation
The system separates strategic planning from operational execution:
- **Spatial Grid Tensor ($24 \times 10 \times 10$)**:
  - Channels 0–4: One-hot crop identities (Wheat, Carrot, Tomato, Strawberry, Melon).
  - Channels 5–7: Continuous tile states (Crop Growth Ratio $[0, 1]$, Soil Moisture $[0, 1]$, Fertilizer Active $\{0, 1\}$).
  - Channels 8–10: Livestock Pen Type (Coop, Pasture, Empty).
  - Channels 11–13: Livestock Animal Type (Goose, Sheep, Cow).
  - Channels 14–15: Livestock State (Hunger $[0, 1]$, Product Ready $\{0, 1\}$).
  - Channels 16–17: Facility Footprints (Shed Tiles $\{0, 1\}$, Town Delivery Zones $\{0, 1\}$).
  - Channel 18: Farmhand Position Density Map.
  - Channels 19–23: Static Invariant Geometry (`Quadrant_Cost_Map`, `Distance_to_Shed`, `Distance_to_Shop`, `Shop_Delta_Row`, `Shop_Delta_Col`, `Unlock_Status_Map`).
- **Scalar Economic Vector (72 Dimensions)**:
  - Turn & Phase Clocks: Normalized turn $t / 720$, day phase $t \pmod{24} / 24$, season progress.
  - Liquid Cash & Velocity: Symlog bank cash, 24-turn rolling cash delta, projected wage liability $\text{Workers} \times \$20$.
  - Commodity Market Dynamics: 9 spot prices, baseline ratios $P_i / P_{i,\text{base}}$, elasticity/crash metrics $I_i / I_{i,0}$.
  - Downstream Town Shop Demands: Active buying contracts across local consumer shops.
  - Inventory & Logistics: Shed item counts $[0, 100]$, shed saturation ratio, seed inventory counts.
  - Public Opponent Profile: Opponent cash, worker count, tile count, quadrant unlock flags.

### Architectural Decision 2: FiLM Conditioning & Decoupled Multi-Head Policy
- **Backbone**: 2-layer MLP projects 72-dim scalars to 128-dim economic embedding $\mathbf{z}_{\text{econ}}$. A 3-block Convolutional ResNet (64 channels) is modulated via affine Feature-wise Linear Modulation $\text{FiLM}(\mathbf{F}) = \gamma(\mathbf{z}_{\text{econ}}) \odot \mathbf{F} + \beta(\mathbf{z}_{\text{econ}})$.
- **Decoupled Actor Heads**:
  - `Crop Head`: $1 \times 1$ Conv $\to 5 \times 10 \times 10$ logits.
  - `Livestock Head`: Linear projection $\to 3$ discrete herd quota logits (Goose, Sheep, Cow).
  - `Workforce Head`: Linear projection $\to 13$ discrete hiring logits ($0..12$ farmhands) with pre-softmax budget masks.
  - `Quadrant Head`: Linear projection $\to 1$ sigmoid unlock logit with affordability masks.
  - `Autonomous Seed Head`: Linear projection $\to 5$ discrete seed replenishment logits.
  - `Market Intent Head`: Linear projection $\to 9$ continuous sigmoid liquidation fractions $[0, 1]$.
- **Critic Head**: 1001-bin categorical projection over $[-15.0, +15.0]$ symlog cash scale plus 1-dim sigmoid win-probability head.

### Architectural Decision 3: Two-Stage Market Guardrails & Hungarian Micro Assignment
- **Stage 1 (Policy Intent)**: Policy emits continuous liquidation fractions $f_i \in [0, 1]$ per commodity.
- **Stage 2 (Deterministic Guardrail)**:
  - Cow feed reservation: Clamps Wheat liquidation such that $\text{Wheat}_{\text{shed}} - \text{Wheat}_{\text{sold}} \ge \text{Cows} \times 2$.
  - Midnight shed overflow: If $t \pmod{24} == 23$ and total shed items $> 100$, excess items are automatically dumped to market at spot prices instead of being discarded by the engine.
- **Micro Task Assignment**: Formulates worker chore assignment as bipartite matching. Cost matrix $\mathbf{C}_{ij} = \text{Distance}(w_i, t_j) - \lambda \cdot \text{Logit}(t_j) + \text{Urgency}(t_j)$ solved via `scipy.optimize.linear_sum_assignment` with biological urgency overrides for starving animals, dry crops, and ripe harvests.

### Architectural Decision 4: AWIL Sample Weighting & Quality Replay Pipeline
- **Quality Filter**: Replays with final cash $< \$50\text{k}$ and consecutive duplicate idle PASS steps are pruned. Retains qualifying games from both players with per-player representation caps.
- **AWIL Advantage Normalization**: Value baseline $V_\phi(s)$ evaluates transitions. Advantages $\hat{A}_t = R_t - V_\phi(s_t)$ are normalized per seasonal phase bucket $b \in \{0..4\}$ ($\mu_b, \sigma_b$), with sample loss weights computed as:
  $$w_t = g_i \cdot \operatorname{clip}\left(\exp\left(\frac{\hat{A}_t - \mu_b}{\tau \cdot \sigma_b}\right), 0.2, 5.0\right)$$
- **Synchronized D4 Augmentation**: Random dihedral group $D_4$ rotations ($k \times 90^\circ$) and reflections applied to spatial grids and action targets $(r, c) \to (c, 9-r)$, quadrupling dataset volume without corrupting non-spatial market features.

### Architectural Decision 5: Two-Scale Hierarchical World Model
- **Micro Dynamics ($g_{\text{micro}}$)**: $z_{t+1} = g_{\text{micro}}(z_t, \mathbf{a}_{\text{strat}})$ models 1-step within-day transitions in 128-dim latent space.
- **Macro Day-Skip Dynamics ($g_{\text{day}}$)**: $z_{d+1, 0} = g_{\text{day}}(z_{d, 23}, \mathbf{a}_{\text{day}})$ unrolls 24-step day boundaries in a single step for long-horizon seasonal planning.
- **Auxiliary Decoders**: Latents are physically grounded by auxiliary heads predicting next-morning crop yields ($\hat{Y}_{d+1} \in \mathbb{R}^{2 \times 10 \times 10}$), commodity spot prices ($\hat{P}_{d+1} \in \mathbb{R}^9$), and town shop demands ($\hat{D}_{d+1} \in \mathbb{R}^3$).
- **Two-Group Staged Loss Balancing**: Group 1 policy heads balanced dynamically via GradNorm; Group 2 value ($\lambda_{\text{val}}=1.0$) and auxiliary foresight decoders ($\lambda_{\text{aux}}=0.1$) locked to fixed manual weights.

### Architectural Decision 6: Maturity-Aware PBRS & GAE Credit Assignment
- **PBRS Formulation**: Step reward $r'_t = r_t + \gamma \Phi(s_{t+1}, t+1) - \Phi(s_t, t)$ with relative potential $\Phi(s, t) = \Phi_{\text{champ}}(s, t) - \Phi_{\text{opp}}(s, t)$.
- **Asset Decay Schedule**:
  - Crops: $\omega_{\text{crop}}(t) = 1.0$ if $(T - t) \ge \text{MaturationTurns}$, decaying to $0.0$ if $(T - t) < \text{HarvestSlack}$.
  - Livestock: $\omega_{\text{animal}}(t) = 1.0$ if $(T - t) \ge 240$ turns (10-day break-even), decaying linearly to $0.0$ at $(T - t) < 48$ turns.
  - Seeds: $\omega_{\text{seed}}(t) = 1.0$ if $(T - t) \ge \text{GrowTurns} + 24$, decaying to $0.0$ otherwise.
  - Shed Goods: $\omega_{\text{shed}}(t) = 1.0$ if $(T - t) \ge 24$, $0.5$ if $2 \le (T - t) < 24$, and $0.0$ at $t \ge 718$.
  - Terminal Deadweight: $2.0 \times \text{Deadweight}$ penalty applied at $t = 718$.
- **Pathology Purge**: Completely remove piecewise $\$3,000$ cash growth/loss penalty terms and uniform Monte Carlo broadcast.
- **GAE Parameters**: Discount factor $\gamma = 0.995$, GAE parameter $\lambda = 0.95$.

### Architectural Decision 7: Frontier 7 League Dynamics & Curriculum Resolution
- **Pillar 7.1 (Single-Champion PFSP League)**: 100% of gradient updates dedicated to the primary Champion bot. Matchmaking samples 40% from rolling FIFO pool of up to 30 past checkpoints weighted by $P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$, 40% self-play mirror with Dirichlet noise, and 20% fixed Mega League heuristic specialists.
- **Pillar 7.2 (80/20 Curriculum Warmup)**: 80% Turn 0 starts, 10% Midgame starts ($D_{\text{start}} \sim \text{Uniform}(12, 18)$), 10% Endgame starts ($D_{\text{start}} \sim \text{Uniform}(22, 27)$) fast-forwarded on CPU in $<150\text{ms}$ with GAE advantage recursion on sub-trajectories $[t_{\text{start}}, 718]$.
- **Pillar 7.3 (Mask-Safe Exploration & Cosine Entropy Decay)**: Mask-safe Dirichlet root noise ($\alpha=0.3, \epsilon=0.25$) applied strictly over pre-softmax action masks. Policy entropy loss weight annealed via cosine schedule from $0.05 \to 0.002$. Evaluation Gumbel temperature $\tau = 1.0$ for $t < 48$, transitioning to $\tau = 0.0$ for $t \ge 48$.

---

## Testing Decisions

### Testing Principles
- Tests must validate observable, external game-theoretic behaviors and financial margins rather than internal neural activations.
- PBRS reward telescoping must be mathematically verified over arbitrary episode slices, proving that intermediate shaped rewards sum exactly to terminal relative cash difference.
- Zero-deadweight compliance must be tested by evaluating residual unharvested crops, unplanted seeds, and unsold shed inventory at Turn 719.

### Seams to Test

```
+-----------------------------------------------------------------------------------------------+
|                                     TESTING SEAM TOPOLOGY                                     |
|                                                                                               |
|  [Seam 1: Agent Environment Seam (Highest Seam)]                                              |
|  agent(obs, config) <---> kaggle_environments.make("kaggriculture")                           |
|                                                                                               |
|  [Seam 2: Model & Decoupled Policy Seam]                                                      |
|  forward(spatial, scalar, masks) -> (crop_logits, herd_logits, hire_logits, val_dist, ...)     |
|                                                                                               |
|  [Seam 3: Trajectory GAE & PBRS Seam]                                                         |
|  process_match(observations, value_estimates) -> (shaped_rewards, advantages, value_targets)   |
|                                                                                               |
|  [Seam 4: AWIL Dataset & World Model Seam]                                                    |
|  build_awil_dataset(...) -> dataset_stats                                                     |
|  imagine_micro(z_t, a_strat) / imagine_day_skip(z_23, a_day) -> next_latents                   |
|                                                                                               |
|  [Seam 5: PFSP League & Fast-Forward Curriculum Seam]                                         |
|  sample_opponent() -> opponent_fn                                                             |
|  fast_forward_simulation(t_start) -> (obs_start, sub_trajectory)                              |
+-----------------------------------------------------------------------------------------------+
```

1. **Seam 1: Agent Environment Seam (Highest / Primary Seam)**:
   - **Interface**: `agent(obs, config=None) -> Dict[str, Any]` evaluated via `kaggle_environments.make("kaggriculture")`.
   - **Validation**: Full 720-step match execution against `"starter"`, `"random"`, and `"pass"` baselines. Asserts non-negative cash balances, valid action syntax, zero worker collisions, and net worth superior to baselines.

2. **Seam 2: Model & Decoupled Policy Seam**:
   - **Interface**: `network(spatial_tensor, scalar_vector, action_masks) -> Dict[str, torch.Tensor]`.
   - **Validation**: Verifies that all factorized output heads emit tensors of correct shapes, pre-softmax masks clamp illegal hiring and land purchase logits to $-\infty$, and value distributions sum to 1.0 without NaNs.

3. **Seam 3: Trajectory GAE & PBRS Processor Seam**:
   - **Interface**: `TrajectoryGAEProcessor.process_match(observations, value_estimates) -> (shaped_rewards, advantages, value_targets)`.
   - **Validation**: Verifies that shaped rewards telescope to terminal cash ($\sum r'_t \equiv \text{Cash}_T - \text{Cash}_0$), GAE advantage bounds remain within finite range $[-15.0, +15.0]$, and deadweight assets on Day 28 receive zero potential value.

4. **Seam 4: AWIL Dataset & Hierarchical World Model Seam**:
   - **Interface**: `build_awil_dataset(raw_replays, output_path)` and `imagine_day_skip(z_23, a_day)`.
   - **Validation**: Verifies quality filtering ($< \$50\text{k}$ pruned), synchronized D4 rotation equivariance, advantage weight bounds $w_t \in [0.2, 5.0]$, and non-collapsing auxiliary yield/price decoders.

5. **Seam 5: PFSP League & Fast-Forward Curriculum Seam**:
   - **Interface**: `AlphaGoatMegaLeague.sample_opponent()` and `fast_forward_match(t_start)`.
   - **Validation**: Verifies PFSP probability weighting inversely proportional to win rate, $<150\text{ms}$ heuristic fast-forward execution, and valid sub-trajectory GAE recursion from Turn 718 down to $t_{\text{start}}$.

### Prior Art
- `src/agents/hrl_12worker_dispatcher.py`: Hungarian assignment and path collision avoidance logic.
- `src/training/mega_league.py`: Multi-personality tournament harness and Elo tracking.
- `src/training/overnight_rl_pipeline.py`: Parallel rollout generation and experience replay storage.

---

## Out of Scope

- Multi-network concurrent optimization leagues (e.g., 3-tier AlphaStar main/exploiter networks) due to GPU compute budget limits.
- Modifying the underlying `kaggle-environments` simulation engine rules or action point economics.
- Online search hyperparameter grid tuning during live Kaggle evaluation steps.

---

## Further Notes

- All domain terminology strictly aligns with [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md).
- Packaging pipeline automatically serializes trained FP16 weights into a standalone `submission.py` single file ready for immediate submission via `kaggle competitions submit kaggriculture -f submission.py`.
