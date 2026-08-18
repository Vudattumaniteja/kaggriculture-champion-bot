# Kaggriculture Domain Glossary

Glossary and canonical terminology for the Kaggriculture simulation environment and competitive agent development.

## Game & Simulation Entities

**Agent**:
An autonomous decision-making program that receives observations each turn and submits farm management actions to maximize final net worth.
_Avoid_: Bot, player, model (when referring to the actor).

**Turn**:
A single discrete step in the 720-turn (30-day season) simulation.
_Avoid_: Tick, round, frame.

**Quadrant**:
A purchasable expansion zone on the farm grid used for crops and livestock pens.
_Avoid_: Land plot, chunk, tile group.

**Farmhand**:
A hired worker deployed to perform physical labor (planting, watering, harvesting, animal care).
_Avoid_: Worker, laborer, NPC.

**Market Price**:
The dynamic unit price of commodities fluctuating based on supply elasticity, sales volume, and competitor actions.
_Avoid_: Commodity cost, exchange rate.

## Modeling & AlphaZero Architecture

**Forward Model**:
A fast, lightweight internal Python/NumPy simulation clone of game rules used for MCTS node expansion and tree rollouts without calling the heavy external engine.
_Avoid_: Environment clone, simulator copy.

**Policy-Value Network**:
A dual-head neural network outputting action probability distribution $\pi(a|s)$ and scalar state value $v(s)$ (expected terminal profit / win probability).
_Avoid_: Actor-critic (in MCTS context), scoring model.

**Action Mask**:
A boolean filter dynamically pruning illegal or immediately bankrupting actions before policy softmax and tree branch expansion.
_Avoid_: Action filter, legal move list.

**Imitation Warm-Start**:
Supervised pre-training of the Policy-Value network on expert heuristic gameplay datasets before initiating self-play reinforcement learning.
_Avoid_: Behavioral cloning baseline, pre-training phase.

**Evaluation Tournament**:
An automated batch simulation runner evaluating agents head-to-head across multiple seeded episodes to compute win rates and net worth distributions.
_Avoid_: Validation harness, backtest loop.

**Hierarchical 2-Tier Architecture**:
A decoupled agent control system where a neural network outputs high-level strategic macro targets and spatial heatmaps, while a deterministic assignment engine solves micro pathfinding, chore execution, and worker collision resolution.
_Avoid_: Monolithic end-to-end policy, flat multi-agent controller.

**Town Shop**:
Specialized downstream consumer establishments (e.g. Bakery, Pizza Shop, Brunch Spot) that purchase specific agricultural commodities, directly influencing market absorption and price floors.
_Avoid_: Village vendor, store node.

**Market Price Elasticity / Crash Indicator**:
A computed financial metric tracking price divergence and market saturation relative to equilibrium base inventory ($I_0$), signaling whether dumping crops will crash spot prices.
_Avoid_: Price swing, market health score.

**Opponent Public Profile**:
The vector of observable opponent state (cash balance, tile development, worker headcount, quadrant unlocks) extracted from public observation feeds without requiring private shed access.
_Avoid_: Enemy telemetry, rival state tensor.

**Decoupled Multi-Head Policy**:
An actor network topology featuring separate, factorized output heads for distinct operational decisions (crop spatial heatmaps, livestock infrastructure, workforce recruiting, quadrant acquisition, market liquidation) avoiding categorical label squashing.
_Avoid_: Monolithic policy head, flat classifier.

**Micro Assignment Engine**:
The deterministic operational solver that translates high-level spatial heatmaps and quotas into collision-free agent paths, Hungarian task matching, and tool actions.
_Avoid_: Heuristic script, low-level bot.

**FiLM Conditioning (Feature-wise Linear Modulation)**:
A neural modulation mechanism where global economic embeddings compute affine scale and shift parameters ($\gamma, \beta$) that modulate convolutional visual feature maps, dynamically sensitizing spatial perception to market prices and liquidity.
_Avoid_: Feature concatenation, embedding adder.

**Distributional Two-Hot Value Representation**:
A critic formulation representing terminal net worth as a probability distribution over discrete logarithmic cash buckets, trained via cross-entropy loss to eliminate MSE gradient explosion across multi-order-of-magnitude cash scales.
_Avoid_: Scalar cash predictor, MSE value head.

**Pre-Softmax Action Masking**:
An analytical filter that sets invalid or bankrupting action logits to $-\infty$ prior to policy softmax computation, guaranteeing mathematical validity and preventing wasted exploration mass.
_Avoid_: Post-filter clipping, soft penalty learning.

**Two-Stage Market Guardrail**:
A hybrid liquidation system where continuous policy liquidation fractions pass through deterministic safety constraints (cow feed wheat reservation and shed midnight overflow emergency dump) before dispatching market orders.
_Avoid_: Raw market order output, unguarded liquidator.

**Neural-Weighted Hungarian Matching**:
A global bipartite task assignment formulation that minimizes total travel distance while maximizing neural spatial priority logits and penalizing critical chore neglect.
_Avoid_: Greedy worker dispatcher, naive nearest-neighbor picker.

**Autonomous Seed Head**:
A dedicated neural policy head outputting discrete purchasing volume for each seed variety, constrained by cash availability and current seed buffer targets.
_Avoid_: Heuristic seed buyer, hardcoded seed loop.

**Advantage-Weighted Imitation Learning (AWIL)**:
A sample-weighting formulation for expert replay imitation that scales individual transition loss by normalized, clipped advantage $\hat{A}_t = R_t - V(s_t)$, learning from high-quality decisions while downweighting blunders regardless of final game score.
_Avoid_: Uniform behavioral cloning, scalar score-only weighting.

**Two-Scale Hierarchical World Model**:
A dual-frequency recurrent dynamics engine containing within-day micro transitions ($g_{\text{micro}}: z_t \to z_{t+1}$) for unit-level simulation and macro day-skip transitions ($g_{\text{day}}: z_{d,23} \to z_{d+1,0}$) supervised by auxiliary next-day yield and price decoders for long-horizon seasonal MCTS.
_Avoid_: Monolithic 1-step unroller, flat 720-step simulator.

**Spatial Feature Channel Mapping**:
A representation strategy that embeds geometric and economic invariants (quadrant unlock costs, shed distance fields, town shop vectors, and unlock states) directly as $10 \times 10$ spatial feature channels, enabling equivariant D4 spatial data augmentation with synchronized action coordinate transforms.
_Avoid_: Static coordinate hardcoding, decoupled scalar quadrant labels.

**Two-Group Staged Loss Balancing**:
A multi-task optimization architecture that dynamically balances policy action head gradients via GradNorm while locking distributional value ($\lambda_{\text{val}} = 1.0$) and auxiliary foresight decoders ($\lambda_{\text{aux}} = 0.1$) to fixed manual coefficients to prevent task collapse.
_Avoid_: Homoscedastic uncertainty weighting across all heads, unweighted sum loss.

**Maturity-Aware Potential-Based Reward Shaping (PBRS)**:
A state potential formulation $\Phi(s, t) = \text{Cash}_t + \sum \omega_k(t) V_k(s_t)$ that dynamically decays planted crops, livestock, unplanted seeds, and shed inventory as the episode horizon approaches Turn 719, guaranteeing mathematical policy invariance while penalizing trapped deadweight.
_Avoid_: Arbitrary heuristic bonus, step reward shaping without potential telescoping.

**Prioritized Fictitious Self-Play (PFSP) Matchmaker**:
A dynamic multi-agent tournament selector that tracks participant Elo ratings and samples sparring opponents inversely proportional to champion win rate ($P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$), over-indexing on hard adversaries while preserving generalist resilience.
_Avoid_: Uniform self-play mirror, round-robin scheduler.

**Heuristic Fast-Forward Warmup**:
A sub-trajectory generation mechanism that fast-forwards early turns ($0 \to t_{\text{start}}$) in $<150\text{ms}$ using randomized pairings of heuristic specialists to inject organic midgame ($t \in [288, 432]$) and endgame ($t \in [528, 648]$) states for dense neural liquidation credit assignment.
_Avoid_: Offline state serialization hacking, static midgame snapshot injection.

**Mask-Safe Dirichlet Exploration**:
An MCTS root exploration technique that perturbs action priors with Dirichlet noise strictly over analytical pre-softmax action masks, preventing exploration probability mass from leaking into illegal or bankrupting actions.
_Avoid_: Unmasked Dirichlet noise, post-perturbation clipping.

**Grilling Model Workspace**:
The isolated module directory located at `grilling model/` (`C:\Users\Manit\Desktop\kaggle\grilling model\`) containing all source code, models, training routines (SSL/AWIL/RL), datasets, replay parsers, evaluation tests, and standalone submission artifacts for the unified championship agent.
_Avoid_: Scattering experimental scripts across root directory.



