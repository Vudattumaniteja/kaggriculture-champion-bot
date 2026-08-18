# Feature Specification: Kaggriculture Hierarchical ML/SSL/RL Bot Architecture & I/O Pipeline

**Status**: `ready-for-agent`  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)

---

## Problem Statement

Previous reinforcement and imitation learning agents in Kaggriculture suffered from severe structural failure modes:
1. **Label Squashing & Multi-Agent Thrashing**: Attempting to predict flat, monolithic macro action labels (e.g. "Feed Cows") squashed 12 simultaneous farmhands into a single action, causing 42% of training examples to collapse onto feeding cows and leading workers to abandon tasks mid-path.
2. **Market Blindness**: Agents lacked explicit price elasticity, saturation crash metrics ($I_t / I_0$), and town shop demand tracking, causing them to dump crops into crashing markets or trigger catastrophic midnight shed waste.
3. **Value Head Gradient Instability**: Scalar MSE regression on raw cash balances spanning $\$3,000$ to $\$200,000+$ caused severe gradient explosion during training.
4. **Spatial-Economic Disconnect**: Spatial convolutional filters lacked real-time awareness of cash constraints and market spot prices.

## Solution

Build a unified **Hierarchical 2-Tier Architecture** that separates high-level strategic planning from low-level operational execution:
1. **Rich Observation Encoding**: A $24 \times 10 \times 10$ multi-layer spatial grid capturing land ownership, crop growth stages, soil moisture, fertilizer status, animal states, and worker density heatmaps, paired with a $72$-dimensional deep macro-economic state vector capturing market dynamics A-to-Z, town shop demand, wage liabilities, and opponent telemetry.
2. **FiLM-Modulated Convolutional Trunk**: Economic and market embeddings dynamically modulate spatial vision filters via Feature-wise Linear Modulation ($\gamma, \beta$), ensuring tile evaluation is financially contextualized.
3. **Decoupled Multi-Head Policy**: Independent factorized output heads for Crop Allocation ($5 \times 10 \times 10$), Livestock Target Herd Size (3 dims), Workforce Recruiting ($0..12$), Land Acquisition ($[0, 1]$), Autonomous Seed Replenishment (5 dims), and Continuous Market Liquidation ($9 \times [0, 1]$).
4. **Distributional Value & Win-Probability Critic**: A 64-bin two-hot log-cash distribution head trained via cross-entropy loss alongside an auxiliary win-probability head.
5. **Two-Stage Market Guardrails & Hungarian Micro Assignment**: Market fractions pass through deterministic cow feed reservation and midnight overflow protection, while tile routing and tool execution are solved via global linear sum assignment with biological urgency penalties.

## User Stories

1. As a competitive player, I want the bot to encode the 10×10 farm grid into a 24-channel spatial tensor, so that crop growth, soil moisture, animal health, and worker positions are cleanly separated.
2. As a competitive player, I want the bot to track spot prices, baseline ratios ($P / P_{\text{base}}$), and crash indicators ($I / I_0$) across all 9 commodities, so that crops are never dumped into collapsed markets.
3. As a competitive player, I want the bot to track town shop demand (Bakery, Pizza Shop, Brunch Spot, etc.), so that high-margin downstream purchasing opportunities are captured.
4. As a competitive player, I want the bot to extract the public opponent profile (cash balance, unlocked quadrants, workforce size, tile development), so that the agent can respond to competitor market pressure.
5. As a competitive player, I want the neural network to fuse spatial and scalar inputs using FiLM modulation, so that economic conditions directly guide spatial tile evaluation.
6. As a competitive player, I want the policy to output a decoupled $5 \times 10 \times 10$ crop priority heatmap, so that tile planting decisions across different quadrants do not conflict with each other.
7. As a competitive player, I want the policy to output dedicated livestock target headcounts for Cows, Sheep, and Geese, so that herd expansion is balanced against daily feed supplies.
8. As a competitive player, I want the policy to output a dedicated workforce hiring head ($0..12$), so that farmhand recruitment scales dynamically with seasonal labor demand.
9. As a competitive player, I want the policy to output a quadrant acquisition probability, so that land expansion ($1\text{k}, 2\text{k}, 4\text{k}$) occurs at financially optimal turns.
10. As a competitive player, I want the policy to output autonomous seed replenishment targets, so that required seeds are in stock without over-purchasing.
11. As a competitive player, I want the policy to output continuous market liquidation fractions ($[0, 1]$) per commodity, so that the agent can execute fine-grained inventory sell-offs.
12. As a competitive player, I want a Stage 2 cow feed reservation filter, so that wheat required for daily cow sustenance is never accidentally liquidated at market.
13. As a competitive player, I want a Stage 2 midnight shed overflow auto-liquidation guardrail, so that excess items above the 100-item shed capacity are monetized instead of trashed at day end.
14. As a competitive player, I want pre-softmax analytical action masking on hiring and land acquisition heads, so that illegal or bankrupting moves are eliminated before policy sampling.
15. As a competitive player, I want the Critic to output a 64-bin distributional two-hot log-cash distribution, so that value gradients remain stable across wide revenue scales.
16. As a competitive player, I want the Critic to output an auxiliary win-probability head, so that head-to-head match dominance is explicitly modeled.
17. As a competitive player, I want the Micro Assignment Engine to solve worker dispatch via Hungarian linear sum assignment, so that worker travel time is minimized with zero path collisions.
18. As a competitive player, I want biological urgency penalties integrated into the Hungarian cost matrix, so that starving animals, dry crops, and ripe harvests are serviced before low-priority tasks.
19. As a competitive player, I want the complete agent packaged into a single-file callable contract `agent(obs, config=None)`, so that it can be submitted directly to Kaggle and benchmarked locally.

## Implementation Decisions

### Observation Encoder
- Spatial tensor: 24 channels of shape $(24, 10, 10)$ containing one-hot crop identities, continuous growth/yield metrics, moisture $[0, 1]$, fertilizer flags, livestock types, shed footprints, and worker position densities.
- Scalar vector: 72 dimensions containing normalized phase clocks, cash metrics, cash velocity, wage liabilities, spot price ratios, inventory saturation ratios, town shop coverage, shed saturation percentage, seed reserves, and opponent public features.
- Temporal context: 1-step Markovian snapshot with engineered 24-turn momentum and velocity features.

### Neural Backbone
- 2-layer MLP projection converting 72-dim scalar features into a 128-dim economic embedding.
- 3-block Convolutional ResNet with 64 channels, modulated by FiLM affine parameters $(\gamma, \beta)$ computed from the economic embedding.
- Dual representations produced: spatial feature map $\mathbf{Z}_{\text{spatial}} \in \mathbb{R}^{64 \times 10 \times 10}$ and global pooled latent $\mathbf{z}_{\text{global}} \in \mathbb{R}^{256}$.

### Actor & Critic Heads
- Spatial Crop Head: $1 \times 1$ Conv yielding $5 \times 10 \times 10$ logits.
- Livestock Head: Linear projection yielding 3 count/categorical logits.
- Workforce Head: Linear projection yielding 13 discrete hiring logits with pre-softmax budget masks.
- Quadrant Head: Linear projection yielding 1 sigmoid unlock logit with affordability masks.
- Autonomous Seed Head: Linear projection yielding 5 crop seed purchasing logits.
- Market Intent Head: Linear projection yielding 9 continuous sigmoid fractions $[0, 1]$.
- Critic Value Head: 64-bin softmax logits over logarithmic terminal cash thresholds.
- Win-Probability Head: 1 sigmoid logit.

### Micro Assignment & Execution
- Two-stage market order compiler enforcing wheat reserve floors and end-of-day shed capacity constraints.
- Task generator building bipartite graph of $N$ farmhands to $M$ field tasks.
- Cost matrix $\mathbf{C}_{ij} = \text{Dist}(w_i, t_j) - \lambda \cdot \text{Logit}(t_j) + \text{Urgency}(t_j)$ solved via `scipy.optimize.linear_sum_assignment`.

## Testing Decisions

- **Primary Seam (Agent Level)**: Test through `agent(obs, config)` conforming to the standard competition API and evaluated against `"starter"`, `"random"`, and `"pass"` baselines via `kaggle_environments.make("kaggriculture")`.
- **Secondary Seam (Feature Encoder Level)**: Verify shape, finite bounds, and channel values of `encode_observation(obs)` on synthetic edge-case game states (bankrupt, fully unlocked, max animals, max shed).
- **Secondary Seam (Model Forward Pass Level)**: Verify that `forward(spatial, scalar, masks)` outputs all expected head tensors with correct dimensions and without NaN values.
- **Secondary Seam (Assignment Solver Level)**: Test Hungarian assignment under heavy task loads (12 workers, 30 tasks) ensuring 1-to-1 task uniqueness and zero collision steps.

## Out of Scope

- Offline dataset mining and replay downloading pipelines.
- Self-supervised imitation loss functions and multi-GPU training scripts.
- MCTS search loop hyperparameter tuning.
- Self-play reinforcement learning rollout workers.

## Further Notes

- All terminology strictly aligns with [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md).
- Follows the single-file submission contract specified in [AGENTS.md](file:///C:/Users/Manit/Desktop/kaggle/AGENTS.md).
