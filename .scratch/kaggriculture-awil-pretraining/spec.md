# Feature Specification: Advantage-Weighted Imitation Learning (AWIL) & Two-Scale Hierarchical World Model Pre-Training Pipeline

**Status**: `ready-for-agent`  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Architecture Spec Reference**: [.scratch/kaggriculture-architecture/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-architecture/spec.md)  
**Master Grilling Blueprint**: [docs/grilling_session_ssl_and_imitation_learning.md](file:///C:/Users/Manit/Desktop/kaggle/docs/grilling_session_ssl_and_imitation_learning.md)  

---

## Problem Statement

Previous imitation and self-supervised pre-training pipelines in Kaggriculture suffered from four catastrophic failure modes that prevented neural agents from matching Grandmaster performance:
1. **Flashcard Mislabling (Label Squashing)**: Squashing 12 simultaneous farmhand actions into a single discrete macro-action caused $42.4\%$ of dataset steps to collapse onto feeding cows, while rare high-leverage actions like purchasing land were squashed to $0.25\%$, starving the network of expansion gradients.
2. **Action Blindness & Outcome Conflation**: Uniform behavioral cloning treated all turns in a $\$140\text{k}$ match as equally optimal (learning the player's blunders), while completely discarding brilliant opening crop rotations from $\$80\text{k}$ games that collapsed in the endgame.
3. **Dynamics Representation Mismatch (Frankenstein States)**: World-model heads predicted future scalar bank balances but paired them with frozen spatial grid images, causing forward lookahead to evaluate rich endgame cash on empty starting farms.
4. **Spatial Asymmetry & Coordinate Memorization**: Convolutional layers memorized arbitrary board orientations and hardcoded quadrant positions instead of learning spatial-economic invariants (cost-to-shed, cost-to-unlock, distance-to-shop).

## Solution

Build a unified **Advantage-Weighted Imitation Learning (AWIL) & Two-Scale Hierarchical World Model Pre-Training Pipeline**:
1. **Decoupled Multi-Head Target Extraction**: Extract dense rolling spatial crop heatmaps ($5 \times 10 \times 10$), continuous livestock herd capacities, active workforce recruitment counts ($0..12$), exponential land expansion lead-in ramps ($e^{-\Delta t / 12}$), and masked inventory liquidation fractions from 440,000 Grandmaster replay transitions.
2. **Two-Scale Hierarchical World Model**: Train coordinated micro-dynamics ($g_{\text{micro}}: z_t \times \mathbf{a}_{\text{strat}} \to z_{t+1}$) for within-day unit simulation and macro day-skip dynamics ($g_{\text{day}}: z_{d, 23} \times \mathbf{a}_{\text{day}} \to z_{d+1, 0}$) supervised by next-morning physical decoders ($\hat{Y}_{d+1}$ yields, $\hat{P}_{d+1}$ prices, $\hat{D}_{d+1}$ shop demand).
3. **Advantage-Weighted Sample Scoring (AWIL)**: Filter replays with a $\$50\text{k}$ quality floor and duplicate-pass pruning, train a state-value baseline $V_\phi(s)$, standardize advantages $\hat{A}_t = \frac{R_t - V(s_t) - \mu_A}{\sigma_A}$ within seasonal phase buckets, and weight transition losses by $w_t = g_i \cdot \operatorname{clip}\left(e^{\hat{A}_t / \tau}, w_{\min}, w_{\max}\right)$.
4. **Spatial Feature Channel Mapping & Equivariant D4 Augmentation**: Embed quadrant costs, shed distance fields, shop vectors, and unlock states directly into the $24 \times 10 \times 10$ spatial grid, applying synchronized D4 spatial rotations to tensors and action targets $(r, c) \to (c, 9-r)$.
5. **Two-Group Staged Loss Balancing**: Dynamically balance policy action heads using GradNorm while locking distributional value ($\lambda_{\text{val}} = 1.0$) and auxiliary world decoders ($\lambda_{\text{aux}} = 0.1$) to fixed manual weights.

## User Stories

1. As a machine learning engineer, I want the replay parser to extract $5 \times 10 \times 10$ rolling crop allocation heatmaps from replay farm tiles, so that the crop head receives dense spatial supervisory gradients on every turn.
2. As a machine learning engineer, I want the replay parser to generate an exponential lead-in ramp $y_t = \exp(-\Delta t / 12)$ for the 12 turns prior to land acquisition, so that the policy learns to anticipate capital allocation before expansion.
3. As a machine learning engineer, I want the replay parser to compute continuous market liquidation fractions $\frac{\text{sold}}{\text{inventory}}$ and mask empty inventory items, so that trade supervision reflects proportional inventory strategy.
4. As a machine learning engineer, I want the dataset generator to prune matches with final score $< \$50,000$, so that low-quality random baselines do not pollute the imitation pool.
5. As a machine learning engineer, I want the dataset generator to prune identical consecutive PASS transitions, so that zero-action idle steps do not cause policy apathy.
6. As a machine learning engineer, I want the dataset generator to include both Player 0 and Player 1 transitions if both scored $\ge \$50,000$, so that competitive dual-player dynamics are captured.
7. As a machine learning engineer, I want the dataset generator to cap maximum transitions contributed by any single player/match, so that the model does not overfit to one specific playstyle.
8. As a machine learning engineer, I want to train a value baseline network $V_\phi(s)$ predicting terminal returns $R_t$ in Stage 1, so that state-value expectations are established before policy weighting.
9. As a machine learning engineer, I want advantage estimates $\hat{A}_t = R_t - V_\phi(s_t)$ normalized independently within 5 seasonal phase buckets, so that early-game opening moves are not overshadowed by late-game dollar scales.
10. As a machine learning engineer, I want transition sample weights computed via $w_t = g_i \cdot \operatorname{clip}(e^{\hat{A}_t / \tau}, 0.2, 5.0)$, so that brilliant decisions are strongly imitated while mistakes are suppressed.
11. As a machine learning engineer, I want the observation encoder to embed quadrant unlock costs, shed distance fields, and shop direction vectors into $10 \times 10$ spatial feature channels, so that spatial filters learn invariant economic geometry.
12. As a machine learning engineer, I want data augmentation to apply synchronized D4 rotations and reflections to the $24 \times 10 \times 10$ spatial tensor and action target coordinates $(r, c) \to (c, 9-r)$, so that effective training data volume is quadrupled without breaking spatial validity.
13. As a machine learning engineer, I want non-spatial market orders and global economic scalars left invariant during D4 augmentation, so that commodity pricing is not corrupted.
14. As a machine learning engineer, I want the Two-Scale Hierarchical World Model to train micro-step dynamics $g_{\text{micro}}(z_t, \mathbf{a}_{\text{strat}}) \to z_{t+1}$, so that within-day unit movements and action point expenditures are modeled in latent space.
15. As a machine learning engineer, I want the Two-Scale Hierarchical World Model to train macro day-skip dynamics $g_{\text{day}}(z_{d, 23}, \mathbf{a}_{\text{day}}) \to z_{d+1, 0}$, so that long-horizon seasonal MCTS can plan in 3–5 macro steps.
16. As a machine learning engineer, I want auxiliary next-morning decoders predicting crop yield grids ($\hat{Y}_{d+1}$), commodity spot prices ($\hat{P}_{d+1}$), and town shop demand ($\hat{D}_{d+1}$), so that the latent embedding $z$ is physically grounded and protected against representation collapse.
17. As a machine learning engineer, I want policy action heads balanced dynamically using GradNorm, so that gradients from the spatial crop head do not overpower workforce hiring or land acquisition.
18. As a machine learning engineer, I want the 64-bin distributional two-hot value head and auxiliary decoders locked to fixed manual loss weights ($\lambda_{\text{val}}=1.0, \lambda_{\text{aux}}=0.1$), so that value calibration remains unconditionally stable during policy optimization.

## Implementation Decisions

### Dataset Extraction & Quality Curation
- Raw replay parsing extracts decoupled targets: spatial crop heatmaps ($5 \times 10 \times 10$) on owned tiles, livestock target headcounts (3 dims), workforce headcount ($0..12$), exponential land expansion ramps, seed inventory buffer deficits (5 dims), and masked liquidation fractions ($9 \times [0, 1]$).
- Dataset filtering applies a strict $\$50\text{k}$ terminal return threshold, deduplicates consecutive static PASS steps, ingests both qualifying players, and caps per-player representation to maintain strategic diversity.

### Advantage-Weighted Imitation (AWIL) Pipeline
- 2-Stage training schedule: Stage 1 trains and freezes the distributional value baseline $V_\phi(s)$ alongside auxiliary foresight decoders.
- Stage 2 computes phase-standardized advantages across 5 seasonal phase buckets (Days 0–5, 6–12, 13–18, 19–24, 25–29) and serializes clipped sample weights $w_t \in [0.2, 5.0]$.
- Policy training optimizes weighted imitation loss $-\sum_t w_t \log \pi_\theta(a_t \mid s_t)$.

### Spatial Channel Mapping & D4 Augmentation
- $24 \times 10 \times 10$ spatial feature map contains dynamic farm channels paired with static spatial maps: normalized `Quadrant_Cost_Map` ($0, 0.25, 0.50, 1.00$), normalized `Distance_to_Shed` field, `Distance_to_Shop` field, `Shop_Delta_Row`, `Shop_Delta_Col`, and `Unlock_Status_Map`.
- D4 augmentation randomly rotates ($k \times 90^\circ$) and flips the 24-channel grid while synchronously rotating spatial target coordinates $(r, c) \to (c, 9-r)$. Non-spatial actions and scalar features remain invariant.

### Two-Scale Hierarchical World Model
- Micro dynamics $g_{\text{micro}}(z_t, \mathbf{a}_{\text{strat}}) \to z_{t+1}$ operates on 128-dim latent states conditioned on high-level strategic policy vectors.
- Macro day-skip dynamics $g_{\text{day}}(z_{d, 23}, \mathbf{a}_{\text{day}}) \to z_{d+1, 0}$ predicts next morning's state in a single transition, supervised by auxiliary physical decoders ($\hat{Y}_{d+1} \in \mathbb{R}^{2 \times 10 \times 10}$ yields/moisture, $\hat{P}_{d+1} \in \mathbb{R}^9$ spot price ratios, $\hat{D}_{d+1} \in \mathbb{R}^3$ shop demand).

### Two-Group Staged Loss Balancing
- Group 1 (Strategic Policy Heads: Crop, Workforce, Land, Seed, Market) dynamically balanced via GradNorm.
- Group 2 (Distributional Value Head $\lambda_{\text{val}} = 1.0$ and Auxiliary Foresight Decoders $\lambda_{\text{aux}} = 0.1$) locked to fixed manual weights.

## Testing Decisions

### Seam 1: AWIL Pre-Training Seam (Highest / Primary Seam)
- **Interface**: `train_awil(dataset_path, config) -> (policy_loss, value_loss, aux_loss, model_checkpoint)`
- **Validation**: Verify that training on the curated dataset converges smoothly over epochs, GradNorm balances policy gradients across all factorized heads, and checkpoint weights output finite tensors without NaN or inf values.

### Seam 2: Dataset Generation & AWIL Scoring Seam
- **Interface**: `build_awil_dataset(raw_replays, output_path) -> dataset_stats`
- **Validation**: Verify that replays $< \$50\text{k}$ and duplicate PASS steps are pruned, spatial feature channels contain correct normalized cost/distance values, D4 transforms preserve spatial validity and rotate action targets correctly, and computed advantage weights satisfy $w_t \in [0.2, 5.0]$.

### Seam 3: Two-Scale Hierarchical World Model Seam
- **Interface**: `imagine_micro(z_t, a_strat) -> z_next` and `imagine_day_skip(z_23, a_day) -> z_morning`
- **Validation**: Verify that latent rollouts produce 128-dim embeddings that decode into valid next-morning crop yields ($\hat{Y}_{d+1}$) and commodity spot prices ($\hat{P}_{d+1}$) without representation collapse.

## Out of Scope

- Live Kaggle submission compilation and packaging (handled in Issue #1).
- Hungarian worker micro pathfinding and tool execution (handled in Issue #1).
- Self-play reinforcement learning rollout workers and Gumbel MuZero tree search (handled in Issue #3).

## Further Notes

- All domain terminology strictly conforms to [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md).
- Follows the locked 2-Tier Architecture blueprint specified in [.scratch/kaggriculture-architecture/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-architecture/spec.md).
