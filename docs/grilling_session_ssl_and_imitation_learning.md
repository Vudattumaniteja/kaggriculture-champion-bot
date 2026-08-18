# Master Specification: Supervised Learning (SL), AWIL & Hierarchical World Model Pre-Training Pipeline

**Status**: `LOCKED & APPROVED`  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Architecture Spec Reference**: [.scratch/kaggriculture-architecture/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-architecture/spec.md)  

---

## 1. Executive Summary

This specification locks down the complete dataset engineering, advantage-weighted imitation learning, and self-supervised world dynamics pre-training pipeline for the Kaggriculture championship agent.

Through systematic design tree resolution, we have eliminated the four fatal flaws of previous behavioral cloning bots (label squashing, worker chore thrashing, Frankenstein state daydreaming, and 1-ply search degeneracy).

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     SUPERVISED & SELF-SUPERVISED PRE-TRAINING PIPELINE                           │
├────────────────────────────────┬────────────────────────────────┬────────────────────────────────┤
│ 1. Multi-Head Rolling Targets  │ 2. Advantage-Weighted AWIL     │ 3. Two-Scale Hierarchical Dyn  │
│ (Dense Tile Allocation 5x10x10 │ (Sample Weighting via          │ (Micro t->t+1 within day +     │
│ & Exponential Land Lead-in)    │ w_t = g_i * clip(exp(A/tau)))  │ Macro Day-Skip d->d+1 MCTS)    │
├────────────────────────────────┴────────────────────────────────┴────────────────────────────────┤
│ 4. Spatial Feature Channel Mapping (Invariants embedded in 10x10 planes + Equivariant D4)        │
│ 5. Two-Group Staged Loss Balancing (Policy GradNorm + Fixed Value/Aux Weights)                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Settled Decisions Across the 5 Design Tree Frontiers

### Frontier 1: Replay Target Extraction (Decoupled Heads without Label Squashing)
- **Decision**: **Option 1B — Dense Rolling Farm-State Allocation & Exponential Lead-in Ramps**.
- **Specification**:
  1. **Spatial Crop Heatmap ($5 \times 10 \times 10$)**: Target for each owned tile $(r, c)$ is the active crop type currently growing or designated across a rolling horizon ($t \to t+12$), providing dense spatial gradient signals across all 100 tiles every turn.
  2. **Livestock Target (3 dims)**: Continuous target herd capacity $(\text{cows}, \text{sheep}, \text{geese})$ maintained by the Grandmaster during the current game phase.
  3. **Workforce Head ($0..12$)**: Target is the total active workforce headcount $N \in [0..12]$ maintained at turn $t$.
  4. **Land Unlock Probability ($[0, 1]$)**: Exponential ramp target $y_t = \exp(-\Delta t / 12)$ for the 12 turns leading up to a land purchase at $t_{\text{buy}}$, teaching the network to anticipate capital allocation and prepare for expansion.
  5. **Autonomous Seed Head (5 dims)**: Seed purchasing buffer targets extracted from active seed acquisition orders.
  6. **Continuous Market Liquidation ($9 \times [0, 1]$)**: Continuous liquidation fraction $\frac{\text{sold}}{\text{inventory}}$ on turns with market transactions; items with 0 inventory are masked out of the loss.

---

### Frontier 2: Self-Supervised World Dynamics (Two-Scale Hierarchical World Model)
- **Decision**: **Option 2D — Two-Scale Hierarchical World Model ($g_{\text{micro}} + g_{\text{day}}$)**.
- **Specification**:
  1. **Micro-Dynamics ($g_{\text{micro}}: z_t \times a_t \to z_{t+1}$)**: Predicts immediate step-by-step state transitions within the current day ($t \to t+1$) for worker movement, immediate action point costs, and inventory changes.
  2. **Macro Day-Skip Dynamics ($g_{\text{day}}: z_{d, 23} \times a_{\text{macro}} \to z_{d+1, 0}$)**: Takes the end-of-day latent state $z_{d, 23}$ and predicts the next morning's state $z_{d+1, 0}$ in 1 single step, enabling MCTS to plan across the 30-day season in 3–5 macro steps.
  3. **Auxiliary Foresight Decoders**: Ground the latent space $z$ by supervising next-morning physical decoders:
     - **Crop Yield & Maturity Grid ($\hat{Y}_{d+1} \in \mathbb{R}^{2 \times 10 \times 10}$)**: Expected tile maturity and soil moisture.
     - **Town Market Spot Prices ($\hat{P}_{d+1} \in \mathbb{R}^9$)**: Expected commodity price ratios $\frac{P_{d+1}}{P_{\text{base}}}$.
     - **Town Shop Absorption ($\hat{D}_{d+1} \in \mathbb{R}^3$)**: Expected downstream demand elasticity.

---

### Frontier 3: Dataset Curation & Advantage-Weighted Imitation Learning (AWIL)
- **Decision**: **Advantage-Weighted Imitation Learning with Multi-Tier Quality Filter**.
- **Specification**:
  1. **Quality Floor**: Prune all matches where the final score is $< \$50,000$.
  2. **Deduplication**: Remove identical, static PASS-spam transitions where no actions or state deltas occur.
  3. **Both Players Ingested**: Include both Player 0 and Player 1 trajectories if their individual return $\ge \$50,000$.
  4. **Diversity Cap**: Limit maximum transitions from any single player/game to prevent mode collapse on one specific playstyle.
  5. **Value Baseline Model**: Train a state-value baseline $V_\phi(s_t) \to R_t$ using distributional two-hot regression.
  6. **Advantage Calculation**:
     $$\hat{A}_t = \frac{(R_t - V_\phi(s_t)) - \mu_A}{\sigma_A + \epsilon}$$
  7. **Transition Weighting**:
     $$w_t = g_i(\text{game quality}) \cdot \operatorname{clip}\left(\exp\left(\frac{\hat{A}_t}{\tau}\right), w_{\min}, w_{\max}\right)$$
     where $g_i \in [1.0, 4.0]$, $\tau = 1.0$, $w_{\min} = 0.2$, $w_{\max} = 5.0$.

---

### Frontier 4: Spatial Feature Channel Mapping & Equivariant Data Augmentation
- **Decision**: **Spatial Feature Channel Maps with Synchronized D4 Action Transforms**.
- **Specification**:
  1. **Spatial Tensor ($24 \times 10 \times 10$)**:
     - *Dynamic Channels*: Player ownership, crop stages, soil moisture, seeds, animals, worker positions.
     - *Static Invariant Channels*: `Quadrant_Cost_Map` ($0, 0.25, 0.50, 1.00$), `Distance_to_Shed` field, `Distance_to_Shop` field, `Shop_Delta_Row`, `Shop_Delta_Col`, and `Unlock_Status_Map`.
  2. **D4 Geometric Augmentation**:
     - Apply random rotations ($k \in \{0, 1, 2, 3\} \times 90^\circ$) and random horizontal flips to the full $24 \times 10 \times 10$ tensor.
     - Synchronously rotate spatial action target coordinates: $(r, c) \to (c, 9-r)$ under $90^\circ$ clockwise rotation.
     - Leave global scalar features and non-spatial market orders invariant.

---

### Frontier 5: Multi-Task Loss Formulation & Balancing
- **Decision**: **Option D — Two-Group Staged Loss Balancing**.
- **Specification**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}}^{\text{GradNorm}} + \lambda_{\text{val}} \mathcal{L}_{\text{val}} + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}$$
  1. **Group 1 (Strategic Policy Heads)**: Dynamically balance gradients across `crop_head`, `workforce_head`, `market_head`, `land_head`, and `seed_head` using GradNorm / equal gradient norm scaling.
  2. **Group 2 (Value & World Dynamics Heads)**: Locked to fixed manual coefficients:
     - $\lambda_{\text{val}} = 1.0$ (64-bin Distributional Two-Hot Categorical Cross-Entropy).
     - $\lambda_{\text{aux}} = 0.1$ (MSE on next-day price ratios and crop yield grids).

---

## 3. Master Training Loss Objective

$$\mathcal{L}_{\text{AWIL}}(\theta) = -\sum_{t} w_t \cdot \left[ \sum_{k \in \text{Heads}} \lambda_k(t) \log \pi_\theta^{(k)}(a_t^{(k)} \mid s_t) \right] + 1.0 \cdot \mathcal{L}_{\text{val}}(s_t, R_t) + 0.1 \cdot \mathcal{L}_{\text{aux}}(s_t, s_{d+1})$$

where:
- $\mathcal{L}_{\text{crop}}$: Spatial Cross-Entropy masked on owned/farmable tiles.
- $\mathcal{L}_{\text{work}}$: 13-class Cross-Entropy with pre-softmax budget masks.
- $\mathcal{L}_{\text{quad}}$: Weighted Binary Cross-Entropy ($w_{\text{pos}} = 8.0$) for expansion timing.
- $\mathcal{L}_{\text{mkt}}$: Masked Binary Cross-Entropy on active inventory fractions.
- $\mathcal{L}_{\text{val}}$: Two-Hot Cross-Entropy over 64 logarithmic bins.
- $\mathcal{L}_{\text{aux}}$: $\text{MSE}(\hat{P}_{d+1}, P_{d+1}) + \text{MSE}(\hat{Y}_{d+1}, Y_{d+1})$.

---

## 4. Implementation Checklist

- [ ] **Feature Extraction**: Update `src/models/encoder.py` to construct the 24-channel spatial tensor with embedded `Quadrant_Cost_Map`, `Distance_to_Shed`, and `Shop_Vectors`.
- [ ] **Replay Parser**: Update `src/training/build_top_leaderboard_dataset.py` with dense rolling tile allocation, exponential land lead-in, and $< \$50\text{k}$ pruning.
- [ ] **Value Baseline Trainer**: Train $V_\phi(s)$ on the curated dataset to compute advantages $\hat{A}_t$.
- [ ] **AWIL Data Loader**: Implement PyTorch Dataset with synchronized D4 spatial transforms and clipped advantage weights $w_t$.
- [ ] **Two-Scale World Model**: Integrate $g_{\text{micro}}$ and $g_{\text{day}}$ into `src/models/` for long-horizon seasonal MCTS.
- [ ] **Staged Loss Trainer**: Implement `src/training/train_awil.py` with Group 1 GradNorm and Group 2 fixed value/aux weights.
