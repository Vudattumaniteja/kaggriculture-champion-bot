# Issue 01: Advantage-Weighted Imitation Learning (AWIL) & Two-Scale Hierarchical World Model Pre-Training Pipeline

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/2  
**Spec Reference**: [.scratch/kaggriculture-awil-pretraining/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-awil-pretraining/spec.md)  

**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Grilling Session Reference**: [docs/grilling_session_ssl_and_imitation_learning.md](file:///C:/Users/Manit/Desktop/kaggle/docs/grilling_session_ssl_and_imitation_learning.md)  

---

## Summary
Implement the complete offline pre-training pipeline for the Kaggriculture championship agent:
1. **Replay Parser & Target Extractor**: Extract decoupled spatial crop heatmaps ($5 \times 10 \times 10$), livestock quotas, workforce headcount targets ($0..12$), exponential land expansion ramps, and masked liquidation fractions from 440,000 Grandmaster replay transitions.
2. **Spatial Feature Channel Mapping**: Embed `Quadrant_Cost_Map`, `Distance_to_Shed`, and `Shop_Vectors` into the $24 \times 10 \times 10$ spatial grid, with synchronized D4 action coordinate augmentation.
3. **Quality Filtering & Curation**: Filter out games $< \$50\text{k}$, prune static duplicate PASS transitions, retain qualifying trajectories from both players, and enforce a per-player contribution cap.
4. **AWIL Advantage Scorer**: Train a Stage 1 value baseline $V_\phi(s)$, standardize advantages $\hat{A}_t$ within seasonal phase buckets, and compute clipped sample weights $w_t = g_i \cdot \operatorname{clip}(e^{\hat{A}_t / \tau}, 0.2, 5.0)$.
5. **Two-Scale Hierarchical World Model**: Train micro-step dynamics ($g_{\text{micro}}$) for within-day transitions and macro day-skip dynamics ($g_{\text{day}}$) for long-horizon seasonal MCTS, supervised by auxiliary next-day yield ($\hat{Y}_{d+1}$) and price ($\hat{P}_{d+1}$) decoders.
6. **Two-Group Staged Trainer**: Implement `train_awil.py` with Group 1 GradNorm policy balancing and Group 2 fixed value/auxiliary loss coefficients ($\lambda_{\text{val}}=1.0, \lambda_{\text{aux}}=0.1$).

## Testing Seams
- **AWIL Training Seam**: `train_awil(dataset, config) -> (policy_loss, value_loss, aux_loss, model_checkpoint)` convergence and gradient stability.
- **Dataset Seam**: `build_awil_dataset(raw_replays, output_path) -> dataset_stats` quality filtering, spatial channel validation, and advantage weight bounds $w_t \in [0.2, 5.0]$.
- **Hierarchical World Model Seam**: `imagine_micro(z_t, a_strat)` and `imagine_day_skip(z_23, a_day)` forward simulation and auxiliary decoding.
