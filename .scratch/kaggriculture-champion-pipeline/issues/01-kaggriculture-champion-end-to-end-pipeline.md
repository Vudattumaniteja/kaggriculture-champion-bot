# Issue 01: Kaggriculture Unified Champion Bot End-to-End Pipeline

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4  
**Spec Reference**: [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Tracked GitHub Issues**:
- [GitHub Issue #1: Hierarchical ML/SSL/RL Bot Architecture & I/O Pipeline](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/1)
- [GitHub Issue #2: Advantage-Weighted Imitation Learning (AWIL) & Two-Scale Hierarchical World Model Pre-Training Pipeline](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/2)
- [GitHub Issue #3: RL Rewards, Potential-Based Shaping (PBRS), GAE Credit Assignment & League Ecosystem](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/3)

---

## Summary

Implement and validate the complete end-to-end championship pipeline for Kaggriculture:

1. **Tier 1 (Hierarchical 2-Tier Architecture & I/O Pipeline)**:
   - $(24, 10, 10)$ spatial encoder + $72$-dim scalar economic & market state vector.
   - FiLM-modulated ResNet convolutional trunk conditioning visual perception on financial dynamics.
   - Decoupled multi-head policy: spatial crop heatmaps ($5 \times 10 \times 10$), livestock herd quotas (3 dims), workforce recruitment ($0..12$), land unlocks ($[0, 1]$), autonomous seed replenishment (5 dims), and continuous market liquidation ($9 \times [0, 1]$).
   - 1001-bin two-hot symlog categorical value critic + auxiliary win-probability head.
   - Two-stage market guardrails (cow wheat reserve, midnight shed overflow) & Hungarian micro task dispatcher with biological urgency penalties.

2. **Tier 2 (AWIL Pre-Training & Two-Scale Hierarchical World Model)**:
   - High-quality replay parsing ($> \$50\text{k}$, deduplicated PASS steps, both players) with D4 coordinate augmentation.
   - Two-stage AWIL advantage scoring standardized across 5 seasonal phase buckets with clipped sample weights $w_t \in [0.2, 5.0]$.
   - Two-scale hierarchical world model ($g_{\text{micro}}$ within-day, $g_{\text{day}}$ day-skip) supervised by auxiliary yield, price, and town shop demand decoders.
   - Two-group staged loss balancing: GradNorm policy balancing + fixed value ($\lambda_{\text{val}}=1.0$) and auxiliary ($\lambda_{\text{aux}}=0.1$) coefficients.

3. **Tier 3 (Maturity-Aware PBRS, GAE Credit Assignment & League Ecosystem)**:
   - Purge $3\text{k}$ bank kink and flat Monte Carlo broadcast.
   - Maturity-Aware PBRS state potential with biological asset discounting and $2.0 \times$ terminal deadweight surcharge.
   - GAE temporal credit assignment ($\gamma=0.995, \lambda=0.95$).
   - 500k-transition Prioritized Experience Replay buffer logging verified causal macro actions.
   - Single-Champion PFSP league matchmaker (40% historical checkpoints, 40% mirror self-play, 20% Mega League specialists).
   - 80/20 curriculum fast-forward warmup in $<150\text{ms}$ with sub-trajectory GAE credit recursion.
   - Mask-safe Dirichlet root exploration ($\alpha=0.3, \epsilon=0.25$) and cosine policy entropy decay ($0.05 \to 0.002$).

---

## Testing Seams

- **Seam 1 (Agent Environment Seam - Highest)**: `agent(obs, config=None)` evaluated on `kaggle_environments.make("kaggriculture")` against standard `"starter"` baseline.
- **Seam 2 (Model & Decoupled Policy Seam)**: `network(spatial_tensor, scalar_vector, action_masks)` tensor output shapes, pre-softmax masks, and unit-sum value distributions.
- **Seam 3 (Trajectory GAE & PBRS Seam)**: `TrajectoryGAEProcessor.process_match(observations, value_estimates)` reward telescoping, advantage bounds, and Day 28 zero-potential deadweight discounting.
- **Seam 4 (AWIL Dataset & World Model Seam)**: `build_awil_dataset(raw_replays, output_path)` and `imagine_day_skip(z_23, a_day)` quality filtering, D4 equivariance, and non-collapsing auxiliary decoders.
- **Seam 5 (PFSP League & Fast-Forward Curriculum Seam)**: `AlphaGoatMegaLeague.sample_opponent()` and `fast_forward_match(t_start)` probability weighting and $<150\text{ms}$ execution latency.
