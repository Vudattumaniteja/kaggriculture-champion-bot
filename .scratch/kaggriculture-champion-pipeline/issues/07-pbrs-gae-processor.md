# Issue 07: Maturity-Aware Potential-Based Reward Shaping (PBRS) & Trajectory GAE Processor

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/11  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Target Workspace**: `grilling model/` (`C:\Users\Manit\Desktop\kaggle\grilling model\`)  

---

## Objective
Implement `MaturityAwarePBRS` state potential with biological asset discounting schedules and `TrajectoryGAEProcessor` ($\gamma=0.995, \lambda=0.95$) in `grilling model/src/pbrs_gae.py` for step-wise temporal credit assignment and peak market arbitrage timing.

## Acceptance Criteria
- [ ] Implement `grilling model/src/pbrs_gae.py`:
  - Completely purges the piecewise $\$3\text{k}$ bank kink (`growth_bonus` / `cap_loss_penalty`) and flat Monte Carlo trajectory broadcast.
  - Potential function $\Phi(s, t) = \text{Cash}_t + \sum \omega_k(t) V_k(s_t)$ discounts immature crops, livestock break-even, unplanted seeds, and shed goods as $t \to 719$.
  - Mathematically proves exact reward telescoping $\sum_{t=0}^{T-1} r'_t \equiv \text{Cash}_T - \text{Cash}_0$ and enforces $2.0 \times \text{Deadweight}$ penalty at Turn 718.
  - Computes GAE advantages ($\gamma=0.995, \lambda=0.95$) and two-hot symlog target distributions over $[-15.0, +15.0]$.
- [ ] All code and tests strictly reside in `grilling model/`.

## Verification Plan
- [ ] `python -m unittest discover -s "grilling model/tests" -p "test_pbrs_gae.py"` passes.
- [ ] Test verifies that $\sum r'_t$ strictly equals terminal relative cash across 50 full and truncated synthetic match trajectories.

## Dependencies
- Blocked by: #8 (Issue 04: Micro Assignment & Guardrails)
