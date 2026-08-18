# Issue 03: Decoupled Multi-Head Policy with Pre-Softmax Analytical Action Masking

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/7  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Target Workspace**: `grilling model/` (`C:\Users\Manit\Desktop\kaggle\grilling model\`)  

---

## Objective
Build the decoupled factorized policy heads in `grilling model/src/network.py` for spatial crop heatmaps ($5 \times 10 \times 10$), livestock quotas, workforce recruitment, quadrant acquisition, autonomous seed replenishment, and continuous market liquidation with pre-softmax analytical masking.

## Acceptance Criteria
- [ ] Implement multi-head actor projections in `grilling model/src/network.py`:
  - Spatial crop head emits $5 \times 10 \times 10$ logits conditioned on owned quadrant masks.
  - Workforce hiring head emits 13 logits masked by wage affordability $\text{Workers} \times \$20 \le \text{Cash}$.
  - Quadrant acquisition head emits unlock logit masked by quadrant unlock costs ($\$1\text{k}, \$2\text{k}, \$4\text{k}$).
  - Autonomous seed head emits replenishment logits for 5 crop varieties with inventory buffer bounds.
  - Market intent head emits 9 continuous sigmoid liquidation fractions $[0, 1]$.
- [ ] Pre-softmax masks clamp all illegal or bankrupting action logits to $-\infty$ before softmax/sampling.
- [ ] All code and tests strictly reside in `grilling model/`.

## Verification Plan
- [ ] `python -m unittest discover -s "grilling model/tests" -p "test_policy_heads.py"` passes.
- [ ] Action mask tester verifies that illegal actions receive zero probability mass across 50 constrained test scenarios.

## Dependencies
- Blocked by: #6 (Issue 02: Backbone Trunk & Critic)
