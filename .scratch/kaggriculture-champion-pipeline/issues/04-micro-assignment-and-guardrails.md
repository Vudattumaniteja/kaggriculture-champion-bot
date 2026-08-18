# Issue 04: Two-Stage Market Guardrails & Neural-Weighted Hungarian Micro Assignment Solver

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/8  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  

---

## Objective
Implement the operational micro execution engine that translates continuous policy liquidation fractions through cow feed and shed overflow guardrails, and solves farmhand chore routing via bipartite linear sum assignment with biological urgency penalties.

## Acceptance Criteria
- [ ] Cow feed reservation clamps Wheat market sell orders to preserve at least $\text{Cows} \times 2$ daily feed in shed.
- [ ] Midnight shed overflow guardrail automatically liquidates excess goods $> 100$ items at $t \pmod{24} == 23$ to prevent engine discard.
- [ ] Cost matrix $\mathbf{C}_{ij} = \text{Dist}(w_i, t_j) - \lambda \cdot \text{Logit}(t_j) + \text{Urgency}(t_j)$ solved via Hungarian matching (`scipy.optimize.linear_sum_assignment`).
- [ ] Biological urgency overrides prioritize starving animals, dry crops, and ripe harvests with zero worker-worker path collisions.
- [ ] Assembles final valid submission action dictionary matching Kaggle format `{farmer: [...], hands: [...], market: [...]}`.

## Verification Plan
- [ ] `python -m unittest tests/test_micro_assignment.py` passes.
- [ ] 100-step simulation against environment verifies 100% legal moves and zero path collisions under 12-worker load.

## Dependencies
- Blocked by: #7 (Issue 03: Policy Heads & Masking)
