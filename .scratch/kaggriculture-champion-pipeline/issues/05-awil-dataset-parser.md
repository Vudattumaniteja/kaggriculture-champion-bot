# Issue 05: Grandmaster Replay Dataset Parser with D4 Coordinate Augmentation & AWIL Scorer

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/9  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  

---

## Objective
Implement high-volume replay parsing of 440k Grandmaster transitions with quality filtering ($> \$50\text{k}$), PASS deduplication, exponential land lead-in ramps ($e^{-\Delta t / 12}$), synchronized D4 spatial augmentation, and phase-bucketed AWIL advantage sample weighting.

## Acceptance Criteria
- [ ] Replay parser extracts rolling $5 \times 10 \times 10$ crop heatmaps, exponential land expansion ramps, masked trade fractions, and livestock quotas.
- [ ] Quality filter removes replays with terminal return $< \$50\text{k}$, deduplicates consecutive idle PASS transitions, and caps per-player representation.
- [ ] Synchronized D4 augmentation rotates/reflects spatial grids and action targets $(r, c) \to (c, 9-r)$ while keeping scalar economic features invariant.
- [ ] Stage 1 baseline $V_\phi(s)$ standardizes advantages $\hat{A}_t$ across 5 seasonal phase buckets and computes clipped sample weights $w_t \in [0.2, 5.0]$.

## Verification Plan
- [ ] `python -m unittest tests/test_awil_dataset.py` passes.
- [ ] Dataset integrity script validates that all generated sample weights satisfy $0.2 \le w_t \le 5.0$ and D4 transforms preserve spatial validity.

## Dependencies
- Blocked by: #8 (Issue 04: Micro Assignment & Guardrails)
