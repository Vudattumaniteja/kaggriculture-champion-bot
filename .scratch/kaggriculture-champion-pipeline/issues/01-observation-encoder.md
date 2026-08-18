# Issue 01: 24-Channel Spatial & 72-Dim Scalar Observation Encoder with Invariant Geometric Fields

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/5  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  

---

## Objective
Implement the complete observation encoder converting raw Kaggriculture environment observations into a 24-channel $(24, 10, 10)$ spatial tensor (with embedded `Quadrant_Cost_Map`, `Distance_to_Shed`, and `Shop_Vectors`) and a 72-dimensional scalar economic state vector.

## Acceptance Criteria
- [ ] `encode_observation(obs)` outputs `x_spatial` of shape $(24, 10, 10)$ and `x_scalar` of shape $(72,)$ with bounded finite values.
- [ ] Channels 0–18 cleanly represent crop one-hot identities, growth ratios, moisture, fertilizer flags, livestock types/hunger, shed/delivery zones, and farmhand density.
- [ ] Channels 19–23 correctly contain invariant geometry fields: normalized quadrant cost map ($0, 0.25, 0.50, 1.00$), shed distance transform, shop distance transform, and shop row/col delta vectors.
- [ ] Scalar vector includes turn clocks ($t/720, \text{phase}/24$), symlog cash, projected wage liabilities, 9 commodity spot prices, baseline ratios ($P_i/P_{i,\text{base}}$), crash indicators ($I_i/I_{i,0}$), town shop demands, and opponent public telemetry.
- [ ] Handles edge-case states gracefully (bankrupt, fully unlocked board, max animals, full shed) without NaNs or infs.

## Verification Plan
- [ ] `python -m unittest tests/test_encoder.py` passes.
- [ ] Verification script validates tensor shapes, channel bounds, and normalization across 100 sample game steps.

## Dependencies
- Blocked by: None — can start immediately.
