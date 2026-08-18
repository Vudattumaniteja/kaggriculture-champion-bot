# Issue 10: Mask-Safe Dirichlet Exploration, Cosine Entropy Decay & Standalone Submission Packaging

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/14  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Target Workspace**: `grilling model/` (`C:\Users\Manit\Desktop\kaggle\grilling model\`)  

---

## Objective
Implement mask-safe Dirichlet root noise ($\alpha=0.3, \epsilon=0.25$), cosine-annealed policy entropy loss ($0.05 \to 0.002$), two-stage Gumbel evaluation temperature, and automated standalone packaging into a single-file `grilling model/submission.py` with base85 FP16 weights.

## Acceptance Criteria
- [ ] Dirichlet root noise perturbs policy priors strictly over valid action masks.
- [ ] Policy entropy regularization coefficient anneals via cosine schedule from $0.05 \to 0.002$.
- [ ] Evaluation Gumbel search temperature switches from $\tau=1.0$ ($t < 48$) to $\tau=0.0$ ($t \ge 48$).
- [ ] Automated packaging routine `grilling model/build_submission.py` serializes FP16 weights into single-file `grilling model/submission.py` complying with `agent(obs, config=None)` Kaggle contract.
- [ ] Full 720-step verification test passes against `"starter"` baseline in `kaggle_environments.make("kaggriculture")` using `grilling model/submission.py`.
- [ ] All code, packaging scripts, and submission files strictly reside in `grilling model/`.

## Verification Plan
- [ ] `python -m unittest discover -s "grilling model/tests" -p "test_submission_packaging.py"` passes.
- [ ] Packaging test generates `grilling model/submission.py` and runs a full match against `"starter"`, verifying score $> \$80\text{k}$ and 100% legal syntax.

## Dependencies
- Blocked by: #10 (Issue 06: World Model & AWIL Trainer), #13 (Issue 09: PFSP Matchmaker & Curriculum)
