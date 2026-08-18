# Issue 06: Two-Scale Hierarchical World Model & Two-Group Staged AWIL Trainer

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/10  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Target Workspace**: `grilling model/` (`C:\Users\Manit\Desktop\kaggle\grilling model\`)  

---

## Objective
Implement within-day micro dynamics ($g_{\text{micro}}$) and seasonal macro day-skip dynamics ($g_{\text{day}}$) supervised by auxiliary yield and price decoders in `grilling model/src/world_model.py`, trained via a two-group staged trainer in `grilling model/src/trainer.py` with GradNorm policy balancing.

## Acceptance Criteria
- [ ] Implement `grilling model/src/world_model.py`:
  - Micro-step dynamics $g_{\text{micro}}(z_t, \mathbf{a}_{\text{strat}}) \to z_{t+1}$ models 1-step latent transitions in $\mathbb{R}^{128}$.
  - Macro day-skip dynamics $g_{\text{day}}(z_{d, 23}, \mathbf{a}_{\text{day}}) \to z_{d+1, 0}$ models 24-step day boundaries in single latent step for long-horizon seasonal MCTS.
  - Auxiliary physical decoders predict next-morning crop yields ($\hat{Y}_{d+1}$), commodity spot prices ($\hat{P}_{d+1}$), and town shop demand ($\hat{D}_{d+1}$).
- [ ] Implement staged trainer in `grilling model/src/trainer.py` using GradNorm for Group 1 policy heads while locking Group 2 value ($\lambda_{\text{val}}=1.0$) and auxiliary decoders ($\lambda_{\text{aux}}=0.1$) to fixed manual weights.
- [ ] Pre-trained checkpoint saves valid model weights to `grilling model/weights/` ready for self-play RL fine-tuning.
- [ ] All code and checkpoints strictly reside in `grilling model/`.

## Verification Plan
- [ ] `python -m unittest discover -s "grilling model/tests" -p "test_world_model_training.py"` passes.
- [ ] 5-epoch training dry-run completes with monotonic loss decrease and finite checkpoint weights.

## Dependencies
- Blocked by: #9 (Issue 05: AWIL Dataset Parser)
