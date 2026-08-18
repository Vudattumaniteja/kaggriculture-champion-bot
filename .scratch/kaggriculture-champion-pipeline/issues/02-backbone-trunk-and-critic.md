# Issue 02: FiLM-Modulated SE-ResNet Backbone Trunk & 1001-Bin Symlog Value Critic

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/6  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Target Workspace**: `grilling model/` (`C:\Users\Manit\Desktop\kaggle\grilling model\`)  

---

## Objective
Implement the neural backbone trunk in `grilling model/src/network.py` fusing 72-dim scalar features through FiLM affine modulation onto a 3-block SE-ResNet spatial feature map, paired with a 1001-bin two-hot symlog categorical value critic and auxiliary win-probability head.

## Acceptance Criteria
- [ ] Implement `ChampionBackbone` in `grilling model/src/network.py` projecting 72-dim scalars to 128-dim economic embedding $\mathbf{z}_{\text{econ}}$, generating affine parameters $(\gamma, \beta)$ that modulate 64-channel SE-ResNet convolutions.
- [ ] Forward pass outputs spatial feature map $\mathbf{Z}_{\text{spatial}} \in \mathbb{R}^{64 \times 10 \times 10}$ and global pooled latent $\mathbf{z}_{\text{global}} \in \mathbb{R}^{256}$.
- [ ] Distributional value head outputs 1001 categorical logits over $[-15.0, +15.0]$ symlog space ($[-\$3.26\text{M}, +\$3.26\text{M}]$).
- [ ] Two-hot soft target transformation and categorical cross-entropy loss function correctly compute value gradients without MSE explosion.
- [ ] Win-probability head outputs 1 sigmoid logit.
- [ ] All code and tests strictly reside in `grilling model/`.

## Verification Plan
- [ ] `python -m unittest discover -s "grilling model/tests" -p "test_backbone_and_critic.py"` passes.
- [ ] Forward and backward gradient propagation passes on batch of 32 synthetic inputs without NaN loss.

## Dependencies
- Blocked by: #5 (Issue 01: Observation Encoder)
