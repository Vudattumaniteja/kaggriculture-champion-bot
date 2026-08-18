# Issue 09: Prioritized Fictitious Self-Play (PFSP) Single-Champion Matchmaker & Heuristic Warmup Curriculum

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/13  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  

---

## Objective
Implement the dynamic PFSP league tournament matchmaker for the single Champion network and the 80/20 heuristic fast-forward warmup curriculum generating midgame and endgame states in $<150\text{ms}$ on CPU.

## Acceptance Criteria
- [ ] Allocates 100% of gradient updates to the active Champion network.
- [ ] Matchmaking samples 40% from rolling FIFO pool of up to 30 past checkpoints ($P(i) \propto \max(0.05, (1 - \text{WinRate}_i)^{1.5})$), 40% self-play mirror with Dirichlet noise, and 20% fixed Mega League heuristic specialists.
- [ ] Curriculum fast-forwards 20% of matches into Midgame ($t \in [288, 432]$) and Endgame ($t \in [528, 648]$) in $<150\text{ms}$ on CPU.
- [ ] Sub-trajectory transitions $[t_{\text{start}}, 718]$ recurse GAE advantages cleanly without contaminating neural training with heuristic warmup steps.

## Verification Plan
- [ ] `python -m unittest tests/test_pfsp_curriculum.py` passes.
- [ ] Match simulation benchmark verifies 10 fast-forward curriculum matches execute in $< 1.5$ seconds total on CPU.

## Dependencies
- Blocked by: #12 (Issue 08: PER Buffer & Causal Logger)
