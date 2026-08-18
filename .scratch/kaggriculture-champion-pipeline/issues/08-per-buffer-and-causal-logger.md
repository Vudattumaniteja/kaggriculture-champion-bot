# Issue 08: 500k Verified Prioritized Experience Replay (PER) & Causal Action Logger

**Status**: `ready-for-agent`  
**GitHub Issue**: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/12  
**Parent Spec**: [Master Spec (Issue #4)](https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/4) | [.scratch/kaggriculture-champion-pipeline/spec.md](file:///C:/Users/Manit/Desktop/kaggle/.scratch/kaggriculture-champion-pipeline/spec.md)  
**Domain Context**: [CONTEXT.md](file:///C:/Users/Manit/Desktop/kaggle/CONTEXT.md)  
**Target Workspace**: `grilling model/` (`C:\Users\Manit\Desktop\kaggle\grilling model\`)  

---

## Objective
Implement a high-performance 500,000-transition Prioritized Experience Replay buffer in `grilling model/src/replay_buffer.py` using binary SumTree proportional sampling ($\alpha=0.6, \beta: 0.4 \to 1.0$) that records verified causal actions dispatched by the Hungarian engine.

## Acceptance Criteria
- [ ] Implement `grilling model/src/replay_buffer.py`:
  - Binary SumTree data structure supports $O(\log N)$ priority updates and proportional sampling across 500k transitions.
  - Logs verified executed macro actions $a_{\text{eff}}$ rather than unexecutable raw target intents to prevent replay buffer corruption.
  - Computes importance sampling correction weights $w_i = (N \cdot P(i))^{-\beta} / \max(w)$ with linear annealing $\beta: 0.4 \to 1.0$.
  - Supports fast batch sampling, disk serialization, and checkpoint resuming.
- [ ] All code and data files strictly reside in `grilling model/`.

## Verification Plan
- [ ] `python -m unittest discover -s "grilling model/tests" -p "test_replay_buffer.py"` passes.
- [ ] Replay buffer throughput benchmark verifies $> 5,000$ sampled transitions/sec on CPU.

## Dependencies
- Blocked by: #11 (Issue 07: PBRS & GAE Processor)
