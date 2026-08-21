# Self-Play League Architecture & Curriculum Design Map

## Destination

A validated, modular Self-Play League and Accelerated Curriculum subsystem in `grilling model/` featuring 5 extreme heuristic specialist archetypes, a loss-weighted PFSP historical rematch ladder, an 80/10/10 fast-forward jumpstart generator running in <150ms, and sub-trajectory prioritized experience replay for weak matches.

## Notes

- Domain: Kaggriculture reinforcement learning self-play and curriculum simulation.
- Skills: `/grilling`, `/domain-modeling`, `/prototype`, `/unslop`.
- Location: All implementations live in `grilling model/src/` and `grilling model/tests/`.
- Standards: Follow the 40/40/20 sparring rule and 80/10/10 fast-forward warmup curriculum documented in `teach/learning-records/0006-self-play-leagues-and-curriculum-design.md`.

## Decisions so far

<!-- index of closed tickets will be recorded here -->
<!-- GitHub Map Issue: https://github.com/Vudattumaniteja/kaggriculture-champion-bot/issues/26 -->
<!-- Child Tickets:
  #27 Parameterized Heuristic Specialist Archetypes (wayfinder:prototype)
  #28 PFSP Checkpoint and Loss-Weighted Rematch Ladder (wayfinder:grilling)
  #29 Fast-Forward Warmup Pairing Matrix & State Diversity (wayfinder:prototype)
  #30 Sub-Trajectory Prioritized Experience Replay for Weak Matches (wayfinder:grilling)
  #31 Training Loop Integration & Benchmark Unit Tests (wayfinder:task)
-->

## Not yet specified

<!-- Fog of war: in-scope questions to revisit as the frontier advances -->

- Adaptive curriculum phase shifting: When and how the 80/10/10 jumpstart ratio should transition to 90/5/5 or 100/0/0 as the policy approaches grandmaster convergence.
- Checkpoint Elo rating decay: Whether historical checkpoints should undergo Elo rating decay or retirement if the champion achieves a 100% win rate over 50 consecutive matches.
- Multi-process rollout parallelism: Scaling CPU fast-forward rollouts across multiple worker cores if batch size demands increase.

## Out of scope

- Distributed multi-node GPU rollout clusters (single-machine training is the target).
- Modifying simulation engine files outside `grilling model/`.
