# Parameterized Heuristic Specialist Archetypes

Type: prototype
Status: closed
Blocked by: none
GitHub Issue: #27

## Question

How should the 5 heuristic specialist archetypes (DeterministicGrandmaster, CarrotMonoculture, MelonRusher, DairySyndicate, TownShopSaturator) be parameterized and integrated into the `HeuristicSpecialist` engine using the existing high-performance chore and pathfinding logic from `submission_deterministic.py`?

## Resolution

- Implemented `SpecialistConfig` and `ParameterizedHeuristicFarmer` in `grilling model/prototype/specialists.py`.
- Configured presets for `DeterministicGrandmaster`, `CarrotMonoculture`, `MelonRusher`, `DairySyndicate`, and `TownShopSaturator`.
- Integrated with `HeuristicSpecialist` in `grilling model/src/league.py`.
- Added unit tests in `grilling model/prototype/test_specialists.py` (all tests passing).
- Executed 10 pairwise matches and generated HTML replays via `grilling model/prototype/run_tournament.py`.
