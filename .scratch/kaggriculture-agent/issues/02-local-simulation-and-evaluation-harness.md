Type: prototype
Status: resolved
Blocked by: 01

## Question

How should the local multi-match benchmarking harness and tournament runner be implemented using `kaggle-environments` to simulate head-to-head episodes, track win rates and mean net worth, and generate HTML replay traces?

## Answer

Implemented local simulation and tournament harness in `src/evaluation/runner.py` with top-level CLI `evaluate.py`:

1. **Features**:
   - `run_match()`: Runs a single 720-step match between any two agents (built-ins, script paths, or functions).
   - `run_tournament()`: Simulates $N$ episodes with automatic positional swapping (Player 0 / Player 1 alternating) to eliminate first-player bias.
   - Replay Export: Automatically generates standalone interactive HTML replays in `replays/html/` and raw JSON logs in `replays/json/`.
   - Metrics: Win/loss/tie tracking, mean bank balances, standard deviations, profit margins, and markdown scoreboard generation.
2. **Verification**:
   - Tested 3-episode tournament of `starter` vs `random` in ~3.2s total (1.07s/match). `starter` achieved 100% win rate with $3,551 mean profit vs $270 for `random`.
