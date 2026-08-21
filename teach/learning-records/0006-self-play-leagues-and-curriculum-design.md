# Self-Play League Architecture: The 40/40/20 Sparring Rule and 80/10/10 Jumpstart Curriculum

Pure self-play causes intransitive Rock-Paper-Scissors cycling and catastrophic forgetting. The optimal league architecture allocates 40% to active self-play mirror, 40% to Prioritized Fictitious Self-Play (PFSP) historical checkpoints weighted by loss rate, and 20% to fixed deterministic heuristic specialists. Combining this with an 80/10/10 fast-forward warmup curriculum concentrates training compute on critical midgame expansion and endgame liquidation decisions with isolated sub-trajectory GAE.
