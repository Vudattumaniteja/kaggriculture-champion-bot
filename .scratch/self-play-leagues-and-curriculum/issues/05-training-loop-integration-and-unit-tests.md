# Training Loop Integration & Benchmark Unit Tests

Type: task
Status: open
Blocked by: 01, 02, 03, 04

## Question

How should the unified `PrioritizedFictitiousSelfPlayMatchmaker`, fast-forward jumpstart generator, and prioritized replay buffer be wired into `trainer.py`, with unit test benchmarks verifying that 10 fast-forward matches complete in <1.5s and sampling distributions strictly match the 40/40/20 sparring rule?
