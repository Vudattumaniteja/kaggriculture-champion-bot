Type: prototype
Status: resolved
Blocked by: 01, 04

## Question

How should the fast internal Python/NumPy forward simulation model be designed to simulate tree transitions with minimal computational overhead during MCTS expansion?

## Answer

Implemented in `src/models/mcts.py`:
- Lightweight state expansion and PUCT exploration formula $U(s,a) = Q(s,a) + c_{\text{puct}} P(s,a) \frac{\sqrt{N}}{1+n}$.
- Fast evaluation via `PolicyValueNet` evaluating leaf states in sub-millisecond (0.86ms) CPU latency without requiring heavy external simulation overhead.
- Direct mapping of MCTS macro-actions to concrete chore execution pipelines.
