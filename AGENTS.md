# Kaggriculture Agent Workspace Guide

This guide details agent development, local simulation benchmarking, and Kaggle CLI submission for the Kaggriculture competition.

## Game & Simulation Specifications
- **Horizon**: 720 turns (30 days × 24 turns/day).
- **Starting Bank**: $3,000 cash.
- **Board Grid**: 10×10 grid (4 quadrants: NW starting unlocked, NE $1k, SW $2k, SE $4k).
- **Shed Capacity**: 100 non-seed items at `(4,4)`, `(5,4)`, `(4,5)`, `(5,5)`. Excess at end-of-day discarded.
- **Scoring**: Winner determined strictly by highest cash balance at turn 719.

## Agent API & Contract
Your submission agent must be a callable `def agent(obs, config=None):` returning:
```python
{
    "farmer": [op, *args],  # Main farmer action
    "hands": [[op, *args], ...],  # Farmhands actions in order
    "market": [[op, *args], ...],  # Market order queue (max 10)
}
```

### Built-in Opponents
`kaggle-environments` provides three built-in agents:
- `"pass"`: Passes every turn.
- `"random"`: Selects random legal moves.
- `"starter"`: Deterministic baseline farmer.

## Local Benchmarking
```python
from kaggle_environments import make

env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
env.run([my_agent, "starter"])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
```

## Kaggle CLI Workflow
```bash
# Submit single-file bot
kaggle competitions submit kaggriculture -f submission.py -m "Agent v1"

# Check submissions
kaggle competitions submissions kaggriculture

# Download match replay
kaggle competitions replay <EPISODE_ID> -p ./replays
```
