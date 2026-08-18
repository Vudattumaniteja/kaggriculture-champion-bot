"""
80/20 Heuristic Fast-Forward Warmup Curriculum and Sub-Trajectory State Generator.
"""

from typing import Any, Dict, Optional, Tuple
import time
import numpy as np
from kaggle_environments import make

from src.league import HeuristicSpecialist


def sample_curriculum_start_step() -> int:
    """
    Samples episode start step according to 80/20 curriculum schedule:
    - 80% Turn 0
    - 10% Midgame: Day 12..18 (step 288..432)
    - 10% Endgame: Day 22..27 (step 528..648)
    """
    r = np.random.rand()
    if r < 0.80:
        return 0
    elif r < 0.90:
        day = np.random.randint(12, 19)
        return day * 24
    else:
        day = np.random.randint(22, 28)
        return day * 24


def fast_forward_match(
    start_step: int,
    bot1: Optional[Any] = None,
    bot2: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Fast-forwards a match to start_step using fast heuristic bots in < 150ms on CPU.
    Returns observation dictionary at start_step.
    """
    if start_step <= 0:
        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
        env.reset()
        obs = env.steps[0][0].get("observation", {})
        obs["player"] = 0
        return obs

    bot1 = bot1 or "starter"
    bot2 = bot2 or "starter"

    env = make("kaggriculture", configuration={"episodeSteps": start_step + 1}, debug=False)
    env.run([bot1, bot2])
    
    last_step_data = env.steps[-1][0]
    obs = last_step_data.get("observation", {})
    obs["player"] = 0
    return obs
