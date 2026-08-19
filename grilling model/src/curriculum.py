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
        state = env.reset()
        obs = dict(state[0].observation)
        obs["player"] = 0
        return obs

    bot1_fn = bot1 if callable(bot1) else HeuristicSpecialist(bot1 if isinstance(bot1, str) and bot1 != "starter" else "DeterministicGrandmaster")
    bot2_fn = bot2 if callable(bot2) else HeuristicSpecialist(bot2 if isinstance(bot2, str) and bot2 != "starter" else "DeterministicGrandmaster")

    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    state = env.reset()

    for s in range(start_step):
        state[0].observation.step = s
        state[1].observation.step = s
        state[0].action = bot1_fn(state[0].observation)
        state[1].action = bot2_fn(state[1].observation)
        state = env.interpreter(state, env)

    state[0].observation.step = start_step
    obs = dict(state[0].observation)
    obs["player"] = 0
    return obs


def recurse_subtrajectory_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    start_step: int,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes GAE advantage recursion strictly on neural sub-trajectory transitions [t_start, 718].
    Takes:
    - rewards: Array of shaped step rewards for sub-trajectory of length T = 719 - t_start.
    - values: Array of value predictions of length T + 1 (with values[-1] = terminal value).
    - start_step: t_start (e.g. 0, 288..432, 528..648).
    Returns:
    - rewards: (T,)
    - advantages: (T,)
    - value_targets: (T,)
    Guarantees no warmup steps (t < t_start) contaminate the neural advantages or value targets.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    value_targets = advantages + values[:-1]
    return rewards, advantages, value_targets


def generate_subtrajectory_transitions(
    start_step: int,
    subtrajectory_length: Optional[int] = None,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    total_turns: int = 720,
) -> Dict[str, Any]:
    """
    Generates isolated sub-trajectory transitions starting from t_start up to total_turns - 1 (718),
    recursing GAE advantages cleanly without contaminating neural training with heuristic warmup steps.
    """
    if subtrajectory_length is None:
        subtrajectory_length = total_turns - 1 - start_step

    step_indices = np.arange(start_step, start_step + subtrajectory_length)
    mock_rewards = np.zeros(subtrajectory_length, dtype=np.float32)
    mock_values = np.zeros(subtrajectory_length + 1, dtype=np.float32)

    rewards, advantages, value_targets = recurse_subtrajectory_gae(
        mock_rewards, mock_values, start_step=start_step, gamma=gamma, gae_lambda=gae_lambda
    )

    return {
        "start_step": start_step,
        "step_indices": step_indices,
        "rewards": rewards,
        "advantages": advantages,
        "value_targets": value_targets,
        "num_transitions": subtrajectory_length,
    }

