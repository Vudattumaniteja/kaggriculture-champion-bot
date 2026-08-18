"""
AlphaZero RL Champion Agent for Kaggriculture.
Loads 'weights/rl_alphazero_champion.pt' trained with 4-pillar reward shaping
(Win Margin + Explosive Growth - Capital Loss Penalty - Deadweight Trapped Asset Penalty).
"""

import os
import sys
from typing import Any, Dict

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.mcts import SSLMCTSAgent

_champion_weights_path = os.path.join(REPO_ROOT, "weights", "grandmaster_rl_champion.pt")
if not os.path.exists(_champion_weights_path):
    _champion_weights_path = "weights/grandmaster_rl_champion.pt"
if not os.path.exists(_champion_weights_path):
    _champion_weights_path = os.path.join(REPO_ROOT, "weights", "rl_alphazero_champion.pt")
if not os.path.exists(_champion_weights_path):
    _champion_weights_path = os.path.join(REPO_ROOT, "weights", "grandmaster_ssl.pt")

_rl_champion_agent = SSLMCTSAgent(
    model_weights_path=_champion_weights_path,
    num_simulations=60,
    c_puct=1.5,
    temperature=0.0,
    use_action_mask=True,
)


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    """Callable agent entrypoint for Kaggle Environments simulation and tournaments."""
    return _rl_champion_agent(obs, config)


if __name__ == "__main__":
    from kaggle_environments import make

    print("Testing RL Champion Bot against starter locally...")
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([agent, "starter"])
    print(f"Game finished! RL Champion: ${env.steps[-1][0].reward:,.1f} vs Starter: ${env.steps[-1][1].reward:,.1f}")
