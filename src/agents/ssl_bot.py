"""
SSL AlphaZero MCTS Submission Agent for Kaggriculture.
Integrates SSLPolicyValueNet (Self-Supervised World Model Dynamics & Policy-Value Heads)
with Monte Carlo Tree Search (MCTS) and macro-action routing across crops, livestock,
fertilizer management, market arbitrage, and end-game liquidation.
"""

import os
import sys
from typing import Any, Dict

try:
    from src.models.mcts import SSLMCTSAgent
except ImportError:
    # Fallback if working directory is inside src
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from src.models.mcts import SSLMCTSAgent

_weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../weights/grandmaster_ssl.pt"))
if not os.path.exists(_weights_path):
    _weights_path = "weights/grandmaster_ssl.pt"
if not os.path.exists(_weights_path):
    # Fallback to ssl_alphazero.pt if grandmaster_ssl.pt is not yet built
    _weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../weights/ssl_alphazero.pt"))
    if not os.path.exists(_weights_path):
        _weights_path = "weights/ssl_alphazero.pt"

_ssl_mcts_agent = SSLMCTSAgent(
    model_weights_path=_weights_path,
    num_simulations=20,
    c_puct=1.5,
    temperature=0.0,
)


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    """Callable agent entrypoint for Kaggle Environments simulation and tournaments."""
    return _ssl_mcts_agent(obs, config)


if __name__ == "__main__":
    from kaggle_environments import make
    print("Testing SSL Bot against starter locally...")
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([agent, "starter"])
    print(f"Game finished! Reward: {env.steps[-1][0].reward} vs Starter: {env.steps[-1][1].reward}")
