"""
AlphaZero MCTS Submission Agent for Kaggriculture.
Combines deep Policy-Value neural network evaluations with MCTS search and deterministic chore routing.
"""

from typing import Any, Dict

from src.models.mcts import AlphaZeroMCTSAgent

_alphazero_agent = AlphaZeroMCTSAgent(
    model_weights_path="weights/pretrained_alphazero.pt",
    num_simulations=15,
)


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _alphazero_agent(obs, config)
