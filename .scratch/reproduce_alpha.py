import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.evaluation.runner import run_match, run_tournament
from src.models.mcts import SSLMCTSAgent

agent_alpha = SSLMCTSAgent(
    model_weights_path="weights/ssl_alphazero.pt",
    num_simulations=20,
    c_puct=1.5,
    temperature=0.0,
)

summary = run_tournament(
    agent0=agent_alpha,
    agent1="starter",
    episodes=10,
    episode_steps=720,
    swap_positions=True,
    base_seed=42,
)
print("\n" + summary.format_markdown_table())
