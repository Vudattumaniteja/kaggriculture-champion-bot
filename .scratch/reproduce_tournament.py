import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.evaluation.runner import run_match, run_tournament

print("Running tournament with ssl_bot vs starter...")
summary = run_tournament(
    agent0="src/agents/ssl_bot.py",
    agent1="starter",
    episodes=10,
    episode_steps=720,
    swap_positions=True,
    base_seed=42,
)
print("\n" + summary.format_markdown_table())
