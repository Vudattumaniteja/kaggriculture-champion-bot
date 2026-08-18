"""
Simulation optimizer and forensics script to test scaling of HRL 12-worker dispatcher.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kaggle_environments import make
from src.agents.hrl_12worker_dispatcher import HRL12WorkerHungarianDispatcher

def test_config(max_cows=30, max_sheep=30, fertilizer_ratio=0.70):
    agent_inst = HRL12WorkerHungarianDispatcher(
        max_cows=max_cows,
        max_sheep=max_sheep,
        fertilizer_sell_ratio=fertilizer_ratio,
    )
    def custom_agent(obs, config=None):
        return agent_inst(obs, config)
        
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 42}, debug=True)
    env.run([custom_agent, "starter"])
    
    r0 = env.steps[-1][0].reward
    r1 = env.steps[-1][1].reward
    print(f"Result (Cows={max_cows}, Sheep={max_sheep}): Bot=${r0:,.1f} | Starter=${r1:,.1f} | Margin=${r0-r1:+,.1f}")
    return r0

if __name__ == "__main__":
    for c, s in [(20, 20), (30, 30), (35, 35), (40, 40)]:
        test_config(c, s)
