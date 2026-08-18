import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
from kaggle_environments import make
from src.agents.hrl_12worker_dispatcher import agent

for seed in [42, 43, 44, 45, 46]:
    # Match 1: agent as P0
    env0 = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': seed}, debug=True)
    env0.run([agent, 'starter'])
    r0 = env0.steps[-1][0].reward
    
    # Match 2: agent as P1
    env1 = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': seed}, debug=True)
    env1.run(['starter', agent])
    r1 = env1.steps[-1][1].reward
    
    print(f"Seed {seed}: P0={r0:.1f}, P1={r1:.1f}")
