import os
import sys

sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from src.agents.hrl_12worker_dispatcher import agent as hrl_agent
import numpy as np

cashes = []
starter_cashes = []
for ep in range(5):
    seed = 1000 + ep
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([hrl_agent, "starter"])
    c0 = env.steps[-1][0]["reward"]
    c1 = env.steps[-1][1]["reward"]
    cashes.append(c0)
    starter_cashes.append(c1)
    print(f"Seed {seed}: HRL Agent = ${c0:,.0f} vs Starter = ${c1:,.0f}")

print(f"Mean HRL Cash: ${np.mean(cashes):,.0f} (Min: ${np.min(cashes):,.0f}, Max: ${np.max(cashes):,.0f})")
