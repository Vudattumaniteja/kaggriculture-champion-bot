import os
import sys

sys.path.insert(0, os.path.abspath("."))

import numpy as np
from kaggle_environments import make
from submission import agent as my_agent

cashes = []
starter_cashes = []
for ep in range(10):
    seed = 1000 + ep
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
    env.run([my_agent, "starter"])
    c0 = env.steps[-1][0]["reward"]
    c1 = env.steps[-1][1]["reward"]
    cashes.append(c0)
    starter_cashes.append(c1)
    print(f"Episode {ep:02d} (Seed {seed}): MyAgent = ${c0:,.0f} | Starter = ${c1:,.0f} | Margin = ${c0 - c1:+,.0f}")

print("-" * 65)
print(f"Mean MyAgent Cash: ${np.mean(cashes):,.0f} (Min: ${np.min(cashes):,.0f}, Max: ${np.max(cashes):,.0f})")
print(f"Mean Starter Cash: ${np.mean(starter_cashes):,.0f}")
print(f"Win Rate: {sum(1 for c, s in zip(cashes, starter_cashes) if c > s) / len(cashes) * 100:.1f}%")
