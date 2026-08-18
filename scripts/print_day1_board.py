import sys
sys.path.insert(0, '.')
import kaggle_environments
from scripts.test_hybrid_hrl import HybridMarketAgent

bot = HybridMarketAgent()
env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)

state = env.reset()
starter_fn = kaggle_environments.environments.get("kaggriculture")["agents"]["starter"]

for step in range(24):
    obs0 = state[0].observation
    obs1 = state[1].observation
    act0 = bot(obs0)
    act1 = starter_fn(obs1)
    state = env.step([act0, act1])

obs0 = state[0].observation
f0 = obs0["farms"][0]
print("Day 1 Board:")
for r in range(5):
    row = []
    for c in range(5):
        t = f0["tiles"][r][c]
        row.append(str(t))
    print(f"R{r}: {row}")
