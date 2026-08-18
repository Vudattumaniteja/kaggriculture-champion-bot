import sys
sys.path.insert(0, '.')
import kaggle_environments
from scripts.test_full_hybrid import HybridMarketAgent

bot = HybridMarketAgent()
env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)

state = env.reset()
starter_fn = kaggle_environments.environments.get("kaggriculture")["agents"]["starter"]

for step in range(720):
    obs0 = state[0].observation
    obs1 = state[1].observation
    act0 = bot(obs0)
    act1 = starter_fn(obs1)
    state = env.step([act0, act1])

    day = obs0.get("day", 0)
    hour = obs0.get("hour", 0)
    if hour == 0 and day % 2 == 0:
        m0 = obs0["market"]
        print(f"Day {day:02d} | Prices: {m0['prices']}")
