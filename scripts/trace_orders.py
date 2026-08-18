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
    if act0.get("market") and (hour == 0 or len(act0["market"]) > 0):
        # check if money increased or if order succeeded
        if day in (11, 12, 13, 20, 25, 28, 29) and hour in (0, 1, 2, 3, 23):
            f0 = obs0["farms"][0]
            print(f"D{day:02d} H{hour:02d} | Money: ${f0['money']:<8.1f} | Market: {act0['market']}")
