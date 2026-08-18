import sys
sys.path.insert(0, '.')
import kaggle_environments
from scripts.test_hybrid_hrl import HybridMarketAgent

bot = HybridMarketAgent()
env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)

state = env.reset()
starter_fn = kaggle_environments.environments.get("kaggriculture")["agents"]["starter"]

for step in range(120):
    obs0 = state[0].observation
    obs1 = state[1].observation
    act0 = bot(obs0)
    act1 = starter_fn(obs1)
    state = env.step([act0, act1])

    day = obs0.get("day", 0)
    hour = obs0.get("hour", 0)
    if step < 24 or hour == 0:
        f0 = obs0["farms"][0]
        p0 = obs0["private"]
        print(f"Step {step:03d} (D{day:02d}:H{hour:02d}) Money=${f0['money']:<6.1f} Farmer: {act0['farmer']} @ {f0['farmer']} | Market: {act0['market']}")
        print(f"    Hands ({len(act0['hands'])}): {act0['hands']}")
        print(f"    Shed: {p0['shed']}")
        print(f"    Seeds: {p0['seeds']}")
        print(f"    Inventories: {p0.get('inventories')}")
