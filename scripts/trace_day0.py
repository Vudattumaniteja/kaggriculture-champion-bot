import sys
sys.path.insert(0, '.')
import kaggle_environments
from scripts.test_full_hybrid import HybridMarketAgent

bot = HybridMarketAgent()
env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)

state = env.reset()
starter_fn = kaggle_environments.environments.get("kaggriculture")["agents"]["starter"]

print("=== DAY 0 TURN BY TURN ===")
for step in range(24):
    obs0 = state[0].observation
    obs1 = state[1].observation
    act0 = bot(obs0)
    act1 = starter_fn(obs1)
    state = env.step([act0, act1])

    f0 = obs0["farms"][0]
    p0 = obs0["private"]
    print(f"H{step:02d} | Farmer {f0['farmer']}: {act0['farmer']} | Hands {f0['hands']}: {act0['hands']}")
    print(f"     Inv: {p0.get('inventories')} | Shed: {p0.get('shed')} | Seeds: {p0.get('seeds')}")
