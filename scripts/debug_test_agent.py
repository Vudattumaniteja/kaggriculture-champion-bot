import kaggle_environments
from scripts.test_hybrid_agent import HybridMarketAgent

bot = HybridMarketAgent()
env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)

# Run step by step and track stats
state = env.reset()
for step in range(720):
    obs0 = state[0].observation
    obs1 = state[1].observation
    act0 = bot(obs0)
    # starter agent
    starter_fn = kaggle_environments.environments.get("kaggriculture")["agents"]["starter"]
    act1 = starter_fn(obs1)
    state = env.step([act0, act1])

    day = obs0.get("day", 0)
    hour = obs0.get("hour", 0)
    if hour == 0:
        f0 = obs0["farms"][0]
        p0 = obs0["private"]
        m0 = obs0["market"]
        crops = {}
        animals = {}
        for row in f0["tiles"]:
            for tile in row:
                if isinstance(tile, dict):
                    k = tile.get("kind")
                    if k == "PLANT":
                        c = tile.get("crop")
                        crops[c] = crops.get(c, 0) + 1
                    elif k in ("PASTURE", "COOP"):
                        a = tile.get("animal", "EMPTY")
                        animals[a] = animals.get(a, 0) + 1
        print(f"Day {day:02d} | Money: ${f0['money']:<8.1f} | Crops: {crops} | Animals: {animals} | Shed: {p0['shed']} | Seeds: {p0['seeds']}")

print("Final Reward:", [s.reward for s in state])
