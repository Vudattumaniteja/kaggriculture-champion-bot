import sys
sys.path.insert(0, '.')
import kaggle_environments
import json
from scripts.test_full_hybrid import HybridMarketAgent

bot = HybridMarketAgent()
env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
env.run([bot, "starter"])

# Dump replay to scratch
with open(".scratch/our_run.json", "w") as f:
    json.dump(env.toJSON(), f)

print("Saved .scratch/our_run.json. Comparing with Abracadabra replay...")

with open("replays/top_competitors/episode-93954943-replay.json") as f:
    abra_data = json.load(f)

with open(".scratch/our_run.json") as f:
    our_data = json.load(f)

print(f"{'Day':<4} | {'Our Money':<10} | {'Abra Money':<10} | {'Our Crops':<20} | {'Abra Crops':<20} | {'Our Shed':<25} | {'Abra Shed':<25}")
print("-" * 120)

for day in range(30):
    s_idx = day * 24
    our_obs = our_data["steps"][s_idx][0]["observation"]
    abra_obs = abra_data["steps"][s_idx][0]["observation"]

    our_f = our_obs["farms"][0]
    abra_f = abra_obs["farms"][0]

    our_crops = {}
    for r in our_f["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                c = t.get("crop")
                our_crops[c] = our_crops.get(c, 0) + 1

    abra_crops = {}
    for r in abra_f["tiles"]:
        for t in r:
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                c = t.get("crop")
                abra_crops[c] = abra_crops.get(c, 0) + 1

    our_shed_str = str({k: v for k, v in our_obs["private"]["shed"].items() if v > 0})
    abra_shed_str = str({k: v for k, v in abra_obs["private"]["shed"].items() if v > 0})

    print(f"D{day:02d}  | ${our_f['money']:<9.0f} | ${abra_f['money']:<9.0f} | {str(our_crops):<20} | {str(abra_crops):<20} | {our_shed_str:<25} | {abra_shed_str:<25}")
