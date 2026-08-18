import json

with open("replays/top_competitors/episode-93954943-replay.json") as f:
    data = json.load(f)

steps = data["steps"]

for day in range(30):
    step_idx = day * 24
    obs = steps[step_idx][0]["observation"]
    f0 = obs["farms"][0]
    p0 = obs["private"]
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
    print(f"Day {day:02d} | Money: ${f0['money']:<8.1f} | Crops: {crops} | Animals: {animals} | Shed: {p0['shed']} | HiresToday: {f0['hires_today']}")
