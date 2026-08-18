import json

with open('replays/top_competitors/episode-93954943-replay.json') as f:
    replay = json.load(f)

steps = replay['steps']

for d in [1, 5, 10, 15, 20]:
    t = d * 24 + 12
    obs = steps[t][0]['observation']
    farm = obs['farms'][0]
    tiles = farm['tiles']
    print(f"\n================ Day {d} Tile Map ================")
    for r in range(10):
        row_str = []
        for c in range(10):
            if (c, r) in [(4, 4), (5, 4), (4, 5), (5, 5)]:
                row_str.append("SHED")
            else:
                tile = tiles[r][c]
                if tile == "LOCKED":
                    row_str.append("LOCK")
                elif tile is None:
                    row_str.append("....")
                elif isinstance(tile, dict):
                    k = tile.get("kind")
                    if k == "PLANT":
                        crop = tile.get("crop")
                        age = d - tile.get("planted_day", d)
                        fert = "F" if tile.get("fertilized_until_day", -1) >= d else "."
                        row_str.append(f"{crop[:2]}{age}{fert}")
                    elif k in ("PASTURE", "COOP"):
                        a = tile.get("animal", "EMP")
                        row_str.append(f"P:{a[:2]}")
                    elif k == "WEED":
                        row_str.append("WEED")
                    else:
                        row_str.append(str(k)[:4])
        print(f"R{r:02d}: " + " ".join(f"{x:>4}" for x in row_str))
