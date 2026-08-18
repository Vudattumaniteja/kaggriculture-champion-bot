import json

with open("replays/top_competitors/episode-93954943-replay.json") as f:
    data = json.load(f)

steps = data["steps"]

# Check fertilizer generation and livestock harvesting
total_fert_collected = 0
total_milk_harvested = 0
total_wool_harvested = 0

for s in range(len(steps)):
    act = steps[s][0].get("action", {})
    farmer_act = act.get("farmer", [])
    hands_act = act.get("hands", [])
    all_acts = [farmer_act] + hands_act
    for a in all_acts:
        if a:
            if a[0] == "COLLECT_FERTILIZER":
                total_fert_collected += 1
            elif a[0] == "HARVEST":
                pass
    m = act.get("market", [])
    for order in m:
        if order and order[0] == "BUY_PRODUCT" and order[1] == "FERTILIZER":
            print(f"Buy fert: {order}")

print(f"Total COLLECT_FERTILIZER actions: {total_fert_collected}")

# Check market buy/sell for fertilizer
for s in range(len(steps)):
    act = steps[s][0].get("action", {})
    m = act.get("market", [])
    obs = steps[s][0]["observation"]
    day = obs["day"]
    hour = obs["hour"]
    for order in m:
        if order and order[0] == "SELL" and order[1] == "FERTILIZER":
            if day in (5, 10, 15, 20, 25, 26, 27, 28, 29) and hour in (0, 1, 12, 23):
                print(f"D{day:02d} H{hour:02d} | Sold fert: {order} | Money: ${obs['farms'][0]['money']}")
