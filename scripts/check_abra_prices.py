import json

with open("replays/top_competitors/episode-93954943-replay.json") as f:
    abra_data = json.load(f)

obs0 = abra_data["steps"][0][0]["observation"]
print("Abracadabra prices at Step 0:", obs0["market"]["prices"])

obs100 = abra_data["steps"][100][0]["observation"]
print("Abracadabra prices at Step 100:", obs100["market"]["prices"])

obs700 = abra_data["steps"][700][0]["observation"]
print("Abracadabra prices at Step 700:", obs700["market"]["prices"])
