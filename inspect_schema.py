import json
import os
from collections import defaultdict, Counter

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

# Let's inspect the game configuration, market prices, town shops, day-by-day cash, revenues, costs
print("Config:", replay.get("configuration"))
print("Spec:", replay.get("specification", {}).get("configuration"))

# Let's check step 0 observation details
obs0 = steps[0][0]["observation"]
print("Global keys in obs:", [k for k in obs0.keys() if k not in ["farms", "private"]])
print("Town shops at step 0:", obs0.get("shops", {}))
print("Market at step 0:", obs0.get("market", {}))
