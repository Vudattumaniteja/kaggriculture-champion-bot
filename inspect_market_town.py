import json

with open(r"C:\Users\Manit\Downloads\93943624.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

# Let's check town unlocks and sales
print("=== TOWN SHOPS OVER TIME ===")
town_unlocks = []
for i, s in enumerate(steps):
    t = s[0]["observation"].get("town", {})
    shops = t.get("unlocked_shops", [])
    if len(shops) > len(town_unlocks):
        new_shops = [x for x in shops if x not in town_unlocks]
        town_unlocks = list(shops)
        print(f"Step {i} (Day {s[0]['observation']['day']}, Hr {s[0]['observation']['hour']}): New shops = {new_shops} (All: {shops})")

print("\n=== ALL NON-EMPTY MARKET ORDERS ===")
for i, s in enumerate(steps):
    mo0 = s[0].get("action", {}).get("market", [])
    mo1 = s[1].get("action", {}).get("market", [])
    if mo0 or mo1:
        print(f"Step {i:03d} (D{s[0]['observation']['day']:02d}:H{s[0]['observation']['hour']:02d}):")
        if mo0:
            print(f"  P0: {mo0}")
        if mo1:
            print(f"  P1: {mo1}")
