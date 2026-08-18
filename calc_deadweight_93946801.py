import json
import os

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]
obs0 = steps[-1][0]["observation"]
obs1 = steps[-1][1]["observation"]

farm0 = obs0["farms"][0]
farm1 = obs1["farms"][1]

shed0 = obs0["private"]["shed"]
shed1 = obs1["private"]["shed"]

seeds0 = obs0["private"]["seeds"]
seeds1 = obs1["private"]["seeds"]

# Pricing of assets:
# Seeds purchase price: Carrot: 10, Wheat: 20, Tomato: 50, Strawberry: 80, Melon: 80
# Animals: Goose: 100, Sheep: 400, Cow: 1000
# Structures: Coop: 50, Pasture: 100
# Land: NE: 1000, SW: 2000, SE: 4000
# Crops on field: seed price or expected harvest value

print("=== DEADWEIGHT ASSETS AT TURN 719 ===")
print("\n--- Player 0 (alfphafarm) ---")
print("Shed Inventory:", {k: v for k, v in shed0.items() if v > 0})
print("Seed Inventory:", {k: v for k, v in seeds0.items() if v > 0})
# Tiles
p0_structures = []
p0_crops = []
p0_weeds = 0
for r in range(10):
    for c in range(10):
        t = farm0["tiles"][r][c]
        if isinstance(t, dict):
            k = t.get("kind")
            if k in ["COOP", "PASTURE"]:
                p0_structures.append((r, c, k, t.get("animal")))
            elif k == "PLANT":
                p0_crops.append((r, c, t.get("crop"), t.get("stage"), t.get("watered")))
            elif k == "WEED":
                p0_weeds += 1
print(f"Structures ({len(p0_structures)}):", p0_structures)
print(f"Field Crops ({len(p0_crops)}):", p0_crops)
print(f"Weeds: {p0_weeds}")

print("\n--- Player 1 (Ankit0017) ---")
print("Shed Inventory:", {k: v for k, v in shed1.items() if v > 0})
print("Seed Inventory:", {k: v for k, v in seeds1.items() if v > 0})
p1_structures = []
p1_crops = []
p1_weeds = 0
for r in range(10):
    for c in range(10):
        t = farm1["tiles"][r][c]
        if isinstance(t, dict):
            k = t.get("kind")
            if k in ["COOP", "PASTURE"]:
                p1_structures.append((r, c, k, t.get("animal")))
            elif k == "PLANT":
                p1_crops.append((r, c, t.get("crop"), t.get("stage"), t.get("watered")))
            elif k == "WEED":
                p1_weeds += 1
print(f"Structures ({len(p1_structures)}):", p1_structures)
print(f"Field Crops ({len(p1_crops)}):", p1_crops)
print(f"Weeds: {p1_weeds}")
