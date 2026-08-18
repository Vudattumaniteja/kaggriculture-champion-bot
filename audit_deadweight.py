import json

with open(r"C:\Users\Manit\Downloads\93943624.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]
s0_final = steps[-1][0]["observation"]
s1_final = steps[-1][1]["observation"]

farm0 = s0_final["farms"][0]
farm1 = s1_final["farms"][1]

shed0 = s0_final["private"]["shed"]
shed1 = s1_final["private"]["shed"]

seeds0 = s0_final["private"]["seeds"]
seeds1 = s1_final["private"]["seeds"]

costs = {
    "COOP": 200,
    "PASTURE": 300,
    "GOOSE": 300,
    "COW": 400,
    "SHEEP": 500,
    "WHEAT_SEED": 10,
    "CARROT_SEED": 20,
    "TOMATO_SEED": 50,
    "STRAWBERRY_SEED": 100,
    "MELON_SEED": 80,
    "LAND_NE": 1000,
}

print("=== FINAL REWARDS & CASH ===")
print(f"P0 Reward: {steps[-1][0]['reward']}, Cash: ${farm0['money']}")
print(f"P1 Reward: {steps[-1][1]['reward']}, Cash: ${farm1['money']}")

print("\n=== FINAL SHED INVENTORY ===")
print("P0 Shed:", {k: v for k, v in shed0.items() if v > 0})
print("P1 Shed:", {k: v for k, v in shed1.items() if v > 0})

print("\n=== FINAL SEEDS INVENTORY ===")
print("P0 Seeds:", {k: v for k, v in seeds0.items() if v > 0})
print("P1 Seeds:", {k: v for k, v in seeds1.items() if v > 0})

# Count tiles
def audit_tiles(tiles):
    structures = []
    crops = []
    weeds = 0
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if isinstance(t, dict):
                k = t.get("kind")
                if k in ["COOP", "PASTURE"]:
                    structures.append((k, t.get("animal"), (c, r)))
                elif k == "PLANT":
                    crops.append((t.get("crop"), t.get("yield_units", 0), (c, r)))
                elif k == "WEED":
                    weeds += 1
    return structures, crops, weeds

struct0, crops0, weeds0 = audit_tiles(farm0["tiles"])
struct1, crops1, weeds1 = audit_tiles(farm1["tiles"])

print("\n=== FINAL TILES AUDIT ===")
print(f"P0 Structures ({len(struct0)}):", struct0)
print(f"P1 Structures ({len(struct1)}):", struct1)
print(f"P0 Crops ({len(crops0)}):", crops0)
print(f"P1 Crops ({len(crops1)}):", crops1)
print(f"P0 Weeds: {weeds0}, P1 Weeds: {weeds1}")

# Calculate Trapped Deadweight
def calc_deadweight(farm, shed, seeds, struct, crops):
    val = 0
    breakdown = {}
    
    # Land
    quads = farm.get("unlocked_quadrants", [])
    if "NE" in quads:
        breakdown["LAND_NE"] = 1000
        val += 1000
    if "SW" in quads:
        breakdown["LAND_SW"] = 2000
        val += 2000
    if "SE" in quads:
        breakdown["LAND_SE"] = 4000
        val += 4000
        
    # Shed items
    for item, cnt in shed.items():
        if cnt > 0:
            c = costs.get(item, 0)
            breakdown[f"SHED_{item}"] = c * cnt
            val += c * cnt
            
    # Seeds
    for s, cnt in seeds.items():
        if cnt > 0:
            c = costs.get(f"{s}_SEED", 0)
            breakdown[f"SEED_{s}"] = c * cnt
            val += c * cnt
            
    # Structures & Animals on field
    for k, an, pos in struct:
        breakdown[f"STRUCT_{k}_{pos}"] = costs.get(k, 0)
        val += costs.get(k, 0)
        if an:
            breakdown[f"FIELD_ANIMAL_{an}_{pos}"] = costs.get(an, 0)
            val += costs.get(an, 0)
            
    for crop_name, y, pos in crops:
        c = costs.get(f"{crop_name}_SEED", 0)
        breakdown[f"CROP_{crop_name}_{pos}"] = c
        val += c
        
    return val, breakdown

dw0, b0 = calc_deadweight(farm0, shed0, seeds0, struct0, crops0)
dw1, b1 = calc_deadweight(farm1, shed1, seeds1, struct1, crops1)

print(f"\nP0 Total Trapped Deadweight: ${dw0}")
print("P0 Deadweight Breakdown:", b0)

print(f"\nP1 Total Trapped Deadweight: ${dw1}")
print("P1 Deadweight Breakdown:", b1)

print(f"\nNet Worth (Cash + Deadweight):")
print(f"P0 Total Assets: ${farm0['money'] + dw0}")
print(f"P1 Total Assets: ${farm1['money'] + dw1}")
