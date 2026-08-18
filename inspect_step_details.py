import json

REPLAY_PATH = r"C:\Users\Manit\Downloads\93943624.json"

with open(REPLAY_PATH, "r") as f:
    replay = json.load(f)

steps = replay["steps"]

print("=== STEP 0-10 DETAILS ===")
for i in range(10):
    print(f"Step {i} (Day {steps[i][0]['observation']['day']}, Hr {steps[i][0]['observation']['hour']}):")
    print("  P0 Act:", steps[i][0].get("action"))
    print("  P1 Act:", steps[i][1].get("action"))
    print("  P0 Shed:", steps[i][0].get("observation", {}).get("private", {}).get("shed"))
    print("  P0 Seeds:", steps[i][0].get("observation", {}).get("private", {}).get("seeds"))
    print("  P0 Farmer pos:", steps[i][0].get("observation", {}).get("farms", [{}])[0].get("farmer"))

print("\n=== STEP 64-75 (FIRST MOVEMENT DIVERGENCE) ===")
for i in range(63, 75):
    print(f"Step {i} (Day {steps[i][0]['observation']['day']}, Hr {steps[i][0]['observation']['hour']}):")
    print("  P0 Act:", steps[i][0].get("action"))
    print("  P1 Act:", steps[i][1].get("action"))
    print("  P0 Pos: farmer=", steps[i][0].get("observation", {}).get("farms", [{}])[0].get("farmer"), "hands=", steps[i][0].get("observation", {}).get("farms", [{}])[0].get("hands"))
    print("  P1 Pos: farmer=", steps[i][1].get("observation", {}).get("farms", [{}])[1].get("farmer"), "hands=", steps[i][1].get("observation", {}).get("farms", [{}])[1].get("hands"))

print("\n=== STEP 408-415 (MONEY DIVERGENCE 1) ===")
for i in range(408, 415):
    print(f"Step {i} (Day {steps[i][0]['observation']['day']}, Hr {steps[i][0]['observation']['hour']}):")
    print("  P0 Act:", steps[i][0].get("action"))
    print("  P1 Act:", steps[i][1].get("action"))
    print("  P0 Money:", steps[i][0].get("observation", {}).get("farms", [{}])[0].get("money"), "Seeds:", steps[i][0].get("observation", {}).get("private", {}).get("seeds"))
    print("  P1 Money:", steps[i][1].get("observation", {}).get("farms", [{}])[1].get("money"), "Seeds:", steps[i][1].get("observation", {}).get("private", {}).get("seeds"))

print("\n=== STEP 650-658 (WHAT IS HAPPENING ON DAY 27?) ===")
for i in range(650, 660):
    print(f"Step {i} (Day {steps[i][0]['observation']['day']}, Hr {steps[i][0]['observation']['hour']}):")
    print("  P0 Act:", steps[i][0].get("action"))
    print("  P1 Act:", steps[i][1].get("action"))
    print("  P0 Money:", steps[i][0].get("observation", {}).get("farms", [{}])[0].get("money"))
    print("  P1 Money:", steps[i][1].get("observation", {}).get("farms", [{}])[1].get("money"))
