import json

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

print("=== MELON MARKET DYNAMICS & PRICE COLLAPSE ===")
for t in range(0, len(steps), 24):
    day = t // 24
    obs = steps[t][0]["observation"]
    melon_price = obs.get("market", {}).get("prices", {}).get("MELON", 0)
    melon_inv = obs.get("market", {}).get("inventory", {}).get("MELON", 0)
    
    # Check sales on this day
    p0_melon_sold = 0
    p1_melon_sold = 0
    for step_in_day in steps[t:t+24]:
        # p0
        for o in step_in_day[0].get("action", {}).get("market", []):
            if isinstance(o, list) and len(o) > 1 and o[0] == "SELL" and o[1] == "MELON":
                p0_melon_sold += (o[2] if len(o) > 2 else 1)
        # p1
        for o in step_in_day[1].get("action", {}).get("market", []):
            if isinstance(o, list) and len(o) > 1 and o[0] == "SELL" and o[1] == "MELON":
                p1_melon_sold += (o[2] if len(o) > 2 else 1)
                
    print(f"Day {day:02d}: Price=${melon_price:<3} MktInv={melon_inv:<6} | P0 Sold={p0_melon_sold:<2} | P1 Sold={p1_melon_sold:<2}")
