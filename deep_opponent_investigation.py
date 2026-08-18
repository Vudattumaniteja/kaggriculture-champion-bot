import json
import os
from collections import defaultdict, Counter

with open(r"C:\Users\Manit\Downloads\93946801.json", "r") as f:
    replay = json.load(f)

steps = replay["steps"]

# Analyze Player 1's turn-by-turn logic
p1_actions_by_turn = []
for t, step in enumerate(steps):
    s = step[1] # Player 1
    obs = s.get("observation", {})
    act = s.get("action", {})
    day = obs.get("day", t // 24)
    hour = obs.get("hour", t % 24)
    
    farmer = act.get("farmer", []) if isinstance(act, dict) else []
    hands = act.get("hands", []) if isinstance(act, dict) else []
    market = act.get("market", []) if isinstance(act, dict) else []
    
    my_farm = obs.get("farms", [])[1] if len(obs.get("farms", [])) > 1 else {}
    money = my_farm.get("money", 0.0)
    farmer_pos = my_farm.get("farmer", {}).get("pos") if "farmer" in my_farm else None
    hands_pos = [h.get("pos") for h in my_farm.get("hands", [])]
    
    p1_actions_by_turn.append({
        "turn": t, "day": day, "hour": hour, "money": money,
        "farmer_act": farmer, "hands_act": hands, "market": market,
        "farmer_pos": farmer_pos, "hands_pos": hands_pos
    })

print("First 30 turns of Player 1:")
for a in p1_actions_by_turn[:30]:
    print(f"Turn {a['turn']:03d} (D{a['day']:02d} H{a['hour']:02d}) Cash=${a['money']:<6.0f} | Farmer: {a['farmer_act']} @ {a['farmer_pos']} | Market: {a['market']} | Hands: {a['hands_act']}")
