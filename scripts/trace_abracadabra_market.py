import json

with open("replays/top_competitors/episode-93954943-replay.json") as f:
    data = json.load(f)

steps = data["steps"]

for step_idx in range(len(steps)):
    obs = steps[step_idx][0]["observation"]
    day = obs["day"]
    hour = obs["hour"]
    # get market orders submitted at this step
    act = steps[step_idx][0].get("action", {})
    if act and act.get("market"):
        m_orders = act["market"]
        # print non-trivial market orders
        print(f"Day {day:02d} H{hour:02d} | Money: ${obs['farms'][0]['money']:<7.1f} | Market Orders: {m_orders}")
