import sys, json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open(".scratch/parsed_top10_replays.json", "r", encoding="utf-8") as f:
    episodes = json.load(f)

for ep in episodes:
    p0 = ep["players"]["0"]["name"]
    p1 = ep["players"]["1"]["name"]
    r0 = ep["players"]["0"]["final_reward"]
    r1 = ep["players"]["1"]["final_reward"]
    meta = ep.get("meta", {})
    t_rank = meta.get("team_rank", "N/A")
    t_name = meta.get("team_name", "N/A")
    print(f"Ep {ep['episode_id']} | Target Rank {t_rank:2} ({t_name:20}) | P0: '{p0}' (${r0:8,}) | P1: '{p1}' (${r1:8,})")
