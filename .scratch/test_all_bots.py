import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from src.agents.rl_champion_bot import agent as rl_agent
from src.agents.hybrid_expert import agent as hybrid_agent
from src.agents.balanced_farmer import agent as crop_agent
from src.agents.arbitrage_bot import agent as arb_agent
from src.agents.livestock_bot import agent as live_agent
import submission

agents = {
    "RL_Champion": rl_agent,
    "Hybrid_Expert": hybrid_agent,
    "Crop_Farmer": crop_agent,
    "Arbitrage": arb_agent,
    "Livestock": live_agent,
    "Current_Submission": submission.agent,
}

print(f"{'Agent':<20} | {'vs Starter':<15} | {'vs Random':<15} | {'vs Pass':<15} | {'Avg Latency':<12}")
print("-" * 85)

for name, ag in agents.items():
    res = {}
    total_time = 0
    total_steps = 0
    for opp in ["starter", "random", "pass"]:
        t0 = time.time()
        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
        env.run([ag, opp])
        dur = time.time() - t0
        r0 = env.steps[-1][0].reward
        r1 = env.steps[-1][1].reward
        res[opp] = f"${r0:,.0f} vs ${r1:,.0f}"
        total_time += dur
        total_steps += len(env.steps)
    
    avg_lat = (total_time * 1000) / total_steps
    print(f"{name:<20} | {res['starter']:<15} | {res['random']:<15} | {res['pass']:<15} | {avg_lat:.2f} ms/turn")
