import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from src.agents.rl_champion_bot import agent as rl_agent

opponents = ["starter", "random", "pass"]
for opp in opponents:
    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([rl_agent, opp])
    dur = time.time() - t0
    step_count = len(env.steps)
    r0 = env.steps[-1][0].reward
    r1 = env.steps[-1][1].reward
    status0 = env.steps[-1][0].status
    status1 = env.steps[-1][1].status
    
    # Check for any errors or illegal moves in steps
    errors = [s for s in env.steps if s[0].status == "ERROR" or s[1].status == "ERROR"]
    
    print(f"Match vs {opp.upper()}:")
    print(f"  Steps: {step_count}")
    print(f"  RL Champion Bank: ${r0:,.2f} (Status: {status0})")
    print(f"  {opp} Bank: ${r1:,.2f} (Status: {status1})")
    print(f"  Errors/Illegal: {len(errors)}")
    print(f"  Total Duration: {dur:.2f}s | Avg Turn Latency: {dur*1000/step_count:.2f}ms/turn")
    print("-" * 50)
