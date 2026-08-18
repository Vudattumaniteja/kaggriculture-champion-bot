import os
import sys

sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from submission import agent as my_agent

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1000}, debug=True)
env.run([my_agent, "starter"])
print("Final status:", env.state[0].status, env.state[1].status)
print("Final rewards:", env.state[0].reward, env.state[1].reward)

for step_idx in range(0, 720, 48):
    p0 = env.steps[step_idx][0]
    obs = p0["observation"]
    my_farm = obs["farms"][0]
    private = obs.get("private", {})
    action = p0.get("action", {})
    print(f"Step {step_idx:03d} (Day {obs.get('day')}, Hr {obs.get('hour')}): Cash=${my_farm.get('money')} | Unlocked={my_farm.get('unlocked_quadrants')} | Shed={private.get('shed')} | Action={action}")

p0_final = env.steps[-1][0]
obs_final = p0_final["observation"]
print(f"Final Step 719: Cash=${obs_final['farms'][0].get('money')} | Shed={obs_final.get('private', {}).get('shed')} | Seeds={obs_final.get('private', {}).get('seeds')}")
