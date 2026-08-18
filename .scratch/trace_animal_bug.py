import os
import sys

sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from submission import agent as my_agent

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1000}, debug=True)
env.run([my_agent, "starter"])

for step_idx in range(96, 160):
    p0 = env.steps[step_idx][0]
    obs = p0["observation"]
    my_farm = obs["farms"][0]
    private = obs.get("private", {})
    action = p0.get("action", {})
    tiles = my_farm["tiles"]
    coops = [(c, r, t) for r, row in enumerate(tiles) for c, t in enumerate(row) if isinstance(t, dict) and t.get("kind") == "COOP"]
    pastures = [(c, r, t) for r, row in enumerate(tiles) for c, t in enumerate(row) if isinstance(t, dict) and t.get("kind") == "PASTURE"]
    print(f"Step {step_idx:03d} (D{obs.get('day')},H{obs.get('hour')}): Cash={my_farm.get('money')} | Farmer={my_farm.get('farmer')} | Hands={my_farm.get('hands')} | Shed={private.get('shed')} | Coops={coops} | Pastures={pastures} | Action={action}")
