from kaggle_environments import make
from src.agents.balanced_farmer import MultiTileCropAgent

bot = MultiTileCropAgent()
env = make("kaggriculture", configuration={"episodeSteps": 200}, debug=True)
env.reset()

for step in range(120):
    obs0 = env.state[0].observation
    act0 = bot(obs0)
    env.step([act0, {"farmer": ["PASS"], "hands": [], "market": []}])
    p0 = env.state[0]
    if step % 24 == 0 or act0["market"] and "SELL" in str(act0["market"]):
        print(f"Step {step:3d} (Day {obs0.get('day')}, Hr {obs0.get('hour')}): Money={p0.observation['farms'][0]['money']}, Shed={p0.observation['private']['shed']}, FarmerAct={act0['farmer']}, Market={act0['market']}")
