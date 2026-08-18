import os, sys
sys.path.insert(0, os.path.abspath("."))
from kaggle_environments import make
from src.agents.hrl_12worker_dispatcher import agent

def analyze_seed(seed):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    env.run([agent, "starter"])
    print(f"=== SEED {seed} ANALYSIS (Final: ${env.steps[-1][0].reward}) ===")
    
    for d in range(30):
        s_idx = d * 24
        obs = env.steps[s_idx][0].observation
        farm = obs['farms'][0]
        shed = obs.get('private', {}).get('shed', {})
        tiles = farm.get('tiles', [])
        animals = 0
        pastures = 0
        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                if isinstance(t, dict):
                    if t.get('kind') == 'PASTURE':
                        pastures += 1
                        if t.get('animal'):
                            animals += 1
        prices = obs.get('market', {}).get('prices', {})
        fert_price = prices.get('FERTILIZER', 0)
        wool_price = prices.get('WOOL', 0)
        milk_price = prices.get('MILK', 0)
        print(f"Day {d:02d}: Cash=${farm['money']:>7.0f}, Pastures={pastures:>2}, Animals={animals:>2}, Shed={shed.get('COW',0)}C/{shed.get('SHEEP',0)}S/{shed.get('WHEAT',0)}W/{shed.get('FERTILIZER',0)}F, Prices: Fert={fert_price}, Wool={wool_price}, Milk={milk_price}")

print("Analyzing Seed 43 (Low)")
analyze_seed(43)
print("\nAnalyzing Seed 44 (High)")
analyze_seed(44)
