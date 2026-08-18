import os, sys
sys.path.insert(0, os.path.abspath("."))
from kaggle_environments import make
from scripts.test_enhanced_dispatcher import EnhancedHRL12WorkerDispatcher

def diagnose(seed):
    agent_inst = EnhancedHRL12WorkerDispatcher()
    def test_agent(obs, config=None):
        return agent_inst(obs, config)
        
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    env.run([test_agent, "starter"])
    print(f"=== SEED {seed} DIAGNOSIS (Final: ${env.steps[-1][0].reward}) ===")
    
    for d in range(30):
        s_idx = d * 24
        obs = env.steps[s_idx][0].observation
        farm = obs['farms'][0]
        shed = obs.get('private', {}).get('shed', {})
        tiles = farm.get('tiles', [])
        animals = sum(1 for r in tiles for t in r if isinstance(t, dict) and t.get('animal'))
        pastures = sum(1 for r in tiles for t in r if isinstance(t, dict) and t.get('kind') == 'PASTURE')
        print(f"Day {d:02d}: Cash=${farm['money']:>7.0f}, Pastures={pastures:>2}, Animals={animals:>2}, Shed={shed.get('COW',0)}C/{shed.get('SHEEP',0)}S/{shed.get('WHEAT',0)}W/{shed.get('FERTILIZER',0)}F, Hires={farm.get('hires_today')}")

diagnose(40)
