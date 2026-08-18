import os, sys
sys.path.insert(0, os.path.abspath("."))
import time
from kaggle_environments import make

# Let's import hrl_12worker_dispatcher
from src.agents.hrl_12worker_dispatcher import HRL12WorkerHungarianDispatcher

def evaluate_agent(dispatcher_cls, num_seeds=10):
    rewards_p0 = []
    rewards_p1 = []
    
    agent_inst = dispatcher_cls()
    def test_agent(obs, config=None):
        return agent_inst(obs, config)
    
    for s in range(40, 40 + num_seeds):
        # As P0
        env0 = make("kaggriculture", configuration={"episodeSteps": 720, "seed": s})
        env0.run([test_agent, "starter"])
        r0 = env0.steps[-1][0].reward
        rewards_p0.append(r0)
        
        # As P1
        env1 = make("kaggriculture", configuration={"episodeSteps": 720, "seed": s})
        env1.run(["starter", test_agent])
        r1 = env1.steps[-1][1].reward
        rewards_p1.append(r1)
        
        print(f"Seed {s:02d}: P0=${r0:>8,.1f} | P1=${r1:>8,.1f}")
        
    print(f"=== Summary ({num_seeds} seeds) ===")
    print(f"P0 Mean: ${sum(rewards_p0)/len(rewards_p0):,.1f} (Min: ${min(rewards_p0):,.1f}, Max: ${max(rewards_p0):,.1f})")
    print(f"P1 Mean: ${sum(rewards_p1)/len(rewards_p1):,.1f} (Min: ${min(rewards_p1):,.1f}, Max: ${max(rewards_p1):,.1f})")

if __name__ == "__main__":
    evaluate_agent(HRL12WorkerHungarianDispatcher, num_seeds=5)
