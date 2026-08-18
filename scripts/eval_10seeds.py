import os, sys
sys.path.insert(0, os.path.abspath("."))
from kaggle_environments import make
from src.agents.hrl_12worker_dispatcher import HRL12WorkerHungarianDispatcher

def evaluate_seeds(start_seed=40, count=10):
    dispatcher = HRL12WorkerHungarianDispatcher()
    def agent(obs, config=None):
        return dispatcher(obs, config)
        
    p0_scores = []
    p1_scores = []
    
    print(f"=== Testing HRL 12-Worker Dispatcher on {count} seeds ({start_seed}..{start_seed+count-1}) ===")
    for s in range(start_seed, start_seed + count):
        env0 = make("kaggriculture", configuration={"episodeSteps": 720, "seed": s})
        env0.run([agent, "starter"])
        r0 = env0.steps[-1][0].reward
        p0_scores.append(r0)
        
        env1 = make("kaggriculture", configuration={"episodeSteps": 720, "seed": s})
        env1.run(["starter", agent])
        r1 = env1.steps[-1][1].reward
        p1_scores.append(r1)
        
        print(f"Seed {s:02d}: P0=${r0:>8,.1f} | P1=${r1:>8,.1f}")
        
    print(f"\n=== Overall Results ===")
    print(f"P0 Mean: ${sum(p0_scores)/len(p0_scores):,.1f} (Min: ${min(p0_scores):,.1f}, Max: ${max(p0_scores):,.1f})")
    print(f"P1 Mean: ${sum(p1_scores)/len(p1_scores):,.1f} (Min: ${min(p1_scores):,.1f}, Max: ${max(p1_scores):,.1f})")
    print(f"Overall 20-Game Mean: ${(sum(p0_scores)+sum(p1_scores))/(2*len(p0_scores)):,.1f}")

if __name__ == "__main__":
    evaluate_seeds(40, 10)
