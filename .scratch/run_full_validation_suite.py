import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from kaggle_environments import make
import submission

def run_evaluation_suite():
    results = {"starter": [], "random": [], "pass": []}
    
    # 1. vs STARTER (10 matches with varied seeds)
    print(">>> Running 10-match evaluation vs STARTER...")
    for m_idx in range(1, 11):
        seed = 42000 + m_idx
        t0 = time.time()
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        env.run([submission.agent, "starter"])
        dur = time.time() - t0
        r0 = env.steps[-1][0].reward
        r1 = env.steps[-1][1].reward
        errs = [s for s in env.steps if s[0].status == "ERROR"]
        results["starter"].append({
            "match": m_idx, "seed": seed, "r0": r0, "r1": r1, "margin": r0 - r1,
            "winner": "submission" if r0 > r1 else "starter", "errors": len(errs),
            "dur": dur, "latency_ms": dur * 1000.0 / len(env.steps)
        })
        print(f"  Match #{m_idx:02d}: submission ${r0:,.1f} vs starter ${r1:,.1f} | Winner: {'submission' if r0 > r1 else 'starter'} | Latency: {dur*1000.0/len(env.steps):.2f}ms/turn")

    # 2. vs RANDOM (5 matches)
    print("\n>>> Running 5-match evaluation vs RANDOM...")
    for m_idx in range(1, 6):
        seed = 52000 + m_idx
        t0 = time.time()
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        env.run([submission.agent, "random"])
        dur = time.time() - t0
        r0 = env.steps[-1][0].reward
        r1 = env.steps[-1][1].reward
        errs = [s for s in env.steps if s[0].status == "ERROR"]
        results["random"].append({
            "match": m_idx, "seed": seed, "r0": r0, "r1": r1, "margin": r0 - r1,
            "winner": "submission" if r0 > r1 else "random", "errors": len(errs),
            "dur": dur, "latency_ms": dur * 1000.0 / len(env.steps)
        })
        print(f"  Match #{m_idx:02d}: submission ${r0:,.1f} vs random ${r1:,.1f} | Winner: {'submission' if r0 > r1 else 'random'} | Latency: {dur*1000.0/len(env.steps):.2f}ms/turn")

    # 3. vs PASS (5 matches)
    print("\n>>> Running 5-match evaluation vs PASS...")
    for m_idx in range(1, 6):
        seed = 62000 + m_idx
        t0 = time.time()
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
        env.run([submission.agent, "pass"])
        dur = time.time() - t0
        r0 = env.steps[-1][0].reward
        r1 = env.steps[-1][1].reward
        errs = [s for s in env.steps if s[0].status == "ERROR"]
        results["pass"].append({
            "match": m_idx, "seed": seed, "r0": r0, "r1": r1, "margin": r0 - r1,
            "winner": "submission" if r0 > r1 else "pass", "errors": len(errs),
            "dur": dur, "latency_ms": dur * 1000.0 / len(env.steps)
        })
        print(f"  Match #{m_idx:02d}: submission ${r0:,.1f} vs pass ${r1:,.1f} | Winner: {'submission' if r0 > r1 else 'pass'} | Latency: {dur*1000.0/len(env.steps):.2f}ms/turn")

    with open(".scratch/full_validation_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n[OK] Validation data saved to .scratch/full_validation_data.json")

if __name__ == "__main__":
    run_evaluation_suite()
