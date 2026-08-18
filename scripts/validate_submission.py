"""
Comprehensive Validation Script for Standalone submission.py.
Executes full 720-turn matches against:
1. 'starter' as Player 0 (debug=True)
2. 'starter' as Player 1 (debug=True)
3. 'random' as Player 0 and Player 1 (debug=True)
4. 'pass' as Player 0 and Player 1 (debug=True)

Measures:
- Turn-by-turn latency (mean, min, max, p95, p99)
- Action legality & rule violations (0 tolerance)
- Simulation errors & timeouts
- Financial breakdown & final scores
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import json
import numpy as np
from collections import defaultdict

# Direct import of standalone submission.py
import submission

from kaggle_environments import make

def run_match_validation(p0_agent, p1_agent, match_name, seed=42):
    print(f"\n=======================================================")
    print(f"RUNNING MATCH: {match_name} (Seed: {seed})")
    print(f"=======================================================")
    
    latencies = []
    
    # Wrap agent to accurately measure per-turn latency
    def instrumented_agent(obs, config=None):
        t0 = time.perf_counter()
        action = submission.agent(obs, config)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
        return action
        
    p0 = instrumented_agent if p0_agent == "submission" else p0_agent
    p1 = instrumented_agent if p1_agent == "submission" else p1_agent
    
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    
    start_time = time.time()
    env.run([p0, p1])
    total_time = time.time() - start_time
    
    steps = env.steps
    final_step = steps[-1]
    
    r0 = final_step[0].reward
    r1 = final_step[1].reward
    s0_status = final_step[0].status
    s1_status = final_step[1].status
    
    # Check for any errors or illegal moves in env log
    illegal_moves = 0
    timeouts = 0
    for step_idx, step_data in enumerate(steps):
        for p_idx in [0, 1]:
            st = step_data[p_idx].status
            if st == "ERROR":
                illegal_moves += 1
            elif st == "TIMEOUT":
                timeouts += 1

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr)) if len(lat_arr) > 0 else 0.0
    max_lat = float(np.max(lat_arr)) if len(lat_arr) > 0 else 0.0
    min_lat = float(np.min(lat_arr)) if len(lat_arr) > 0 else 0.0
    p95_lat = float(np.percentile(lat_arr, 95)) if len(lat_arr) > 0 else 0.0
    p99_lat = float(np.percentile(lat_arr, 99)) if len(lat_arr) > 0 else 0.0

    print(f"Match Result: P0 Reward=${r0:,.1f} | P1 Reward=${r1:,.1f}")
    print(f"Agent Status: P0={s0_status}, P1={s1_status}")
    print(f"Illegal Moves: {illegal_moves}, Timeouts: {timeouts}")
    print(f"Latency (ms): Mean={mean_lat:.3f}ms, Min={min_lat:.3f}ms, Max={max_lat:.3f}ms, P95={p95_lat:.3f}ms, P99={p99_lat:.3f}ms")
    print(f"Total Match Duration: {total_time:.2f}s across {len(steps)} steps")
    
    return {
        "match_name": match_name,
        "seed": seed,
        "p0_reward": r0,
        "p1_reward": r1,
        "p0_status": s0_status,
        "p1_status": s1_status,
        "illegal_moves": illegal_moves,
        "timeouts": timeouts,
        "mean_latency_ms": mean_lat,
        "max_latency_ms": max_lat,
        "min_latency_ms": min_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "duration_s": total_time,
    }

def main():
    results = []
    
    # 1. vs 'starter' as P0 (debug=True)
    res_p0_starter = run_match_validation("submission", "starter", "HRL Agent (P0) vs Starter (P1)", seed=44)
    results.append(res_p0_starter)
    
    # 2. vs 'starter' as P1 (debug=True)
    res_p1_starter = run_match_validation("starter", "submission", "Starter (P0) vs HRL Agent (P1)", seed=44)
    results.append(res_p1_starter)

    # 3. vs 'random' as P0 (debug=True)
    res_p0_random = run_match_validation("submission", "random", "HRL Agent (P0) vs Random (P1)", seed=44)
    results.append(res_p0_random)
    
    # 4. vs 'random' as P1 (debug=True)
    res_p1_random = run_match_validation("random", "submission", "Random (P0) vs HRL Agent (P1)", seed=44)
    results.append(res_p1_random)

    # 5. vs 'pass' as P0 (debug=True)
    res_p0_pass = run_match_validation("submission", "pass", "HRL Agent (P0) vs Pass (P1)", seed=44)
    results.append(res_p0_pass)

    # 6. vs 'pass' as P1 (debug=True)
    res_p1_pass = run_match_validation("pass", "submission", "Pass (P0) vs HRL Agent (P1)", seed=44)
    results.append(res_p1_pass)

    # Save summary JSON for report generation
    os.makedirs(".scratch", exist_ok=True)
    with open(".scratch/validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nAll validation matches completed successfully. JSON saved to .scratch/validation_results.json")

if __name__ == "__main__":
    main()
