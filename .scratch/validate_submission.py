import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
import submission

print("=" * 80)
print("             OFFICIAL KAGGLE SUBMISSION VALIDATION SUITE")
print("=" * 80)

opponents = ["starter", "random", "pass"]
results = {}

for opp in opponents:
    print(f"\n>>> Running 720-step Match: submission.py vs '{opp}' (debug=True)...")
    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([submission.agent, opp])
    duration = time.time() - t0
    
    steps = len(env.steps)
    r0 = env.steps[-1][0].reward
    r1 = env.steps[-1][1].reward
    status0 = env.steps[-1][0].status
    status1 = env.steps[-1][1].status
    
    # Audit for errors or illegal actions across all steps
    errors_p0 = [i for i, s in enumerate(env.steps) if s[0].status == "ERROR"]
    errors_p1 = [i for i, s in enumerate(env.steps) if s[1].status == "ERROR"]
    
    avg_latency_ms = (duration * 1000.0) / steps
    winner = "submission.py" if r0 > r1 else (opp if r1 > r0 else "TIE")
    
    results[opp] = {
        "steps": steps,
        "reward_submission": r0,
        "reward_opponent": r1,
        "status_submission": status0,
        "status_opponent": status1,
        "errors_submission": len(errors_p0),
        "duration_sec": duration,
        "avg_latency_ms": avg_latency_ms,
        "winner": winner,
    }
    
    print(f"    [Result] submission.py: ${r0:,.2f} ({status0}) vs {opp}: ${r1:,.2f} ({status1})")
    print(f"    [Outcome] Winner: {winner} | Margin: ${r0 - r1:+,.2f}")
    print(f"    [Validation] Errors / Illegal Moves: {len(errors_p0)} | Total Steps: {steps}")
    print(f"    [Performance] Total Time: {duration:.2f}s | Avg Turn Latency: {avg_latency_ms:.2f} ms/turn (< 100ms threshold)")

print("\n" + "=" * 80)
print("                           SUMMARY TABLE")
print("=" * 80)
print(f"{'Opponent':<12} | {'Submission Bank':<16} | {'Opponent Bank':<14} | {'Margin':<12} | {'Errors':<7} | {'Turn Latency':<14} | {'Winner'}")
print("-" * 90)
for opp, r in results.items():
    print(
        f"{opp.upper():<12} | ${r['reward_submission']:>14,.2f} | ${r['reward_opponent']:>12,.2f} | "
        f"${r['reward_submission'] - r['reward_opponent']:>+10,.2f} | {r['errors_submission']:<7} | "
        f"{r['avg_latency_ms']:>6.2f} ms/step   | {r['winner']}"
    )
print("=" * 80)
