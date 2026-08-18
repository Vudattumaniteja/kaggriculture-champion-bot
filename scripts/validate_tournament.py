"""
Tournament Validation Script for Frontier V2 Grandmaster Submission vs Kaggle Starter Baseline.
Evaluates 10 seeded matches, measuring Win Rate, Mean Cash, Margins, Trapped Deadweight, and Inference Latency.
"""

import os
import sys
import time
import numpy as np
from kaggle_environments import make

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from submission import agent as submission_agent
from src.training.frontier_v2_trainer import calculate_trapped_deadweight


class TimedAgent:
    def __init__(self, agent_fn):
        self.agent_fn = agent_fn
        self.times = []

    def __call__(self, obs, config=None):
        t0 = time.perf_counter()
        act = self.agent_fn(obs, config)
        self.times.append((time.perf_counter() - t0) * 1000.0)
        return act


def run_tournament(num_episodes: int = 10):
    print("=" * 80, flush=True)
    print(f" FRONTIER V2 GRANDMASTER TOURNAMENT VALIDATION ({num_episodes} EPISODES VS STARTER)", flush=True)
    print("=" * 80, flush=True)

    wins = 0
    ties = 0
    losses = 0
    bot_cash_list = []
    starter_cash_list = []
    deadweights = []
    latencies_ms = []

    for ep in range(1, num_episodes + 1):
        seed = 1000 + ep * 42
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)

        bot_is_p0 = (ep % 2 == 1)
        timed_bot = TimedAgent(submission_agent)
        agents = [timed_bot, "starter"] if bot_is_p0 else ["starter", timed_bot]

        env.run(agents)

        r0 = float(env.steps[-1][0]["reward"] if env.steps[-1][0]["reward"] is not None else 0.0)
        r1 = float(env.steps[-1][1]["reward"] if env.steps[-1][1]["reward"] is not None else 0.0)

        if bot_is_p0:
            bot_cash, starter_cash = r0, r1
            final_obs = env.steps[-1][0]["observation"]
            bot_idx = 0
        else:
            bot_cash, starter_cash = r1, r0
            final_obs = env.steps[-1][1]["observation"]
            bot_idx = 1

        dw = calculate_trapped_deadweight(final_obs, bot_idx)
        deadweights.append(dw)
        bot_cash_list.append(bot_cash)
        starter_cash_list.append(starter_cash)
        latencies_ms.extend(timed_bot.times)

        margin = bot_cash - starter_cash
        if bot_cash > starter_cash:
            wins += 1
            res = "WIN"
        elif bot_cash < starter_cash:
            losses += 1
            res = "LOSS"
        else:
            ties += 1
            res = "TIE"

        avg_lat = float(np.mean(timed_bot.times))
        p99_lat = float(np.percentile(timed_bot.times, 99))
        print(f"[Match {ep:02d}/{num_episodes:02d}] {res} (as P{bot_idx}) | Frontier: ${bot_cash:,.0f} vs Starter: ${starter_cash:,.0f} "
              f"(Margin: ${margin:+,.0f}) | DW: ${dw:,.0f} | Latency: {avg_lat:.2f}ms (p99: {p99_lat:.2f}ms)", flush=True)

    win_rate = (wins / num_episodes) * 100.0
    mean_bot = float(np.mean(bot_cash_list))
    mean_starter = float(np.mean(starter_cash_list))
    mean_margin = float(np.mean([b - s for b, s in zip(bot_cash_list, starter_cash_list)]))
    mean_dw = float(np.mean(deadweights))
    avg_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))

    print("\n" + "=" * 80, flush=True)
    print(" TOURNAMENT RESULTS SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f" Record: {wins}W - {losses}L - {ties}T (Win Rate: {win_rate:.1f}%)", flush=True)
    print(f" Mean Cash: ${mean_bot:,.0f} vs ${mean_starter:,.0f} (Mean Margin: ${mean_margin:+,.0f})", flush=True)
    print(f" Mean Trapped Deadweight: ${mean_dw:,.0f}", flush=True)
    print(f" Latency: Mean={avg_latency:.2f}ms | p95={p95_latency:.2f}ms | p99={p99_latency:.2f}ms (Target: <5.0ms)", flush=True)
    print("=" * 80, flush=True)

    return {
        "episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": win_rate,
        "mean_cash": mean_bot,
        "mean_starter_cash": mean_starter,
        "mean_margin": mean_margin,
        "mean_deadweight": mean_dw,
        "latency_mean_ms": round(avg_latency, 2),
        "latency_p95_ms": round(p95_latency, 2),
        "latency_p99_ms": round(p99_latency, 2),
    }


if __name__ == "__main__":
    run_tournament(10)
