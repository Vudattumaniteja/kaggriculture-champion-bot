"""
10-Game Tournament Benchmarking Suite vs 'starter' baseline on submission.py.
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import time
import numpy as np
import kaggle_environments
import submission

def run_tournament(n_games=10):
    results = []
    start_time = time.time()

    print(f"Starting {n_games}-Game Benchmark Tournament vs 'starter' (Testing submission.py)...")
    print("-" * 65)
    print(f"{'Game':<6} | {'Bot Score':<12} | {'Starter Score':<15} | {'Margin':<12} | {'Result'}")
    print("-" * 65)

    for g in range(n_games):
        bot = submission.HybridMarketAgent()
        env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
        env.run([bot, "starter"])
        bot_reward = float(env.steps[-1][0].reward)
        opp_reward = float(env.steps[-1][1].reward)
        margin = bot_reward - opp_reward
        res_str = "WIN (100%)" if bot_reward > opp_reward else "LOSS"
        results.append((bot_reward, opp_reward))
        print(f"Game {g+1:02d} | ${bot_reward:<11,.1f} | ${opp_reward:<14,.1f} | +${margin:<10,.1f} | {res_str}")

    bot_scores = [r[0] for r in results]
    opp_scores = [r[1] for r in results]

    print("-" * 65)
    print(f"Tournament Summary ({n_games} Games in {time.time() - start_time:.1f}s):")
    print(f"  Bot Mean Cash    : ${np.mean(bot_scores):,.2f} +/- ${np.std(bot_scores):,.2f}")
    print(f"  Bot Min / Max Cash: ${np.min(bot_scores):,.2f} / ${np.max(bot_scores):,.2f}")
    print(f"  Starter Mean Cash: ${np.mean(opp_scores):,.2f}")
    print(f"  Win Rate         : {sum(1 for b, o in results if b > o) / n_games * 100:.1f}%")
    print("-" * 65)

    return results

if __name__ == "__main__":
    run_tournament(10)
