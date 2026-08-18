"""
Multi-Match Tournament and Forensic Gameplay Battle Test for 30 AlphaGoat Mega League Personalities.

This script:
1. Validates all 30 Mega League personalities across full 720-turn (30-day) matches.
2. Tests head-to-head battles across archetypes:
   - Crop Specialists vs Livestock Masters
   - Town Shop Arbitrage Snipers vs Market Price Crashers
   - Expansion / Labor Archetypes vs Lean Operators
   - Game-Theoretic Adversaries vs Baselines
3. Captures detailed forensic telemetry:
   - Final Cash & Win Margins
   - Chore Throughput (Plant, Water, Harvest, Dig, Animal Care)
   - Labor Deployment (Farmhands hired, wages paid)
   - Liquidation Efficiency (Remaining shed inventory at turn 719)
   - Market Price Impact (Price collapse / dynamics per commodity)
   - Zero-exception & legality validation
4. Updates and reports Prioritized Fictitious Self-Play (PFSP) Elo ratings and leaderboard.
"""

import os
import sys
import time
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from kaggle_environments import make
from src.training.mega_league import (
    MEGA_LEAGUE_PERSONALITIES,
    AlphaGoatMegaLeague,
    CROPS,
    PRODUCTS,
    BASE_PRICES,
)


def extract_match_forensics(env) -> Dict[str, Any]:
    """Extract turn-by-turn forensic metrics from a completed environment run."""
    steps = env.steps
    total_steps = len(steps)

    # Player 0 & Player 1 telemetry
    stats = [
        {
            "cash_history": [],
            "chores": {
                "PLANT": defaultdict(int),
                "WATER": 0,
                "HARVEST": 0,
                "DIG": 0,
                "FERTILIZE": 0,
                "PET": 0,
                "FEED": 0,
                "MILK": 0,
                "SHEAR": 0,
                "COLLECT_EGG": 0,
                "PICKUP": defaultdict(int),
                "MOVE": 0,
                "PASS": 0,
            },
            "market_activity": {
                "BUY_SEED": defaultdict(int),
                "BUY_ANIMAL": defaultdict(int),
                "BUY_LAND": 0,
                "BUY_OTHER": defaultdict(int),
                "SELL": defaultdict(int),
                "HIRE": 0,
            },
            "quadrants_unlocked": ["NW"],
            "final_cash": 0.0,
            "final_shed": {},
            "errors": [],
            "status": "OK",
        }
        for _ in range(2)
    ]

    market_price_history = {c: [] for c in PRODUCTS}

    for t, step in enumerate(steps):
        s0, s1 = step[0], step[1]
        obs0, obs1 = s0.get("observation", {}), s1.get("observation", {})

        # Record market prices
        mkt = obs0.get("market", {})
        cur_prices = mkt.get("prices", {})
        for prod in PRODUCTS:
            if prod in cur_prices:
                market_price_history[prod].append(cur_prices[prod])

        # Track status/errors
        for p_idx, s in enumerate([s0, s1]):
            if s.get("status") in ["ERROR", "INVALID"]:
                stats[p_idx]["errors"].append(f"Step {t}: status={s.get('status')}")
                stats[p_idx]["status"] = s.get("status")

            # Cash history
            obs = obs0 if p_idx == 0 else obs1
            farms = obs.get("farms", [{}, {}])
            if p_idx < len(farms):
                my_farm = farms[p_idx]
                stats[p_idx]["cash_history"].append(my_farm.get("money", 0.0))
                stats[p_idx]["quadrants_unlocked"] = my_farm.get("unlocked_quadrants", ["NW"])

            # Action telemetry
            act = s.get("action", {})
            if isinstance(act, dict):
                # Farmer action
                f_act = act.get("farmer", [])
                all_unit_acts = [f_act] + act.get("hands", [])

                for u_act in all_unit_acts:
                    if not u_act or not isinstance(u_act, list):
                        continue
                    cmd = u_act[0]
                    if cmd == "PLANT" and len(u_act) > 1:
                        stats[p_idx]["chores"]["PLANT"][u_act[1]] += 1
                    elif cmd in ["WATER", "HARVEST", "DIG", "FERTILIZE", "PET", "FEED", "MILK", "SHEAR", "COLLECT_EGG"]:
                        stats[p_idx]["chores"][cmd] += 1
                    elif cmd == "PICKUP" and len(u_act) > 1:
                        stats[p_idx]["chores"]["PICKUP"][u_act[1]] += (u_act[2] if len(u_act) > 2 else 1)
                    elif cmd in ["NORTH", "SOUTH", "EAST", "WEST"]:
                        stats[p_idx]["chores"]["MOVE"] += 1
                    elif cmd == "PASS":
                        stats[p_idx]["chores"]["PASS"] += 1

                # Market orders
                for m_ord in act.get("market", []):
                    if not m_ord or not isinstance(m_ord, list):
                        continue
                    m_cmd = m_ord[0]
                    if m_cmd == "BUY_SEED" and len(m_ord) > 1:
                        stats[p_idx]["market_activity"]["BUY_SEED"][m_ord[1]] += (m_ord[2] if len(m_ord) > 2 else 1)
                    elif m_cmd == "BUY_ANIMAL" and len(m_ord) > 1:
                        stats[p_idx]["market_activity"]["BUY_ANIMAL"][m_ord[1]] += (m_ord[2] if len(m_ord) > 2 else 1)
                    elif m_cmd == "BUY_LAND":
                        stats[p_idx]["market_activity"]["BUY_LAND"] += 1
                    elif m_cmd == "BUY" and len(m_ord) > 1:
                        stats[p_idx]["market_activity"]["BUY_OTHER"][m_ord[1]] += (m_ord[2] if len(m_ord) > 2 else 1)
                    elif m_cmd == "SELL" and len(m_ord) > 1:
                        stats[p_idx]["market_activity"]["SELL"][m_ord[1]] += (m_ord[2] if len(m_ord) > 2 else 1)
                    elif m_cmd == "HIRE":
                        stats[p_idx]["market_activity"]["HIRE"] += (m_ord[1] if len(m_ord) > 1 else 1)

    # Final state
    last_step = steps[-1]
    stats[0]["final_cash"] = last_step[0].get("reward", 0.0)
    stats[1]["final_cash"] = last_step[1].get("reward", 0.0)

    # Final shed inventory
    obs_last = last_step[0].get("observation", {})
    priv0 = obs_last.get("private", {})
    stats[0]["final_shed"] = priv0.get("shed", {})

    obs_last1 = last_step[1].get("observation", {})
    priv1 = obs_last1.get("private", {})
    stats[1]["final_shed"] = priv1.get("shed", {})

    # Final market prices
    final_prices = {prod: (market_price_history[prod][-1] if market_price_history[prod] else BASE_PRICES.get(prod, 0)) for prod in PRODUCTS}

    return {
        "steps_count": total_steps,
        "p0": stats[0],
        "p1": stats[1],
        "final_prices": final_prices,
    }


def run_match(
    name0: str,
    agent0: Any,
    name1: str,
    agent1: Any,
    seed: int = 42,
    steps: int = 720,
) -> Tuple[Dict[str, Any], float, float]:
    """Runs a single head-to-head match and returns full telemetry."""
    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run([agent0, agent1])
    dur = time.time() - t0

    forensics = extract_match_forensics(env)
    p0_cash = forensics["p0"]["final_cash"]
    p1_cash = forensics["p1"]["final_cash"]

    return forensics, p0_cash, p1_cash


def run_baseline_validation_suite(seeds: List[int] = [42]) -> Dict[str, Any]:
    """
    Phase 1: Validate all 30 personalities against KaggleStarter deterministic baseline.
    Verifies legality, exception-free operation, and benchmarks individual performance.
    """
    print("=" * 80)
    print("PHASE 1: 30 MEGA LEAGUE PERSONALITIES vs. KAGGLE STARTER BASELINE")
    print("=" * 80)

    results = {}
    for idx, (name, agent) in enumerate(MEGA_LEAGUE_PERSONALITIES.items(), 1):
        total_p0_cash = 0.0
        total_starter_cash = 0.0
        total_chores = 0
        total_harvests = 0
        total_plantings = 0
        total_hires = 0
        has_error = False

        for seed in seeds:
            forensics, p0_cash, p1_cash = run_match(
                name, agent, "starter", "starter", seed=seed, steps=720
            )
            total_p0_cash += p0_cash
            total_starter_cash += p1_cash

            p0_chores = forensics["p0"]["chores"]
            total_plantings += sum(p0_chores["PLANT"].values())
            total_harvests += p0_chores["HARVEST"]
            total_chores += (
                sum(p0_chores["PLANT"].values())
                + p0_chores["WATER"]
                + p0_chores["HARVEST"]
                + p0_chores["DIG"]
                + p0_chores["FERTILIZE"]
                + p0_chores["PET"]
                + p0_chores["FEED"]
                + p0_chores["MILK"]
                + p0_chores["SHEAR"]
                + p0_chores["COLLECT_EGG"]
            )
            total_hires += forensics["p0"]["market_activity"]["HIRE"]

            if forensics["p0"]["errors"] or forensics["p0"]["status"] != "OK":
                has_error = True

        n = len(seeds)
        mean_p0 = total_p0_cash / n
        mean_starter = total_starter_cash / n
        margin = mean_p0 - mean_starter

        results[name] = {
            "mean_cash": mean_p0,
            "starter_cash": mean_starter,
            "margin": margin,
            "mean_chores": total_chores / n,
            "mean_plantings": total_plantings / n,
            "mean_harvests": total_harvests / n,
            "mean_hires": total_hires / n,
            "has_error": has_error,
        }

        status_flag = "PASS [NO ERRORS]" if not has_error else "FAIL [ERRORS FOUND]"
        print(
            f"[{idx:02d}/30] {name:<26} | Cash: ${mean_p0:>8,.1f} | Starter: ${mean_starter:>7,.1f} | "
            f"Margin: ${margin:>+8,.1f} | Chores: {total_chores/n:>4.0f} | {status_flag}"
        )

    return results


def run_cross_archetype_tournament(league: AlphaGoatMegaLeague, seeds: List[int] = [42, 100]) -> List[Dict[str, Any]]:
    """
    Phase 2: Head-to-Head Cross-Archetype Tournament Matches.
    Tests specific strategic rivalries and mechanisms.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: CROSS-ARCHETYPE HEAD-TO-HEAD SHOWDOWNS")
    print("=" * 80)

    marquee_matchups = [
        # 1. Crop Specialist Rivalries
        ("MelonRusher", "CarrotSprinter", "Melon 12-day capital rusher vs Rapid 3-day Carrot cycling"),
        ("MelonRusher", "TomatoMonopolist", "High-margin Melon capex vs Multi-harvest Tomato annuity"),
        ("StrawberryAristocrat", "WheatIndustrialist", "Luxury $120 crop vs Bulk low-margin volume Wheat"),
        ("PortfolioHedger", "CarrotSprinter", "Risk-parity crop basket vs Pure single-crop carrot cycling"),

        # 2. Livestock & Husbandry vs Crop Engines
        ("DairyBaron", "MelonRusher", "Daily $160 milk cow rusher vs 12-day Melon engine"),
        ("GooseEggSwarm", "CarrotSprinter", "Daily $50 Goose egg swarm vs 3-day Carrot sprinter"),
        ("WoolSpecialist", "WheatIndustrialist", "High-value $200 Wool sheep pasture vs Bulk Wheat"),
        ("OrganicFertilizerTycoon", "DairyBaron", "Integrated Fertilizer livestock vs Pure Dairy rusher"),

        # 3. Market Exploitation & Town Arbitrage
        ("MarketPriceCrasher", "MelonRusher", "Early commodity price dumper vs Capital-intensive Melon rusher"),
        ("CommoditySpeculator", "MarketPriceCrasher", "Price-sensitive hoarder/swing trader vs Aggressive dumper"),
        ("PizzaShopSniper", "BakeryMonopolist", "Pizza 3x town multiplier sniper vs Bakery wheat/egg monopolist"),
        ("SmoothieExploiter", "DairyBaron", "Smoothie strawberry/milk sniper vs Dedicated Dairy baron"),

        # 4. Expansion & Labor Dynamics
        ("FourQuadrantOverlord", "NWMinimalist", "Aggressive 4-quadrant 100-tile expansion vs $0 land spend"),
        ("FiveWorkerSwarm", "LeanSoloOperator", "Max 5 daily farmhands swarm vs Zero-labor overhead solo farmer"),
        ("SerpentineChorer", "CarrotSprinter", "Boustrophedon spatial pathfinder vs Standard distance priority"),

        # 5. Game-Theoretic & Baselines
        ("AntiCompetitorShadow", "MelonRusher", "Dynamic opponent crop mirroring/counter vs Melon rusher"),
        ("AntiCompetitorShadow", "CarrotSprinter", "Opponent mirror shadow vs Carrot sprinter"),
        ("GreedySnowballer", "SafePreserver", "100% reinvestment geometric snowballer vs $1,000 cash buffer"),
        ("StochasticPerturbation", "PortfolioHedger", "Noisy perturbed hybrid expert vs Balanced portfolio"),
        ("PastMuZeroChampion", "MelonRusher", "Neural SSL/MCTS hybrid executor vs Melon rusher"),
    ]

    h2h_results = []

    for name0, name1, narrative in marquee_matchups:
        agent0 = MEGA_LEAGUE_PERSONALITIES[name0]
        agent1 = MEGA_LEAGUE_PERSONALITIES[name1]

        for seed in seeds:
            # Match 1: name0 as P0, name1 as P1
            forensics0, p0_cash, p1_cash = run_match(name0, agent0, name1, agent1, seed=seed)
            league.update_match_result(name0, name1, p0_cash, p1_cash)

            # Match 2: Inverted positions (name1 as P0, name0 as P1) to eliminate first-player bias
            forensics1, inv_p1_cash, inv_p0_cash = run_match(name1, agent1, name0, agent0, seed=seed)
            league.update_match_result(name1, name0, inv_p1_cash, inv_p0_cash)

            avg_cash0 = (p0_cash + inv_p0_cash) / 2.0
            avg_cash1 = (p1_cash + inv_p1_cash) / 2.0
            margin = avg_cash0 - avg_cash1
            winner = name0 if margin > 0 else (name1 if margin < 0 else "TIE")

            match_record = {
                "matchup": f"{name0} vs {name1}",
                "narrative": narrative,
                "seed": seed,
                "agent0": name0,
                "agent1": name1,
                "p0_cash": p0_cash,
                "p1_cash": p1_cash,
                "inv_p0_cash": inv_p0_cash,
                "inv_p1_cash": inv_p1_cash,
                "avg_cash0": avg_cash0,
                "avg_cash1": avg_cash1,
                "margin": margin,
                "winner": winner,
                "forensics_p0": forensics0["p0"],
                "forensics_p1": forensics0["p1"],
                "final_prices": forensics0["final_prices"],
            }
            h2h_results.append(match_record)

            print(
                f"[VS] {name0:<22} vs {name1:<22} (Seed {seed:03d}) -> "
                f"${avg_cash0:>7,.0f} vs ${avg_cash1:>7,.0f} | Winner: {winner} (${abs(margin):>+6,.0f})"
            )

    return h2h_results


def run_full_forensic_tournament():
    """Main execution workflow running baseline validation and marquee tournament."""
    start_time = time.time()

    league = AlphaGoatMegaLeague(main_name="PastMuZeroChampion")

    # Phase 1: Baseline validation
    baseline_stats = run_baseline_validation_suite(seeds=[42])

    # Phase 2: Cross-archetype marquee battles
    h2h_matches = run_cross_archetype_tournament(league, seeds=[42, 100])

    total_time = time.time() - start_time

    # Compile League Leaderboard
    league_summary = league.get_league_summary()
    leaderboard = league_summary["leaderboard"]

    print("\n" + "=" * 80)
    print("PHASE 3: ALPHAGOAT MEGA LEAGUE ELO LEADERBOARD & SUMMARY")
    print("=" * 80)
    print(f"{'Rank':<5} {'Personality':<26} {'Elo':<8} {'Matches':<8} {'Record (W-L-T)':<16} {'Win Rate':<10} {'Mean Cash':<12}")
    print("-" * 88)

    for rank, entry in enumerate(leaderboard, 1):
        rec_str = f"{entry['wins']}-{entry['losses']}-{entry['ties']}"
        print(
            f"{rank:<5} {entry['name']:<26} {entry['elo']:<8.1f} {entry['matches']:<8} "
            f"{rec_str:<16} {entry['win_rate']:>6.1f}%   ${entry['mean_cash']:>9,.1f}"
        )

    # Save detailed JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_seconds": round(total_time, 2),
        "total_personalities": len(MEGA_LEAGUE_PERSONALITIES),
        "baseline_validation": baseline_stats,
        "marquee_matches": [
            {
                "matchup": m["matchup"],
                "narrative": m["narrative"],
                "seed": m["seed"],
                "winner": m["winner"],
                "avg_cash0": round(m["avg_cash0"], 1),
                "avg_cash1": round(m["avg_cash1"], 1),
                "margin": round(m["margin"], 1),
                "final_prices": m["final_prices"],
            }
            for m in h2h_matches
        ],
        "leaderboard": leaderboard,
    }

    report_path = os.path.join(REPO_ROOT, "mega_league_tournament_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Full forensic tournament completed in {total_time:.1f}s. Report saved to {report_path}")
    return report


if __name__ == "__main__":
    run_full_forensic_tournament()
