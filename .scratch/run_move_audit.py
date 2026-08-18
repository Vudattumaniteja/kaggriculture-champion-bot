import sys
import os
import json
from collections import defaultdict
sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from src.models.mcts import SSLMCTSAgent
from src.agents.utils import parse_observation

MACRO_NAMES = {
    0: "FARM_CARROTS_INTENSIVE",
    1: "FARM_WHEAT_EXPANSION",
    2: "FARM_DIVERSIFIED",
    3: "BUY_LAND_EXPANSION",
    4: "HIRE_EXTRA_LABOR",
    5: "BUILD_GOOSE_COOP",
    6: "BUILD_PASTURE_LIVESTOCK",
    7: "HARVEST_AND_LIQUIDATE",
    8: "MARKET_ARBITRAGE_TRADE",
    9: "LIVESTOCK_CARE_FEED",
}

BASE_PRICES = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
    "GOOSE": 300,
    "COW": 400,
    "SHEEP": 500,
}

SEED_PRICES = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}

def analyze_match(model_path: str, ep: int, seed: int, is_swapped: bool):
    agent_inst = SSLMCTSAgent(
        model_weights_path=model_path,
        num_simulations=20,
        c_puct=1.5,
        temperature=0.0,
    )

    turn_by_turn = []
    daily_stats = defaultdict(lambda: {
        "start_money": 0,
        "end_money": 0,
        "seed_spend": 0,
        "seeds_bought": defaultdict(int),
        "animal_spend": 0,
        "animals_bought": defaultdict(int),
        "labor_spend": 0,
        "labor_hires": 0,
        "land_spend": 0,
        "land_buys": 0,
        "sales_revenue": 0,
        "items_sold": defaultdict(int),
        "structures_built": defaultdict(int),
        "macro_actions": defaultdict(int),
        "crops_planted": defaultdict(int),
        "crops_harvested": defaultdict(int),
        "animals_placed": defaultdict(int),
        "animals_fed": 0,
        "animals_cared": 0,
        "fertilizer_collected": 0,
        "fertilizer_applied": 0,
        "watered_count": 0,
    })

    def wrapped_agent(obs, config=None):
        state = parse_observation(obs)
        step = state["step"]
        day = state["day"]
        hour = state["hour"]
        money = state["money"]

        if hour == 0:
            daily_stats[day]["start_money"] = money

        macro_act, probs, val = agent_inst.search_best_action(obs)
        act = agent_inst(obs, config)

        daily_stats[day]["macro_actions"][MACRO_NAMES.get(macro_act, str(macro_act))] += 1

        # Track market actions
        for order in act.get("market", []):
            op = order[0]
            if op == "BUY_SEED":
                cname, qty = order[1], order[2]
                daily_stats[day]["seeds_bought"][cname] += qty
                daily_stats[day]["seed_spend"] += SEED_PRICES.get(cname, 20) * qty
            elif op == "BUY_ANIMAL":
                aname, qty = order[1], order[2]
                daily_stats[day]["animals_bought"][aname] += qty
                daily_stats[day]["animal_spend"] += BASE_PRICES.get(aname, 300) * qty
            elif op == "HIRE":
                daily_stats[day]["labor_hires"] += 1
                daily_stats[day]["labor_spend"] += 2
            elif op == "BUY_LAND":
                daily_stats[day]["land_buys"] += 1
                # cost is 1000 for NE, 2000 for SW, 4000 for SE
                unlocked = state["unlocked_quads"]
                cost = 1000 if "NE" not in unlocked else (2000 if "SW" not in unlocked else 4000)
                daily_stats[day]["land_spend"] += cost
            elif op == "SELL":
                item, qty = order[1], order[2]
                daily_stats[day]["items_sold"][item] += qty
                cur_price = state["market_prices"].get(item, BASE_PRICES.get(item, 25))
                daily_stats[day]["sales_revenue"] += cur_price * qty

        # Track farmer / hands operational chores
        all_actions = [act.get("farmer", ["PASS"])] + act.get("hands", [])
        for a in all_actions:
            if not a:
                continue
            op = a[0]
            if op == "BUILD_COOP":
                daily_stats[day]["structures_built"]["COOP"] += 1
            elif op == "BUILD_PASTURE":
                daily_stats[day]["structures_built"]["PASTURE"] += 1
            elif op == "PLANT":
                cname = a[1] if len(a) > 1 else "CARROT"
                daily_stats[day]["crops_planted"][cname] += 1
            elif op == "PLACE":
                aname = a[1] if len(a) > 1 else "ANIMAL"
                daily_stats[day]["animals_placed"][aname] += 1
            elif op == "HARVEST":
                # could be crop or animal
                daily_stats[day]["crops_harvested"]["HARVEST"] += 1
            elif op == "FEED":
                daily_stats[day]["animals_fed"] += 1
            elif op == "CARE":
                daily_stats[day]["animals_cared"] += 1
            elif op == "COLLECT_FERTILIZER":
                daily_stats[day]["fertilizer_collected"] += 1
            elif op == "FERTILIZE":
                daily_stats[day]["fertilizer_applied"] += 1
            elif op == "WATER":
                daily_stats[day]["watered_count"] += 1

        daily_stats[day]["end_money"] = money

        turn_by_turn.append({
            "step": step,
            "day": day,
            "hour": hour,
            "money": money,
            "macro_act": MACRO_NAMES.get(macro_act, str(macro_act)),
            "mcts_val": float(val),
            "farmer": act.get("farmer"),
            "market": act.get("market"),
        })

        return act

    config = {"episodeSteps": 720, "seed": seed}
    env = make("kaggriculture", configuration=config, debug=False)

    if not is_swapped:
        env.run([wrapped_agent, "starter"])
        ssl_idx = 0
        starter_idx = 1
    else:
        env.run(["starter", wrapped_agent])
        ssl_idx = 1
        starter_idx = 0

    final_ssl_reward = env.steps[-1][ssl_idx].reward
    final_starter_reward = env.steps[-1][starter_idx].reward

    # Parse final state
    final_obs = env.steps[-1][ssl_idx].observation
    final_state = parse_observation(final_obs)

    tiles = final_state["tiles"]
    shed = final_state["shed"]
    seeds = final_state["seeds"]
    unlocked_quads = final_state["unlocked_quads"]

    # Trapped assets audit
    trapped_animals = defaultdict(int)
    trapped_coops = 0
    trapped_pastures = 0
    trapped_crops = defaultdict(lambda: {"total": 0, "ripe": 0, "immature": 0, "yield_units": 0})
    empty_coops = 0
    empty_pastures = 0

    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "COOP":
                    trapped_coops += 1
                    anim = t.get("animal")
                    if anim:
                        trapped_animals[anim] += 1
                    else:
                        empty_coops += 1
                elif k == "PASTURE":
                    trapped_pastures += 1
                    anim = t.get("animal")
                    if anim:
                        trapped_animals[anim] += 1
                    else:
                        empty_pastures += 1
                elif k == "PLANT":
                    crop_name = t.get("crop", "UNKNOWN")
                    trapped_crops[crop_name]["total"] += 1
                    yield_u = t.get("yield_units", 0)
                    if yield_u > 0:
                        trapped_crops[crop_name]["ripe"] += 1
                        trapped_crops[crop_name]["yield_units"] += yield_u
                    else:
                        trapped_crops[crop_name]["immature"] += 1

    # Daily cash checkpoints
    cash_by_day = {}
    for d in range(30):
        # find hour 0 of day d
        for t in turn_by_turn:
            if t["day"] == d and t["hour"] == 0:
                cash_by_day[f"Day {d}"] = t["money"]
                break
    cash_by_day["Day 30 (Final)"] = final_ssl_reward

    # Aggregate by Phases:
    # Phase 1: Days 0-9 (Turns 0-239)
    # Phase 2: Days 10-19 (Turns 240-479)
    # Phase 3: Days 20-29 (Turns 480-719)
    def aggregate_phase(d_start, d_end):
        p = {
            "seed_spend": sum(daily_stats[d]["seed_spend"] for d in range(d_start, d_end)),
            "seeds_bought": dict(sum_subdicts([daily_stats[d]["seeds_bought"] for d in range(d_start, d_end)])),
            "animal_spend": sum(daily_stats[d]["animal_spend"] for d in range(d_start, d_end)),
            "animals_bought": dict(sum_subdicts([daily_stats[d]["animals_bought"] for d in range(d_start, d_end)])),
            "labor_spend": sum(daily_stats[d]["labor_spend"] for d in range(d_start, d_end)),
            "labor_hires": sum(daily_stats[d]["labor_hires"] for d in range(d_start, d_end)),
            "land_spend": sum(daily_stats[d]["land_spend"] for d in range(d_start, d_end)),
            "land_buys": sum(daily_stats[d]["land_buys"] for d in range(d_start, d_end)),
            "sales_revenue": sum(daily_stats[d]["sales_revenue"] for d in range(d_start, d_end)),
            "items_sold": dict(sum_subdicts([daily_stats[d]["items_sold"] for d in range(d_start, d_end)])),
            "structures_built": dict(sum_subdicts([daily_stats[d]["structures_built"] for d in range(d_start, d_end)])),
            "macro_actions": dict(sum_subdicts([daily_stats[d]["macro_actions"] for d in range(d_start, d_end)])),
            "crops_planted": dict(sum_subdicts([daily_stats[d]["crops_planted"] for d in range(d_start, d_end)])),
            "crops_harvested": sum(daily_stats[d]["crops_harvested"]["HARVEST"] for d in range(d_start, d_end)),
            "animals_fed": sum(daily_stats[d]["animals_fed"] for d in range(d_start, d_end)),
            "animals_cared": sum(daily_stats[d]["animals_cared"] for d in range(d_start, d_end)),
            "watered_count": sum(daily_stats[d]["watered_count"] for d in range(d_start, d_end)),
        }
        return p

    # Trapped capital values
    animal_trapped_cash = sum(trapped_animals[a] * BASE_PRICES.get(a, 300) for a in trapped_animals)
    shed_trapped_cash = sum(shed.get(item, 0) * BASE_PRICES.get(item, 25) for item in shed)
    seeds_trapped_cash = sum(seeds.get(c, 0) * SEED_PRICES.get(c, 20) for c in seeds)
    unharvested_crops_value = sum(trapped_crops[c]["yield_units"] * BASE_PRICES.get(c, 35) for c in trapped_crops)
    immature_crops_seed_cost = sum(trapped_crops[c]["immature"] * SEED_PRICES.get(c, 20) for c in trapped_crops)

    return {
        "episode": ep,
        "seed": seed,
        "model": os.path.basename(model_path),
        "is_swapped": is_swapped,
        "ssl_reward": final_ssl_reward,
        "starter_reward": final_starter_reward,
        "cash_by_day": cash_by_day,
        "phase_1": aggregate_phase(0, 10),
        "phase_2": aggregate_phase(10, 20),
        "phase_3": aggregate_phase(20, 30),
        "daily_breakdown": {d: {k: v if not isinstance(v, defaultdict) else dict(v) for k, v in daily_stats[d].items()} for d in range(30)},
        "trapped_assets": {
            "living_animals": dict(trapped_animals),
            "empty_coops": empty_coops,
            "empty_pastures": empty_pastures,
            "total_coops": trapped_coops,
            "total_pastures": trapped_pastures,
            "crops_in_ground": {k: dict(v) for k, v in trapped_crops.items()},
            "shed_inventory": dict(shed),
            "seeds_inventory": dict(seeds),
            "unlocked_quads": unlocked_quads,
            "financial_impact": {
                "animal_trapped_cost": animal_trapped_cash,
                "shed_unsold_value": shed_trapped_cash,
                "seeds_unused_value": seeds_trapped_cash,
                "unharvested_ripe_crops_value": unharvested_crops_value,
                "immature_crops_seed_sunk_cost": immature_crops_seed_cost,
                "total_trapped_deadweight": animal_trapped_cash + shed_trapped_cash + seeds_trapped_cash + unharvested_crops_value,
            }
        }
    }

def sum_subdicts(list_of_dicts):
    res = defaultdict(int)
    for d in list_of_dicts:
        for k, v in d.items():
            res[k] += v
    return res

if __name__ == "__main__":
    report_data = {}

    print("--- Running Match #2 (Grandmaster SSL - $6,250) ---")
    report_data["Match_2_Grandmaster"] = analyze_match("weights/grandmaster_ssl.pt", 2, 44, True)

    print("--- Running Match #8 (SSL AlphaZero - $8,288) ---")
    report_data["Match_8_AlphaZero"] = analyze_match("weights/ssl_alphazero.pt", 8, 50, True)

    print("--- Running Match #1 (Grandmaster SSL - $2,567) ---")
    report_data["Match_1_Grandmaster"] = analyze_match("weights/grandmaster_ssl.pt", 1, 43, False)

    print("--- Running Match #6 (Grandmaster SSL - $18) ---")
    report_data["Match_6_Grandmaster"] = analyze_match("weights/grandmaster_ssl.pt", 6, 48, True)

    print("--- Running Match #2 (SSL AlphaZero - $1,106) ---")
    report_data["Match_2_AlphaZero"] = analyze_match("weights/ssl_alphazero.pt", 2, 44, True)

    with open(".scratch/move_audit_data.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print("Audit data dumped successfully to .scratch/move_audit_data.json!")
