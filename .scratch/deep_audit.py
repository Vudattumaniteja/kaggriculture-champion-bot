import sys
import os
import json
from collections import defaultdict
sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from src.models.mcts import SSLMCTSAgent
from src.evaluation.runner import run_match

# Macro action names mapping
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

def analyze_match_deep(model_weights_path: str, ep: int, seed: int, is_swapped: bool):
    agent_inst = SSLMCTSAgent(
        model_weights_path=model_weights_path,
        num_simulations=20,
        c_puct=1.5,
        temperature=0.0,
    )

    history = []
    daily_stats = defaultdict(lambda: {
        "start_bank": 0,
        "end_bank": 0,
        "seeds_bought": defaultdict(int),
        "seed_spend": 0,
        "animals_bought": defaultdict(int),
        "animal_spend": 0,
        "labor_hires": 0,
        "labor_spend": 0,
        "land_buys": 0,
        "land_spend": 0,
        "items_sold": defaultdict(int),
        "sales_revenue": 0,
        "structures_built": defaultdict(int),
        "macro_actions": defaultdict(int),
        "farmer_actions": defaultdict(int),
        "hands_actions": defaultdict(int),
    })

    def wrapped_agent(obs, config=None):
        step = obs.get("step", 0)
        day = step // 24
        hour = step % 24
        
        # MCTS decision
        macro_act, probs, val = agent_inst.search_best_action(obs)
        act = agent_inst(obs, config)

        # Parse observation state
        players = obs.get("players", [])
        # Find which player index ssl_bot is
        p_idx = 1 if is_swapped else 0
        p_state = players[p_idx] if len(players) > p_idx else {}
        bank = p_state.get("bank", 0)

        daily_stats[day]["macro_actions"][MACRO_NAMES.get(macro_act, str(macro_act))] += 1

        # Track market spending and sales
        m_orders = act.get("market", [])
        for order in m_orders:
            op = order[0]
            if op == "BUY_SEED":
                cname, qty = order[1], order[2]
                daily_stats[day]["seeds_bought"][cname] += qty
                # Seed prices: WHEAT 10, CARROT 20, TOMATO 50, STRAWBERRY 100, MELON 80
                s_cost = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(cname, 20) * qty
                daily_stats[day]["seed_spend"] += s_cost
            elif op == "BUY_ANIMAL":
                aname, qty = order[1], order[2]
                daily_stats[day]["animals_bought"][aname] += qty
                a_cost = {"GOOSE": 300, "COW": 400, "SHEEP": 500}.get(aname, 300) * qty
                daily_stats[day]["animal_spend"] += a_cost
            elif op == "HIRE":
                daily_stats[day]["labor_hires"] += 1
                daily_stats[day]["labor_spend"] += 2
            elif op == "BUY_LAND":
                daily_stats[day]["land_buys"] += 1
                # NE=1000, SW=2000, SE=4000
                daily_stats[day]["land_spend"] += 1000 # approximate tracking
            elif op == "SELL":
                item, qty = order[1], order[2]
                daily_stats[day]["items_sold"][item] += qty
                # Estimate price
                p = BASE_PRICES.get(item, 25)
                daily_stats[day]["sales_revenue"] += p * qty

        # Track farmer / hands actions
        f_act = act.get("farmer", ["PASS"])
        f_op = f_act[0] if f_act else "PASS"
        daily_stats[day]["farmer_actions"][f_op] += 1
        if f_op in ["BUILD_COOP", "BUILD_PASTURE"]:
            daily_stats[day]["structures_built"][f_op] += 1

        for h_act in act.get("hands", []):
            h_op = h_act[0] if h_act else "PASS"
            daily_stats[day]["hands_actions"][h_op] += 1
            if h_op in ["BUILD_COOP", "BUILD_PASTURE"]:
                daily_stats[day]["structures_built"][h_op] += 1

        history.append({
            "step": step,
            "day": day,
            "hour": hour,
            "bank": bank,
            "macro_act": MACRO_NAMES.get(macro_act, str(macro_act)),
            "mcts_val": val,
            "farmer": act.get("farmer"),
            "hands": act.get("hands"),
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

    # Parse final board & trapped assets
    final_obs = env.steps[-1][ssl_idx].observation
    p_obs = final_obs.get("players", [{}])[ssl_idx] if "players" in final_obs else {}
    priv_obs = final_obs.get("private", {})
    
    # Board state at turn 719
    tiles = p_obs.get("tiles", []) or p_obs.get("grid", []) or []
    shed = priv_obs.get("shed", {})
    seeds_left = priv_obs.get("seeds", {})
    unlocked_quads = p_obs.get("unlocked_quads", [])

    living_animals = defaultdict(int)
    structures_count = defaultdict(int)
    crops_in_ground = defaultdict(lambda: {"total": 0, "ripe": 0, "immature": 0, "yield_units": 0})
    
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "COOP":
                    structures_count["COOP"] += 1
                    anim = t.get("animal")
                    if anim:
                        living_animals[anim] += 1
                elif k == "PASTURE":
                    structures_count["PASTURE"] += 1
                    anim = t.get("animal")
                    if anim:
                        living_animals[anim] += 1
                elif k == "PLANT":
                    cname = t.get("crop", "UNKNOWN")
                    crops_in_ground[cname]["total"] += 1
                    age = 29 - t.get("planted_day", 29) # day 29 is last day
                    yield_u = t.get("yield_units", 0)
                    if yield_u > 0:
                        crops_in_ground[cname]["ripe"] += 1
                        crops_in_ground[cname]["yield_units"] += yield_u
                    else:
                        crops_in_ground[cname]["immature"] += 1

    # Trapped capital calculation
    trapped_animal_value = sum(living_animals[a] * BASE_PRICES.get(a, 300) for a in living_animals)
    trapped_shed_value = sum(shed.get(item, 0) * BASE_PRICES.get(item, 25) for item in shed if item not in ["GOOSE", "COW", "SHEEP"])
    trapped_seeds_value = sum(seeds_left.get(crop, 0) * {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}.get(crop, 20) for crop in seeds_left)
    trapped_crop_potential = sum(crops_in_ground[c]["ripe"] * BASE_PRICES.get(c, 35) * 4 for c in crops_in_ground) # rough potential
    
    # Calculate bank at boundaries: Day 0, Day 10, Day 20, Day 30
    bank_checkpoints = {}
    for log in history:
        d = log["day"]
        h = log["hour"]
        if h == 0 and d in [0, 5, 10, 15, 20, 25, 29]:
            bank_checkpoints[f"Day {d}"] = log["bank"]
    bank_checkpoints["Day 30 (Final)"] = final_ssl_reward

    # Aggregate by 10-day phases
    phase_summary = {
        "Phase 1 (Days 0-9)": {
            "seed_spend": sum(daily_stats[d]["seed_spend"] for d in range(0, 10)),
            "seeds_bought": dict(sum_dicts([daily_stats[d]["seeds_bought"] for d in range(0, 10)])),
            "animal_spend": sum(daily_stats[d]["animal_spend"] for d in range(0, 10)),
            "animals_bought": dict(sum_dicts([daily_stats[d]["animals_bought"] for d in range(0, 10)])),
            "labor_spend": sum(daily_stats[d]["labor_spend"] for d in range(0, 10)),
            "land_buys": sum(daily_stats[d]["land_buys"] for d in range(0, 10)),
            "land_spend": sum(daily_stats[d]["land_spend"] for d in range(0, 10)),
            "sales_revenue": sum(daily_stats[d]["sales_revenue"] for d in range(0, 10)),
            "items_sold": dict(sum_dicts([daily_stats[d]["items_sold"] for d in range(0, 10)])),
            "coops_built": sum(daily_stats[d]["structures_built"]["BUILD_COOP"] for d in range(0, 10)),
            "pastures_built": sum(daily_stats[d]["structures_built"]["BUILD_PASTURE"] for d in range(0, 10)),
            "macro_actions": dict(sum_dicts([daily_stats[d]["macro_actions"] for d in range(0, 10)])),
        },
        "Phase 2 (Days 10-19)": {
            "seed_spend": sum(daily_stats[d]["seed_spend"] for d in range(10, 20)),
            "seeds_bought": dict(sum_dicts([daily_stats[d]["seeds_bought"] for d in range(10, 20)])),
            "animal_spend": sum(daily_stats[d]["animal_spend"] for d in range(10, 20)),
            "animals_bought": dict(sum_dicts([daily_stats[d]["animals_bought"] for d in range(10, 20)])),
            "labor_spend": sum(daily_stats[d]["labor_spend"] for d in range(10, 20)),
            "land_buys": sum(daily_stats[d]["land_buys"] for d in range(10, 20)),
            "land_spend": sum(daily_stats[d]["land_spend"] for d in range(10, 20)),
            "sales_revenue": sum(daily_stats[d]["sales_revenue"] for d in range(10, 20)),
            "items_sold": dict(sum_dicts([daily_stats[d]["items_sold"] for d in range(10, 20)])),
            "coops_built": sum(daily_stats[d]["structures_built"]["BUILD_COOP"] for d in range(10, 20)),
            "pastures_built": sum(daily_stats[d]["structures_built"]["BUILD_PASTURE"] for d in range(10, 20)),
            "macro_actions": dict(sum_dicts([daily_stats[d]["macro_actions"] for d in range(10, 20)])),
        },
        "Phase 3 (Days 20-29)": {
            "seed_spend": sum(daily_stats[d]["seed_spend"] for d in range(20, 30)),
            "seeds_bought": dict(sum_dicts([daily_stats[d]["seeds_bought"] for d in range(20, 30)])),
            "animal_spend": sum(daily_stats[d]["animal_spend"] for d in range(20, 30)),
            "animals_bought": dict(sum_dicts([daily_stats[d]["animals_bought"] for d in range(20, 30)])),
            "labor_spend": sum(daily_stats[d]["labor_spend"] for d in range(20, 30)),
            "land_buys": sum(daily_stats[d]["land_buys"] for d in range(20, 30)),
            "land_spend": sum(daily_stats[d]["land_spend"] for d in range(20, 30)),
            "sales_revenue": sum(daily_stats[d]["sales_revenue"] for d in range(20, 30)),
            "items_sold": dict(sum_dicts([daily_stats[d]["items_sold"] for d in range(20, 30)])),
            "coops_built": sum(daily_stats[d]["structures_built"]["BUILD_COOP"] for d in range(20, 30)),
            "pastures_built": sum(daily_stats[d]["structures_built"]["BUILD_PASTURE"] for d in range(20, 30)),
            "macro_actions": dict(sum_dicts([daily_stats[d]["macro_actions"] for d in range(20, 30)])),
        },
    }

    return {
        "episode": ep,
        "seed": seed,
        "model": os.path.basename(model_weights_path),
        "is_swapped": is_swapped,
        "ssl_reward": final_ssl_reward,
        "starter_reward": final_starter_reward,
        "bank_checkpoints": bank_checkpoints,
        "phase_summary": phase_summary,
        "trapped_assets": {
            "living_animals": dict(living_animals),
            "structures": dict(structures_count),
            "crops_in_ground": {k: dict(v) for k, v in crops_in_ground.items()},
            "shed": dict(shed),
            "seeds_in_pocket": dict(seeds_left),
            "unlocked_quads": unlocked_quads,
            "trapped_animal_value": trapped_animal_value,
            "trapped_shed_value": trapped_shed_value,
            "trapped_seeds_value": trapped_seeds_value,
            "trapped_crop_potential": trapped_crop_potential,
            "total_trapped_capital_est": trapped_animal_value + trapped_shed_value + trapped_seeds_value + trapped_crop_potential,
        }
    }

def sum_dicts(dict_list):
    res = defaultdict(int)
    for d in dict_list:
        for k, v in d.items():
            res[k] += v
    return res

if __name__ == "__main__":
    results = {}

    # 1. Match #2 Grandmaster ($6,250)
    print("Simulating Grandmaster Match #2 (Seed: 44, Swapped: True)...")
    results["GM_Match_2"] = analyze_match_deep("weights/grandmaster_ssl.pt", 2, 44, is_swapped=True)

    # 2. Match #8 SSL AlphaZero ($8,288)
    print("Simulating AlphaZero Match #8 (Seed: 50, Swapped: True)...")
    results["AZ_Match_8"] = analyze_match_deep("weights/ssl_alphazero.pt", 8, 50, is_swapped=True)

    # 3. Match #1 Grandmaster ($2,567)
    print("Simulating Grandmaster Match #1 (Seed: 43, Swapped: False)...")
    results["GM_Match_1"] = analyze_match_deep("weights/grandmaster_ssl.pt", 1, 43, is_swapped=False)

    # 4. Match #6 Grandmaster ($18 - worst crash)
    print("Simulating Grandmaster Match #6 (Seed: 48, Swapped: True)...")
    results["GM_Match_6"] = analyze_match_deep("weights/grandmaster_ssl.pt", 6, 48, is_swapped=True)

    # 5. Match #2 SSL AlphaZero ($1,106)
    print("Simulating AlphaZero Match #2 (Seed: 44, Swapped: True)...")
    results["AZ_Match_2"] = analyze_match_deep("weights/ssl_alphazero.pt", 2, 44, is_swapped=True)

    # 6. Match #5 Grandmaster ($4,906)
    print("Simulating Grandmaster Match #5 (Seed: 47, Swapped: False)...")
    results["GM_Match_5"] = analyze_match_deep("weights/grandmaster_ssl.pt", 5, 47, is_swapped=False)

    with open(".scratch/deep_audit_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDeep Audit Simulation completed! Results saved to .scratch/deep_audit_results.json")
