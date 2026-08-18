"""
Advanced Hybrid Expert Farmer baseline agent.
Synthesizes multi-crop rotations, livestock husbandry (geese, cows, sheep, coops, pastures),
daily wheat feeding, daily care compounding bonuses, fertilizer application,
land quadrant expansion, and synchronized market arbitrage timing.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from src.agents.utils import DIRS, SHED_TILES, get_manhattan_dist, get_step_towards, parse_observation
except ImportError:
    from utils import DIRS, SHED_TILES, get_manhattan_dist, get_step_towards, parse_observation

CROP_SPECS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
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
}


class HybridExpertAgent:
    def __init__(
        self,
        target_geese: int = 2,
        target_cows: int = 1,
        target_sheep: int = 2,
        wheat_target_tiles: int = 6,
    ):
        self.target_geese = target_geese
        self.target_cows = target_cows
        self.target_sheep = target_sheep
        self.wheat_target_tiles = wheat_target_tiles

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        step = state["step"]
        money = state["money"]
        farmer_pos = state["farmer_pos"]
        hands_pos = state["hands_pos"]
        tiles = state["tiles"]
        shed = state["shed"]
        seeds = state["seeds"]
        inventories = state["inventories"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        market_prices = state["market_prices"]
        unlocked_shops = state["unlocked_shops"]

        market_orders: List[List[Any]] = []

        # --- 1. AUDIT FARM ASSETS & STRUCTURES ---
        living_geese = 0
        living_cows = 0
        living_sheep = 0
        empty_coops = 0
        empty_pastures = 0
        planted_crops: Dict[str, int] = {c: 0 for c in CROP_SPECS}
        animal_tiles: List[Tuple[int, int, str, Dict[str, Any]]] = []
        crop_tiles: List[Tuple[int, int, Dict[str, Any]]] = []
        unlocked_empty_tiles: List[Tuple[int, int]] = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    unlocked_empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    animal = t.get("animal")
                    if k == "COOP":
                        if animal == "GOOSE":
                            living_geese += 1
                            animal_tiles.append((c, r, "GOOSE", t))
                        elif animal is None:
                            empty_coops += 1
                    elif k == "PASTURE":
                        if animal == "COW":
                            living_cows += 1
                            animal_tiles.append((c, r, "COW", t))
                        elif animal == "SHEEP":
                            living_sheep += 1
                            animal_tiles.append((c, r, "SHEEP", t))
                        elif animal is None:
                            empty_pastures += 1
                    elif k == "PLANT":
                        crop_name = t.get("crop", "CARROT")
                        crop_tiles.append((c, r, t))
                        planted_crops[crop_name] = planted_crops.get(crop_name, 0) + 1

        total_living_animals = living_geese + living_cows + living_sheep

        # --- 2. MARKET TRADING & CAPITAL ALLOCATION ---
        # A. Sell harvested animal produce & crops from shed
        for item in ["EGG", "MILK", "WOOL", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "FERTILIZER"]:
            count = shed.get(item, 0)
            if count > 0:
                if item == "FERTILIZER" and len(crop_tiles) > 0 and day <= 22 and count > 3:
                    market_orders.append(["SELL", item, count - 3])
                elif item != "FERTILIZER":
                    market_orders.append(["SELL", item, count])

        # Wheat reserve management for feeding
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_req = total_living_animals
        wheat_safe_res = max(6, daily_feed_req * 3)

        if wheat_in_shed > wheat_safe_res + 10:
            market_orders.append(["SELL", "WHEAT", wheat_in_shed - wheat_safe_res])
        elif wheat_in_shed < daily_feed_req and money >= 60 and day < 28:
            qty = min(6, wheat_safe_res - wheat_in_shed)
            market_orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= 25 * qty
            shed["WHEAT"] = wheat_in_shed + qty

        # B. Land Expansion: Purchase NE and SW early/mid game
        if day <= 16:
            if "NE" not in unlocked and money >= 1250:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and money >= 2500 and day <= 12:
                market_orders.append(["BUY_LAND"])
                money -= 2000

        # C. Labor Hiring (2 farmhands every day for $2 total cost)
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])
            money -= 2

        # D. Animal Purchasing
        geese_in_shed = shed.get("GOOSE", 0)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)

        if (
            empty_coops > geese_in_shed
            and (living_geese + geese_in_shed) < self.target_geese
            and money >= 380
            and day <= 22
        ):
            market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
            money -= 300
            geese_in_shed += 1
        elif (
            empty_pastures > (cows_in_shed + sheep_in_shed)
            and (living_cows + cows_in_shed) < self.target_cows
            and money >= 550
            and day <= 18
        ):
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_in_shed += 1
        elif (
            empty_pastures > (cows_in_shed + sheep_in_shed)
            and (living_sheep + sheep_in_shed) < self.target_sheep
            and money >= 650
            and day <= 16
        ):
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_in_shed += 1

        # E. Seed Purchasing: Balanced between Wheat feed, high-value Melons/Strawberries, and rapid Carrots
        wheat_seeds = seeds.get("WHEAT", 0)
        planted_wheat = planted_crops.get("WHEAT", 0)
        if day < 26 and (wheat_seeds + planted_wheat) < self.wheat_target_tiles and money >= 40:
            qty = min(4, self.wheat_target_tiles - (wheat_seeds + planted_wheat))
            market_orders.append(["BUY_SEED", "WHEAT", qty])
            money -= 10 * qty
            seeds["WHEAT"] = wheat_seeds + qty

        # Other crops portfolio
        if day < 27 and money >= 60:
            if day <= 10 and (seeds.get("MELON", 0) + planted_crops.get("MELON", 0)) < 4 and money >= 200:
                market_orders.append(["BUY_SEED", "MELON", 2])
                money -= 160
            elif day <= 14 and (seeds.get("STRAWBERRY", 0) + planted_crops.get("STRAWBERRY", 0)) < 4 and money >= 250:
                market_orders.append(["BUY_SEED", "STRAWBERRY", 2])
                money -= 200
            elif (seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0)) < 12 and money >= 60:
                qty = min(4, 12 - (seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0)))
                market_orders.append(["BUY_SEED", "CARROT", qty])
                money -= 20 * qty

        # --- 3. FIELD SCANNING & TASK COMPILATION ---
        feed_tasks: List[Tuple[int, int]] = []
        care_tasks: List[Tuple[int, int]] = []
        harvest_animal_tasks: List[Tuple[int, int]] = []
        fertilizer_tasks: List[Tuple[int, int]] = []

        water_crop_tasks: List[Tuple[int, int]] = []
        harvest_crop_tasks: List[Tuple[int, int]] = []
        fertilize_crop_tasks: List[Tuple[int, int]] = []

        weed_tasks: List[Tuple[int, int]] = []
        plant_tasks: List[Tuple[int, int]] = []
        build_coop_tasks: List[Tuple[int, int]] = []
        build_pasture_tasks: List[Tuple[int, int]] = []
        place_animal_tasks: List[Tuple[int, int, str]] = []

        for ax, ay, aname, atile in animal_tiles:
            pos = (ax, ay)
            if not atile.get("fed_today", False):
                feed_tasks.append(pos)
            if not atile.get("cared_today", False):
                care_tasks.append(pos)
            if atile.get("yield_units", 0) > 0:
                harvest_animal_tasks.append(pos)
            if atile.get("fertilizer_available", False):
                fertilizer_tasks.append(pos)

        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            crop = ctile.get("crop", "CARROT")
            cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
            age = day - ctile.get("planted_day", day)
            is_ongoing = cspec["ongoing"]
            yield_u = ctile.get("yield_units", 0)

            if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
            ):
                harvest_crop_tasks.append(pos)
            elif not ctile.get("watered_today", False):
                water_crop_tasks.append(pos)
            elif ctile.get("fertilized_until_day", -1) < day:
                fertilize_crop_tasks.append(pos)

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if isinstance(t, dict):
                    if t.get("kind") == "WEED":
                        weed_tasks.append(pos)
                    elif t.get("kind") == "COOP" and "animal" not in t:
                        place_animal_tasks.append((c, r, "GOOSE"))
                    elif t.get("kind") == "PASTURE" and "animal" not in t:
                        if (living_cows + cows_in_shed) <= (living_sheep + sheep_in_shed):
                            place_animal_tasks.append((c, r, "COW"))
                        else:
                            place_animal_tasks.append((c, r, "SHEEP"))

        unlocked_empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))

        # Build structure tasks
        curr_e_idx = 0
        if (living_geese + empty_coops) < self.target_geese and curr_e_idx < len(unlocked_empty_tiles) and day <= 20:
            build_coop_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1

        if (
            (living_cows + living_sheep + empty_pastures) < (self.target_cows + self.target_sheep)
            and curr_e_idx < len(unlocked_empty_tiles)
            and day <= 18
        ):
            build_pasture_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1

        total_seeds_count = sum(seeds.values())
        if day < 27 and total_seeds_count > 0:
            for p in unlocked_empty_tiles[curr_e_idx:24]:
                plant_tasks.append(p)

        # --- 4. MULTI-WORKER DISPATCH ---
        units = [farmer_pos] + hands_pos
        unit_actions: List[List[Any]] = []
        claimed_tasks: Set[Tuple[int, int]] = set()

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in SHED_TILES

            # 1. Shed Pickup Routine
            if is_at_shed:
                placed_target = next((pt for pt in place_animal_tasks if pt[:2] not in claimed_tasks), None)
                if placed_target is not None:
                    target_animal = placed_target[2]
                    if shed.get(target_animal, 0) > 0 and u_inv.get(target_animal, 0) == 0:
                        unit_actions.append(["PICKUP", target_animal, 1])
                        shed[target_animal] -= 1
                        u_inv[target_animal] = u_inv.get(target_animal, 0) + 1
                        continue

                if feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    qty = min(4, shed.get("WHEAT", 0), len(feed_tasks))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "WHEAT", qty])
                        shed["WHEAT"] -= qty
                        u_inv["WHEAT"] = u_inv.get("WHEAT", 0) + qty
                        continue

                if fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "FERTILIZER", qty])
                        shed["FERTILIZER"] -= qty
                        u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + qty
                        continue

            # 2. Standing Tile Actions
            if isinstance(u_tile, dict):
                k = u_tile.get("kind")
                animal = u_tile.get("animal")

                if k in ["COOP", "PASTURE"]:
                    if animal is not None:
                        if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                            unit_actions.append(["FEED"])
                            u_inv["WHEAT"] -= 1
                            continue
                        if not u_tile.get("cared_today", False):
                            unit_actions.append(["CARE"])
                            continue
                        if u_tile.get("fertilizer_available", False):
                            unit_actions.append(["COLLECT_FERTILIZER"])
                            u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + 1
                            continue
                        if u_tile.get("yield_units", 0) > 0:
                            unit_actions.append(["HARVEST"])
                            continue
                    else:
                        for aname in ["GOOSE", "COW", "SHEEP"]:
                            if u_inv.get(aname, 0) > 0:
                                req_struct = "COOP" if aname == "GOOSE" else "PASTURE"
                                if k == req_struct:
                                    unit_actions.append(["PLACE", aname])
                                    u_inv[aname] -= 1
                                    break
                        if len(unit_actions) > u_idx:
                            continue

                elif k == "PLANT":
                    crop = u_tile.get("crop", "CARROT")
                    cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                    age = day - u_tile.get("planted_day", day)
                    is_ongoing = cspec["ongoing"]
                    yield_u = u_tile.get("yield_units", 0)

                    if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                        is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        unit_actions.append(["HARVEST"])
                        continue
                    if not u_tile.get("watered_today", False):
                        unit_actions.append(["WATER"])
                        continue
                    if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0:
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue

                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue

            elif u_tile is None:
                if u_pos in build_coop_tasks:
                    unit_actions.append(["BUILD_COOP"])
                    build_coop_tasks.remove(u_pos)
                    continue
                elif u_pos in build_pasture_tasks:
                    unit_actions.append(["BUILD_PASTURE"])
                    build_pasture_tasks.remove(u_pos)
                    continue
                elif u_pos in plant_tasks and total_seeds_count > 0 and day < 27:
                    for crop_cand in ["MELON", "STRAWBERRY", "TOMATO", "WHEAT", "CARROT"]:
                        if seeds.get(crop_cand, 0) > 0:
                            unit_actions.append(["PLANT", crop_cand])
                            seeds[crop_cand] -= 1
                            total_seeds_count -= 1
                            break
                    else:
                        unit_actions.append(["PASS"])
                    continue

            # 3. Pathfinding & Task Navigation
            best_target: Optional[Tuple[int, int]] = None
            best_dist = 999

            has_animal_in_hand = any(u_inv.get(a, 0) > 0 for a in ["GOOSE", "COW", "SHEEP"])
            if has_animal_in_hand:
                for pt in place_animal_tasks:
                    pos = pt[:2]
                    if pos not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, pos)
                        if d < best_dist:
                            best_dist = d
                            best_target = pos

            if best_target is None and u_inv.get("WHEAT", 0) > 0 and feed_tasks:
                for task in feed_tasks:
                    if task not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, task)
                        if d < best_dist:
                            best_dist = d
                            best_target = task

            if best_target is None and feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                best_target = nearest_shed
                best_dist = get_manhattan_dist(u_pos, nearest_shed)

            if best_target is None:
                priority_queue = (
                    care_tasks
                    + water_crop_tasks
                    + harvest_animal_tasks
                    + harvest_crop_tasks
                    + fertilizer_tasks
                    + build_coop_tasks
                    + build_pasture_tasks
                    + plant_tasks
                    + weed_tasks
                )
                for task in priority_queue:
                    if task in claimed_tasks:
                        continue
                    d = get_manhattan_dist(u_pos, task)
                    if d < best_dist:
                        best_dist = d
                        best_target = task

            if best_target is not None:
                claimed_tasks.add(best_target)
                unit_actions.append([get_step_towards(u_pos, best_target)])
            else:
                if not is_at_shed:
                    nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                    unit_actions.append([get_step_towards(u_pos, nearest_shed)])
                else:
                    unit_actions.append(["PASS"])

        farmer_action = unit_actions[0] if unit_actions else ["PASS"]
        hands_actions = unit_actions[1:] if len(unit_actions) > 1 else []

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_orders[:10],
        }


hybrid_expert_agent = HybridExpertAgent()


def agent(obs, config=None):
    return hybrid_expert_agent(obs, config)
