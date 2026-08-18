import os
import sys

sys.path.insert(0, os.path.abspath("."))

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from kaggle_environments import make

CROP_SPECS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
}

BASE_PRICES = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

def get_manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_step_towards(curr: Tuple[int, int], target: Tuple[int, int]) -> str:
    cx, cy = curr
    tx, ty = target
    if cx == tx and cy == ty:
        return "PASS"
    dx = tx - cx
    dy = ty - cy
    if abs(dx) >= abs(dy):
        if dx > 0:
            return "EAST"
        elif dx < 0:
            return "WEST"
    if dy > 0:
        return "SOUTH"
    elif dy < 0:
        return "NORTH"
    return "PASS"

def parse_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    opp_farm = farms[1 - player] if 1 - player < len(farms) else {}
    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}
    town = obs.get("town", {}) or {}

    return {
        "day": int(obs.get("day", 0)),
        "hour": int(obs.get("hour", 0)),
        "step": int(obs.get("step", 0)),
        "money": float(my_farm.get("money", 3000.0)),
        "opp_money": float(opp_farm.get("money", 3000.0)),
        "farmer_pos": tuple(my_farm.get("farmer", [4, 4])),
        "hands_pos": [tuple(h) for h in my_farm.get("hands", [])],
        "tiles": my_farm.get("tiles", []),
        "shed": private.get("shed", {}),
        "seeds": private.get("seeds", {}),
        "inventories": private.get("inventories", []),
        "hires_today": int(my_farm.get("hires_today", 0)),
        "unlocked_quads": my_farm.get("unlocked_quadrants", ["NW"]),
        "market_prices": market.get("prices", {}),
        "unlocked_shops": town.get("unlocked_shops", []),
    }

class SmartMarketTimingGrandmasterAgent:
    def __init__(
        self,
        target_geese: int = 2,
        target_cows: int = 1,
        target_sheep: int = 1,
        max_workers_early: int = 2,
        max_workers_late: int = 5,
    ):
        self.target_geese = target_geese
        self.target_cows = target_cows
        self.target_sheep = target_sheep
        self.max_workers_early = max_workers_early
        self.max_workers_late = max_workers_late
        self.shed_tiles = [(4, 4), (5, 4), (4, 5), (5, 5)]

    def __call__(self, obs: Dict[str, Any], config: Any = None, macro_action: Optional[int] = None) -> Dict[str, Any]:
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

        # 1. Turn 718-719 Complete Liquidation Trigger ($0 Deadweight)
        if step >= 718 or (day == 29 and hour >= 22) or macro_action == 7:
            for item in PRODUCTS:
                cnt = shed.get(item, 0)
                if cnt > 0:
                    market_orders.append(["SELL", item, cnt])

            units = [farmer_pos] + hands_pos
            unit_actions = []
            for u_pos in units:
                ux, uy = u_pos
                u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
                if isinstance(u_tile, dict) and u_tile.get("yield_units", 0) > 0:
                    unit_actions.append(["HARVEST"])
                elif u_pos not in self.shed_tiles:
                    nearest = min(self.shed_tiles, key=lambda s: get_manhattan_dist(u_pos, s))
                    unit_actions.append([get_step_towards(u_pos, nearest)])
                else:
                    unit_actions.append(["PASS"])

            return {
                "farmer": unit_actions[0] if unit_actions else ["PASS"],
                "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
                "market": market_orders[:10],
            }

        # 2. Asset & Structure Audit
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
                    if pos not in self.shed_tiles:
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

        # 3. SMART DYNAMIC INVENTORY & MARKET TIMING ENGINE
        # Price Reservation Floors:
        # Fertilizer >= $60, Wool >= $120, Milk >= $100, Melons >= $180
        # Strawberries >= $85, Tomatoes >= $45, Eggs >= $35, Carrots >= $25, Wheat >= $18
        BASE_RESERVATION_FLOORS = {
            "FERTILIZER": 60,
            "WOOL": 120,
            "MILK": 100,
            "MELON": 180,
            "STRAWBERRY": 85,
            "TOMATO": 45,
            "EGG": 35,
            "CARROT": 25,
            "WHEAT": 18,
        }

        # Dynamic Floor Scaling for Days 28-29 Final Flush:
        if step >= 715 or (day == 29 and hour >= 19):
            floor_mult = 0.0  # 100% liquidation
        elif day == 29:
            floor_mult = 0.30  # Aggressive liquidation
        elif day == 28:
            floor_mult = 0.65  # Orderly clearing
        else:
            floor_mult = 1.0   # Smart reservation floor holding

        effective_floors = {
            k: BASE_RESERVATION_FLOORS[k] * floor_mult
            for k in BASE_RESERVATION_FLOORS
        }

        # A. Sell against Reservation Floors
        for item in ["MELON", "WOOL", "MILK", "STRAWBERRY", "TOMATO", "EGG", "CARROT", "FERTILIZER"]:
            count = shed.get(item, 0)
            if count <= 0:
                continue

            cur_price = float(market_prices.get(item, BASE_PRICES.get(item, 50)))
            floor_price = effective_floors.get(item, 0)

            if item == "FERTILIZER":
                needed_res = 4 if (len(crop_tiles) > 0 and day <= 22) else 0
                sellable = max(0, count - needed_res)
                if sellable > 0 and (cur_price >= floor_price or floor_mult == 0.0):
                    market_orders.append(["SELL", item, sellable])
                    shed[item] -= sellable
            else:
                if cur_price >= floor_price or floor_mult == 0.0:
                    market_orders.append(["SELL", item, count])
                    shed[item] -= count

        # B. Wheat Reserve Management for Animal Feed
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_req = total_living_animals
        wheat_safe_res = max(6, daily_feed_req * 3) if day < 28 else 0
        cur_wheat_price = float(market_prices.get("WHEAT", 25))

        if wheat_in_shed > wheat_safe_res:
            surplus_wheat = wheat_in_shed - wheat_safe_res
            if cur_wheat_price >= effective_floors.get("WHEAT", 18) or floor_mult == 0.0:
                market_orders.append(["SELL", "WHEAT", surplus_wheat])
                shed["WHEAT"] -= surplus_wheat
        elif wheat_in_shed < daily_feed_req and money >= 60 and day < 28:
            qty = min(6, wheat_safe_res - wheat_in_shed)
            market_orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= 25 * qty
            shed["WHEAT"] = wheat_in_shed + qty

        # C. 85-Unit Shed Buffering: Trim lower priority inventory if approaching overflow cap (100)
        remaining_shed_items = sum(shed.get(item, 0) for item in PRODUCTS)
        if remaining_shed_items > 85:
            excess_to_trim = remaining_shed_items - 85
            trim_order = ["CARROT", "WHEAT", "TOMATO", "EGG", "STRAWBERRY", "FERTILIZER", "MILK", "WOOL", "MELON"]
            for trim_item in trim_order:
                if excess_to_trim <= 0:
                    break
                avail = shed.get(trim_item, 0)
                if trim_item == "WHEAT":
                    avail = max(0, avail - daily_feed_req)
                elif trim_item == "FERTILIZER" and day <= 22:
                    avail = max(0, avail - 2)

                if avail > 0:
                    trim_qty = min(avail, excess_to_trim)
                    market_orders.append(["SELL", trim_item, trim_qty])
                    shed[trim_item] -= trim_qty
                    excess_to_trim -= trim_qty

        # D. Land Expansion
        if (macro_action in [3, None]) and day <= 24:
            if "NE" not in unlocked and (day <= 14 or money >= 1250) and money >= 1000 and day <= 22:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and (day <= 16 or money >= 2500) and money >= 2000 and day <= 22:
                market_orders.append(["BUY_LAND"])
                money -= 2000
            elif "SE" not in unlocked and "SW" in unlocked and "NE" in unlocked and money >= 5000 and day <= 24:
                market_orders.append(["BUY_LAND"])
                money -= 4000

        # E. Labor Hiring
        max_hires = self.max_workers_late if (day >= 15 or macro_action == 4) else self.max_workers_early
        if day < 28 and hour <= 2 and hires_today < max_hires and money >= 50:
            market_orders.append(["HIRE"])
            money -= 2

        # F. Animals Buying (Strictly check empty pens)
        geese_in_shed = shed.get("GOOSE", 0)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)
        geese_in_hand = sum(inv.get("GOOSE", 0) for inv in inventories)
        cows_in_hand = sum(inv.get("COW", 0) for inv in inventories)
        sheep_in_hand = sum(inv.get("SHEEP", 0) for inv in inventories)

        available_empty_coops = max(0, empty_coops - geese_in_shed - geese_in_hand)
        available_empty_pastures = max(0, empty_pastures - (cows_in_shed + cows_in_hand + sheep_in_shed + sheep_in_hand))

        if available_empty_coops > 0 and (living_geese + geese_in_shed + geese_in_hand) < self.target_geese and money >= 380 and day <= 22:
            market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
            money -= 300
            geese_in_shed += 1
            available_empty_coops -= 1
        elif available_empty_pastures > 0 and (living_cows + cows_in_shed + cows_in_hand) < self.target_cows and money >= 550 and day <= 18:
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_in_shed += 1
            available_empty_pastures -= 1
        elif available_empty_pastures > 0 and (living_sheep + sheep_in_shed + sheep_in_hand) < self.target_sheep and money >= 650 and day <= 16:
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_in_shed += 1
            available_empty_pastures -= 1

        # G. Seed Purchasing
        wheat_seeds = seeds.get("WHEAT", 0)
        planted_wheat = planted_crops.get("WHEAT", 0)
        if day <= 24 and (wheat_seeds + planted_wheat) < 6 and money >= 40:
            qty = min(4, 6 - (wheat_seeds + planted_wheat))
            market_orders.append(["BUY_SEED", "WHEAT", qty])
            money -= 10 * qty
            seeds["WHEAT"] = wheat_seeds + qty

        melon_budget = seeds.get("MELON", 0) + planted_crops.get("MELON", 0)
        if ((day <= 2) or (10 <= day <= 12) or macro_action == 2) and day <= 14 and melon_budget < 6 and money >= 200:
            qty = min(4, 6 - melon_budget)
            market_orders.append(["BUY_SEED", "MELON", qty])
            money -= 80 * qty
            seeds["MELON"] = seeds.get("MELON", 0) + qty

        tomato_budget = seeds.get("TOMATO", 0) + planted_crops.get("TOMATO", 0)
        if (day <= 4 or macro_action == 2) and day <= 5 and tomato_budget < 6 and money >= 150:
            qty = min(4, 6 - tomato_budget)
            market_orders.append(["BUY_SEED", "TOMATO", qty])
            money -= 50 * qty
            seeds["TOMATO"] = seeds.get("TOMATO", 0) + qty

        carrot_budget = seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0)
        if day <= 25 and carrot_budget < 16 and money >= 60:
            qty = min(6, 16 - carrot_budget)
            market_orders.append(["BUY_SEED", "CARROT", qty])
            money -= 20 * qty
            seeds["CARROT"] = seeds.get("CARROT", 0) + qty

        # 4. Field Tasks Scanning
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

            if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 27 and yield_u > 0))) or (
                is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
            ):
                harvest_crop_tasks.append(pos)
            elif not ctile.get("watered_today", False) and day < 28:
                water_crop_tasks.append(pos)
            elif ctile.get("fertilized_until_day", -1) < day and day <= 22:
                fertilize_crop_tasks.append(pos)

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if isinstance(t, dict):
                    if t.get("kind") == "WEED":
                        weed_tasks.append(pos)
                    elif t.get("kind") == "COOP" and t.get("animal") is None:
                        place_animal_tasks.append((c, r, "GOOSE"))
                    elif t.get("kind") == "PASTURE" and t.get("animal") is None:
                        if (cows_in_shed + cows_in_hand) > 0:
                            place_animal_tasks.append((c, r, "COW"))
                        elif (sheep_in_shed + sheep_in_hand) > 0:
                            place_animal_tasks.append((c, r, "SHEEP"))
                        elif (living_cows + cows_in_shed + cows_in_hand) <= (living_sheep + sheep_in_shed + sheep_in_hand):
                            place_animal_tasks.append((c, r, "COW"))
                        else:
                            place_animal_tasks.append((c, r, "SHEEP"))

        unlocked_empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
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
        if day <= 25 and total_seeds_count > 0:
            for p in unlocked_empty_tiles[curr_e_idx:40]:
                plant_tasks.append(p)

        # 5. Multi-Worker Dispatch
        units = [farmer_pos] + hands_pos
        unit_actions: List[List[Any]] = []
        claimed_tasks: Set[Tuple[int, int]] = set()

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in self.shed_tiles

            # Atomic Shed Pickup
            if is_at_shed:
                placed_target = next((pt for pt in place_animal_tasks if pt[:2] not in claimed_tasks), None)
                if placed_target is not None:
                    target_animal = placed_target[2]
                    if shed.get(target_animal, 0) > 0 and u_inv.get(target_animal, 0) == 0:
                        unit_actions.append(["PICKUP", target_animal, 1])
                        shed[target_animal] -= 1
                        u_inv[target_animal] = u_inv.get(target_animal, 0) + 1
                        claimed_tasks.add(placed_target[:2])
                        continue

                if feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    qty = min(4, shed.get("WHEAT", 0), len(feed_tasks))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "WHEAT", qty])
                        shed["WHEAT"] -= qty
                        u_inv["WHEAT"] = u_inv.get("WHEAT", 0) + qty
                        continue

                if fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0 and day <= 22:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "FERTILIZER", qty])
                        shed["FERTILIZER"] -= qty
                        u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + qty
                        continue

            # Standing Tile Actions
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

                    if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 27 and yield_u > 0))) or (
                        is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        unit_actions.append(["HARVEST"])
                        continue
                    if not u_tile.get("watered_today", False) and day < 28:
                        unit_actions.append(["WATER"])
                        continue
                    if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0 and day <= 22:
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
                elif u_pos in plant_tasks and total_seeds_count > 0 and day <= 25:
                    for crop_cand in ["MELON", "TOMATO", "STRAWBERRY", "WHEAT", "CARROT"]:
                        if seeds.get(crop_cand, 0) > 0:
                            unit_actions.append(["PLANT", crop_cand])
                            seeds[crop_cand] -= 1
                            total_seeds_count -= 1
                            break
                    else:
                        unit_actions.append(["PASS"])
                    continue

            # Target Navigation
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
                nearest_shed = min(self.shed_tiles, key=lambda s: get_manhattan_dist(u_pos, s))
                best_target = nearest_shed
                best_dist = get_manhattan_dist(u_pos, nearest_shed)

            if best_target is None:
                priority_queue = (
                    harvest_crop_tasks
                    + harvest_animal_tasks
                    + care_tasks
                    + water_crop_tasks
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
                    nearest_shed = min(self.shed_tiles, key=lambda s: get_manhattan_dist(u_pos, s))
                    unit_actions.append([get_step_towards(u_pos, nearest_shed)])
                else:
                    unit_actions.append(["PASS"])

        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }

smart_agent = SmartMarketTimingGrandmasterAgent()

print("Testing SmartMarketTimingGrandmasterAgent on Seed 1000...")
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1000}, debug=True)
env.run([smart_agent, "starter"])
c0 = env.steps[-1][0]["reward"]
c1 = env.steps[-1][1]["reward"]
print(f"Seed 1000: SmartAgent=${c0:,.0f} vs Starter=${c1:,.0f}")
