"""
Kaggriculture AI Competition: Pure Deterministic Grandmaster Bot.
100% Self-Contained, Standalone Single-File Deterministic Agent.
Combines Town Shop Arbitrage, Livestock Husbandry, Dynamic Labor Scaling, and Zero-Deadweight Liquidation.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

# Domain Constants & Catalogs
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

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

SHOPS_CATALOG = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]


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


class DeterministicGrandmasterFarmer:
    def __init__(self):
        self.target_geese = 2
        self.target_cows = 1

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        player = obs["player"]
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        opp_farm = farms[1 - player] if (1 - player) < len(farms) else {}
        private = obs.get("private", {}) or {}
        market = obs.get("market", {}) or {}
        town = obs.get("town", {}) or {}

        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", 0))
        money = float(my_farm.get("money", 0.0))
        farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
        hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
        tiles = my_farm.get("tiles", [])
        unlocked = my_farm.get("unlocked_quadrants", ["NW"])
        hires_today = int(my_farm.get("hires_today", 0))
        shed = private.get("shed", {}) or {}
        seeds = private.get("seeds", {}) or {}
        inventories = private.get("inventories", []) or []
        market_prices = market.get("prices", {}) or {}
        unlocked_shops = town.get("unlocked_shops", []) or []

        market_orders: List[List[Any]] = []

        # 1. Audit structures & crops
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

        # 2. Market Orders & Arbitrage
        for item in ["EGG", "MILK", "WOOL", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "FERTILIZER"]:
            count = shed.get(item, 0)
            if count > 0:
                if item == "FERTILIZER" and len(crop_tiles) > 0 and day <= 22 and count > 3:
                    market_orders.append(["SELL", item, count - 3])
                elif item != "FERTILIZER":
                    market_orders.append(["SELL", item, count])

        # Wheat reserve management
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_req = total_living_animals
        wheat_safe_res = max(4, daily_feed_req * 3)
        if wheat_in_shed > wheat_safe_res + 8:
            market_orders.append(["SELL", "WHEAT", wheat_in_shed - wheat_safe_res])
        elif wheat_in_shed < daily_feed_req and money >= 60 and day < 28:
            qty = min(6, wheat_safe_res - wheat_in_shed)
            market_orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * 25

        # Land Expansion
        if day <= 16:
            if "NE" not in unlocked and money >= 1200:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and money >= 2400 and day <= 10:
                market_orders.append(["BUY_LAND"])
                money -= 2000

        # Labor Hiring
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 40:
            market_orders.append(["HIRE"])
            money -= 2

        # Animal Purchasing
        if day <= 18:
            if empty_coops > 0 and living_geese < self.target_geese and money >= 400:
                market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
                money -= 300
            elif empty_pastures > 0 and living_cows < self.target_cows and money >= 550 and "PIZZA_SHOP" in unlocked_shops:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400

        # Dynamic Seed Purchasing based on Town Shops
        if day < 27 and money >= 80:
            seed_orders = []
            if day <= 5 and seeds.get("MELON", 0) + planted_crops.get("MELON", 0) < 6 and money >= 400:
                qty = min(3, 6 - (seeds.get("MELON", 0) + planted_crops.get("MELON", 0)))
                seed_orders.append(["MELON", qty])
                money -= qty * 80

            if "BAKERY" in unlocked_shops or total_living_animals > 0:
                if seeds.get("WHEAT", 0) + planted_crops.get("WHEAT", 0) < 8 and money >= 100:
                    qty = min(4, 8 - (seeds.get("WHEAT", 0) + planted_crops.get("WHEAT", 0)))
                    seed_orders.append(["WHEAT", qty])
                    money -= qty * 10

            if "PET_CAFE" in unlocked_shops or day > 10:
                if seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0) < 14 and money >= 160:
                    qty = min(6, 14 - (seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0)))
                    seed_orders.append(["CARROT", qty])
                    money -= qty * 20

            for crop_name, qty in seed_orders:
                if qty > 0:
                    market_orders.append(["BUY_SEED", crop_name, qty])

        # 3. Unit Task Dispatch
        units = [farmer_pos] + hands_pos
        unit_actions: List[List[Any]] = []
        claimed_tasks: Set[Tuple[int, int]] = set()

        feed_tasks: List[Tuple[int, int]] = []
        care_tasks: List[Tuple[int, int]] = []
        harvest_animal_tasks: List[Tuple[int, int]] = []
        fertilizer_tasks: List[Tuple[int, int]] = []

        harvest_crop_tasks: List[Tuple[int, int]] = []
        water_crop_tasks: List[Tuple[int, int]] = []
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
                    elif t.get("kind") == "COOP" and t.get("animal") is None:
                        if shed.get("GOOSE", 0) > 0:
                            place_animal_tasks.append((c, r, "GOOSE"))
                    elif t.get("kind") == "PASTURE" and t.get("animal") is None:
                        if shed.get("COW", 0) > 0:
                            place_animal_tasks.append((c, r, "COW"))
                elif t is None and pos not in SHED_TILES:
                    if day <= 10 and len(animal_tiles) + empty_coops < self.target_geese and len(build_coop_tasks) == 0:
                        build_coop_tasks.append(pos)
                    elif day <= 12 and len(animal_tiles) + empty_pastures < (self.target_geese + self.target_cows) and len(build_pasture_tasks) == 0 and "PIZZA_SHOP" in unlocked_shops:
                        build_pasture_tasks.append(pos)
                    elif day < 27:
                        plant_tasks.append(pos)

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in SHED_TILES

            # 1. Animal placement
            animal_placed = False
            if isinstance(u_tile, dict) and u_tile.get("kind") in ["COOP", "PASTURE"] and u_tile.get("animal") is None:
                needed_an = "GOOSE" if u_tile.get("kind") == "COOP" else "COW"
                if u_inv.get(needed_an, 0) > 0:
                    unit_actions.append(["PLACE_ANIMAL", needed_an])
                    u_inv[needed_an] -= 1
                    animal_placed = True
            if animal_placed:
                continue

            # 2. Shed pickups
            if is_at_shed:
                if shed.get("GOOSE", 0) > 0 and empty_coops > 0 and u_inv.get("GOOSE", 0) == 0:
                    unit_actions.append(["PICKUP", "GOOSE", 1])
                    shed["GOOSE"] -= 1
                    u_inv["GOOSE"] = 1
                    continue
                if shed.get("COW", 0) > 0 and empty_pastures > 0 and u_inv.get("COW", 0) == 0:
                    unit_actions.append(["PICKUP", "COW", 1])
                    shed["COW"] -= 1
                    u_inv["COW"] = 1
                    continue
                if fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    unit_actions.append(["PICKUP", "FERTILIZER", qty])
                    shed["FERTILIZER"] -= qty
                    u_inv["FERTILIZER"] = qty
                    continue

            # 3. Actions on standing tile
            if isinstance(u_tile, dict):
                k = u_tile.get("kind")
                if k in ["COOP", "PASTURE"] and u_tile.get("animal"):
                    if not u_tile.get("fed_today", False):
                        unit_actions.append(["FEED"])
                        u_tile["fed_today"] = True
                        continue
                    if not u_tile.get("cared_today", False):
                        unit_actions.append(["CARE"])
                        u_tile["cared_today"] = True
                        continue
                    if u_tile.get("yield_units", 0) > 0:
                        unit_actions.append(["HARVEST"])
                        u_tile["yield_units"] = 0
                        continue
                    if u_tile.get("fertilizer_available", False):
                        unit_actions.append(["COLLECT_FERTILIZER"])
                        u_tile["fertilizer_available"] = False
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
                        u_tile["watered_today"] = True
                        continue
                    if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0:
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue
                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue
            elif u_tile is None and u_pos not in SHED_TILES:
                if u_pos in build_coop_tasks and day <= 10:
                    unit_actions.append(["BUILD_COOP"])
                    build_coop_tasks.remove(u_pos)
                    continue
                if u_pos in build_pasture_tasks and day <= 12:
                    unit_actions.append(["BUILD_PASTURE"])
                    build_pasture_tasks.remove(u_pos)
                    continue
                if day < 27 and u_pos in plant_tasks:
                    best_crop = "CARROT"
                    if seeds.get("MELON", 0) > 0 and day <= 10:
                        best_crop = "MELON"
                    elif seeds.get("WHEAT", 0) > 0 and (total_living_animals > 0 or "BAKERY" in unlocked_shops):
                        best_crop = "WHEAT"
                    elif seeds.get("CARROT", 0) > 0:
                        best_crop = "CARROT"
                    elif any(seeds.get(c, 0) > 0 for c in CROPS):
                        best_crop = max(CROPS, key=lambda c: seeds.get(c, 0))

                    if seeds.get(best_crop, 0) > 0:
                        unit_actions.append(["PLANT", best_crop])
                        seeds[best_crop] -= 1
                        plant_tasks.remove(u_pos)
                        continue

            # 4. Pathfind to nearest priority chore
            priority_queue = (
                care_tasks + feed_tasks + harvest_animal_tasks + harvest_crop_tasks
                + water_crop_tasks + fertilizer_tasks + fertilize_crop_tasks
                + build_coop_tasks + build_pasture_tasks + plant_tasks + weed_tasks
            )

            best_target = None
            best_dist = 999
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


# Agent Entrypoint
_deterministic_instance = DeterministicGrandmasterFarmer()


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _deterministic_instance(obs, config)
