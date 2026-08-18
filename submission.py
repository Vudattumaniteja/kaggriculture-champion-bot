"""
Flagship 150k Grandmaster Engine.
Autonomous World-Class Engine for Kaggriculture.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["COW", "SHEEP", "GOOSE"]

SHED_TILES: Set[Tuple[int, int]] = {(4, 4), (5, 4), (4, 5), (5, 5)}

# Dedicated Pasture locations (14 maximum, around the central shed)
PASTURE_LOCS: List[Tuple[int, int]] = [
    # NW (6)
    (4, 4), (3, 4), (4, 3), (3, 3), (2, 4), (4, 2),
    # NE (5)
    (5, 4), (6, 4), (6, 3), (5, 2), (7, 4),
    # SW (3)
    (4, 5), (3, 5), (2, 5),
]
PASTURE_SET: Set[Tuple[int, int]] = set(PASTURE_LOCS)

# Dedicated continuous NW Wheat feed tiles (7 tiles)
NW_WHEAT_TILES: Set[Tuple[int, int]] = {
    (0, 0), (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1),
}

def get_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def get_move(curr: Tuple[int, int], target: Tuple[int, int]) -> str:
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

class Task:
    __slots__ = ("task_type", "pos", "utility", "extra")
    def __init__(self, task_type: str, pos: Tuple[int, int], utility: float, extra: Any = None):
        self.task_type = task_type
        self.pos = pos
        self.utility = utility
        self.extra = extra

class Flagship150kEngine:
    def __init__(self, max_cows: int = 10, max_sheep: int = 4):
        self.max_cows = max_cows
        self.max_sheep = max_sheep

    def _get_target_hires(self, day: int) -> int:
        if day == 0:
            return 5
        elif day <= 6:
            return 5
        elif day <= 8:
            return 8
        elif day <= 28:
            return 12
        else:
            return 8

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        player = obs["player"]
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        private = obs.get("private", {}) or {}
        market = obs.get("market", {}) or {}

        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", 0))
        money = float(my_farm.get("money", 0.0))
        farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
        hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
        tiles = my_farm.get("tiles", [])
        unlocked_quads = list(my_farm.get("unlocked_quadrants", ["NW"]))
        hires_today = int(my_farm.get("hires_today", 0))

        shed = dict(private.get("shed", {}) or {})
        seeds = dict(private.get("seeds", {}) or {})
        inventories = [dict(inv) for inv in (private.get("inventories", []) or [])]
        market_prices = market.get("prices", {}) or {}
        shed_total = sum(shed.values())

        # ==========================================
        # 1. SCAN BOARD STRUCTURES, ANIMALS & CROPS
        # ==========================================
        living_animals: List[Tuple[int, int, str, Dict[str, Any]]] = []
        living_cows = 0
        living_sheep = 0
        empty_pastures: List[Tuple[int, int]] = []
        crop_tiles: List[Tuple[int, int, Dict[str, Any]]] = []
        empty_tiles: List[Tuple[int, int]] = []
        weed_tiles: List[Tuple[int, int]] = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                pos = (c, r)
                t = tiles[r][c]
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    animal = t.get("animal")
                    if k == "PASTURE":
                        if animal == "COW":
                            living_cows += 1
                            living_animals.append((c, r, "COW", t))
                        elif animal == "SHEEP":
                            living_sheep += 1
                            living_animals.append((c, r, "SHEEP", t))
                        elif animal is None:
                            empty_pastures.append(pos)
                    elif k == "PLANT":
                        crop_tiles.append((c, r, t))
                    elif k == "WEED":
                        weed_tiles.append(pos)

        total_living_animals = living_cows + living_sheep
        total_structures = len(living_animals) + len(empty_pastures)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)
        unplaced_shed_animals = cows_in_shed + sheep_in_shed

        cows_in_inv = sum(inv.get("COW", 0) for inv in inventories)
        sheep_in_inv = sum(inv.get("SHEEP", 0) for inv in inventories)
        fert_in_inv = sum(inv.get("FERTILIZER", 0) for inv in inventories)
        total_animals_owned = total_living_animals + unplaced_shed_animals + cows_in_inv + sheep_in_inv

        # ==========================================
        # 2. MARKET CONTROLLER (DYNAMIC DRIP ENGINE)
        # ==========================================
        hire_orders: List[List[Any]] = []
        feed_orders: List[List[Any]] = []
        sell_orders: List[List[Any]] = []
        land_orders: List[List[Any]] = []
        seed_orders: List[List[Any]] = []
        animal_orders: List[List[Any]] = []

        # A. Day 0 Turn 0 Initialization ($2,929 spend, $71 buffer)
        if day == 0 and step <= 1:
            for _ in range(5):
                hire_orders.append(["HIRE"])
            hires_today += 5
            money -= 24

            animal_orders.append(["BUY_ANIMAL", "COW", 2])
            money -= 800
            animal_orders.append(["BUY_ANIMAL", "SHEEP", 2])
            money -= 1000

            seed_orders.append(["BUY_SEED", "WHEAT", 7])
            money -= 70
            seed_orders.append(["BUY_SEED", "MELON", 12])
            money -= 960
            feed_orders.append(["BUY_PRODUCT", "WHEAT", 3])
            money -= 75

        elif day == 0 and step == 2 and shed.get("WHEAT", 0) >= 2:
            sell_orders.append(["SELL", "WHEAT", 2])
            money += 50
            shed["WHEAT"] -= 2

        else:
            # 1. Multi-Turn Daily Hiring (Hours 0..3)
            target_hires = self._get_target_hires(day)
            if hour <= 3 and hires_today < target_hires and money >= 2:
                hires_needed = target_hires - hires_today
                hire_chunk = min(hires_needed, 5)
                for _ in range(hire_chunk):
                    hire_orders.append(["HIRE"])
                hires_today += hire_chunk

            # 2. Critical Feed On-Demand (Whenever shed wheat < total_animals + 2)
            wheat_in_shed = shed.get("WHEAT", 0)
            if day <= 27 and total_living_animals > 0 and wheat_in_shed < max(4, total_living_animals + 2) and money >= 25:
                needed_feed = max(2, total_living_animals * 2 - wheat_in_shed)
                buy_feed = min(needed_feed, int(money // 25), 8)
                if buy_feed > 0:
                    feed_orders.append(["BUY_PRODUCT", "WHEAT", buy_feed])
                    money -= buy_feed * 25
                    shed["WHEAT"] = wheat_in_shed + buy_feed

            # 3. Market Sales (PRICE-SENSITIVE DRIP-FEED ENGINE)
            if day >= 28:
                # Endgame complete liquidation flush
                for prod in PRODUCTS:
                    cnt = shed.get(prod, 0)
                    if cnt > 0:
                        sell_orders.append(["SELL", prod, cnt])
            else:
                p_melon = market_prices.get("MELON", 250)
                p_milk = market_prices.get("MILK", 160)
                p_wool = market_prices.get("WOOL", 200)
                p_straw = market_prices.get("STRAWBERRY", 120)
                p_fert = market_prices.get("FERTILIZER", 95)
                p_wheat = market_prices.get("WHEAT", 25)

                near_capacity = (shed_total >= 85)

                # Melons: Always sell upon harvest
                melon_cnt = shed.get("MELON", 0)
                if melon_cnt > 0:
                    sell_m = min(melon_cnt, 4)
                    sell_orders.append(["SELL", "MELON", sell_m])
                    money += sell_m * p_melon
                    shed["MELON"] -= sell_m

                # Milk: Drip-feed if price >= 75 or shed near capacity
                milk_cnt = shed.get("MILK", 0)
                if milk_cnt > 0 and (p_milk >= 75 or near_capacity or money < 300):
                    sell_m = min(milk_cnt, 3)
                    sell_orders.append(["SELL", "MILK", sell_m])
                    money += sell_m * p_milk
                    shed["MILK"] -= sell_m

                # Wool: Drip-feed if price >= 95 or shed near capacity
                wool_cnt = shed.get("WOOL", 0)
                if wool_cnt > 0 and (p_wool >= 95 or near_capacity or money < 300):
                    sell_w = min(wool_cnt, 2)
                    sell_orders.append(["SELL", "WOOL", sell_w])
                    money += sell_w * p_wool
                    shed["WOOL"] -= sell_w

                # Strawberries: Drip-feed if price >= 60 or shed near capacity
                straw_cnt = shed.get("STRAWBERRY", 0)
                if straw_cnt > 0 and (p_straw >= 60 or near_capacity or money < 300):
                    sell_s = min(straw_cnt, 3)
                    sell_orders.append(["SELL", "STRAWBERRY", sell_s])
                    money += sell_s * p_straw
                    shed["STRAWBERRY"] -= sell_s

                # Wheat: Keep safe reserve for feeding, drip-feed excess
                wheat_cnt = shed.get("WHEAT", 0)
                safe_wheat = max(6, total_living_animals + 2)
                if wheat_cnt > safe_wheat + 2 and (p_wheat >= 15 or near_capacity):
                    sell_wheat = min(wheat_cnt - safe_wheat, 3)
                    sell_orders.append(["SELL", "WHEAT", sell_wheat])
                    money += sell_wheat * p_wheat
                    shed["WHEAT"] -= sell_wheat

                # Fertilizer: Keep 2 in shed, drip-feed 2-4 per turn
                fert_cnt = shed.get("FERTILIZER", 0)
                if fert_cnt > 2 and (p_fert >= 65 or near_capacity or money < 300):
                    sell_f = min(fert_cnt - 2, 3)
                    sell_orders.append(["SELL", "FERTILIZER", sell_f])
                    money += sell_f * p_fert
                    shed["FERTILIZER"] -= sell_f
                elif day <= 9 and fert_cnt > 0 and money < 250:
                    sell_f = min(fert_cnt, 2)
                    sell_orders.append(["SELL", "FERTILIZER", sell_f])
                    money += sell_f * p_fert
                    shed["FERTILIZER"] -= sell_f

                for prod in ["CARROT", "TOMATO", "EGG"]:
                    cnt = shed.get(prod, 0)
                    if cnt > 0:
                        sell_orders.append(["SELL", prod, min(cnt, 2)])
                        money += min(cnt, 2) * market_prices.get(prod, 35)
                        shed[prod] -= min(cnt, 2)

            # 4. Land Expansion (TOP PRIORITY over animal buying: NE @ Day 6, SW @ Day 11)
            if "NE" not in unlocked_quads and money >= 1050 and day >= 6:
                land_orders.append(["BUY_LAND"])
                money -= 1000
                unlocked_quads.append("NE")
            elif "SW" not in unlocked_quads and "NE" in unlocked_quads and money >= 2050 and day >= 11:
                land_orders.append(["BUY_LAND"])
                money -= 2000
                unlocked_quads.append("SW")

            # 5. Strawberry & Wheat Seed Purchasing (High priority on Day 6+)
            if day < 26:
                wheat_seeds = seeds.get("WHEAT", 0)
                wheat_tiles = sum(1 for _, _, ct in crop_tiles if ct.get("crop") == "WHEAT")
                target_wheat_seeds = 12 if day <= 24 else 25
                if (wheat_seeds + wheat_tiles) < target_wheat_seeds and money >= 40:
                    buy_w = min(target_wheat_seeds - (wheat_seeds + wheat_tiles), int((money - 30) // 10), 6)
                    if buy_w > 0:
                        seed_orders.append(["BUY_SEED", "WHEAT", buy_w])
                        money -= buy_w * 10
                        seeds["WHEAT"] = wheat_seeds + buy_w

                straw_seeds = seeds.get("STRAWBERRY", 0)
                straw_tiles = sum(1 for _, _, ct in crop_tiles if ct.get("crop") == "STRAWBERRY")

                if day <= 8:
                    target_strawberries = 12
                elif day <= 10:
                    target_strawberries = 20
                elif day <= 24:
                    target_strawberries = 38
                else:
                    target_strawberries = 20

                if "NE" in unlocked_quads and (straw_tiles + straw_seeds) < target_strawberries and money >= 110:
                    needed_s = target_strawberries - (straw_tiles + straw_seeds)
                    buy_s = min(needed_s, int((money - 40) // 100), 6)
                    if buy_s > 0:
                        seed_orders.append(["BUY_SEED", "STRAWBERRY", buy_s])
                        money -= buy_s * 100
                        seeds["STRAWBERRY"] = straw_seeds + buy_s

            # 6. Herd Scaling (10 Cows + 4 Sheep through Day 18)
            if day <= 18 and unplaced_shed_animals <= 2:
                target_herd_cows = 2
                target_herd_sheep = 2
                if day >= 3: target_herd_cows = 3
                if day >= 5: target_herd_cows = 4
                if day >= 7: target_herd_cows = 6; target_herd_sheep = 4
                if day >= 9: target_herd_cows = 8
                if day >= 11: target_herd_cows = 10; target_herd_sheep = 4

                rem_cows = max(0, min(self.max_cows, target_herd_cows) - (living_cows + cows_in_shed + cows_in_inv))
                rem_sheep = max(0, min(self.max_sheep, target_herd_sheep) - (living_sheep + sheep_in_shed + sheep_in_inv))

                min_buffer = 1100.0 if (day == 6 and "NE" not in unlocked_quads) else 250.0
                avail_animal_cash = max(0.0, money - min_buffer)

                if rem_sheep > 0 and avail_animal_cash >= 500:
                    animal_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                    money -= 500
                    sheep_in_shed += 1
                    avail_animal_cash -= 500
                elif rem_cows > 0 and avail_animal_cash >= 400:
                    buy_c = min(rem_cows, int(avail_animal_cash // 400), 2)
                    if buy_c > 0:
                        animal_orders.append(["BUY_ANIMAL", "COW", buy_c])
                        money -= buy_c * 400
                        cows_in_shed += buy_c
                        avail_animal_cash -= buy_c * 400

        # Assemble prioritized market queue (strictly max 10 orders)
        all_market_orders = hire_orders + feed_orders + sell_orders + land_orders + seed_orders + animal_orders
        final_market_orders = all_market_orders[:10]

        # ==========================================
        # 4. CHORE TASK GENERATION
        # ==========================================
        tasks: List[Task] = []

        # 1. Animal Critical Feed (TOP PRIORITY: UTILITY 1200)
        for ax, ay, aname, atile in living_animals:
            if not atile.get("fed_today", False):
                tasks.append(Task("FEED", (ax, ay), utility=1200.0, extra={"animal": aname}))

        # 2. Crop Critical Watering (TOP PRIORITY: UTILITY 1180)
        for cx, cy, ctile in crop_tiles:
            if not ctile.get("watered_today", False):
                tasks.append(Task("WATER_CROP", (cx, cy), utility=1180.0))

        # 3. Place Animals into Pastures (UTILITY 1100)
        if unplaced_shed_animals > 0 or cows_in_inv > 0 or sheep_in_inv > 0:
            for ep in empty_pastures:
                tasks.append(Task("PLACE_PASTURE_ANIMAL", ep, utility=1100.0))

        # 4. Build Pastures on Dedicated Locations (UTILITY 1050)
        pastures_needed = max(0, total_animals_owned - total_structures)
        if pastures_needed > 0 and len(empty_tiles) > 0:
            candidate_spots = [p for p in PASTURE_LOCS if p in empty_tiles]
            if not candidate_spots:
                candidate_spots = sorted([p for p in empty_tiles if p not in NW_WHEAT_TILES], key=lambda p: get_dist(p, (4, 4)))
            for p in candidate_spots[:pastures_needed]:
                tasks.append(Task("BUILD_PASTURE", p, utility=1050.0))

        # 5. Crop Harvesting (UTILITY 850-890)
        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            crop = ctile.get("crop", "WHEAT")
            age = day - ctile.get("planted_day", day)
            yield_units = ctile.get("yield_units", 0)

            if crop == "WHEAT":
                if age >= 4 or (day >= 28 and yield_units > 0):
                    tasks.append(Task("HARVEST_CROP", pos, utility=850.0))
            elif crop == "MELON":
                if age >= 10 or (day >= 28 and yield_units > 0):
                    tasks.append(Task("HARVEST_CROP", pos, utility=890.0))
            elif crop == "STRAWBERRY":
                if yield_units > 0:
                    tasks.append(Task("HARVEST_CROP", pos, utility=880.0))
            else:
                if yield_units > 0 or age >= 4:
                    tasks.append(Task("HARVEST_CROP", pos, utility=700.0))

        # 6. Fertilizer Application to High-Value Crops (Melons & Strawberries)
        total_fert_available = shed.get("FERTILIZER", 0) + fert_in_inv
        if total_fert_available > 0 and day <= 27:
            for cx, cy, ctile in crop_tiles:
                pos = (cx, cy)
                crop = ctile.get("crop", "")
                age = day - ctile.get("planted_day", day)
                fert_until = ctile.get("fertilized_until_day", -1)
                if fert_until < day:
                    if crop == "MELON" and age <= 10:
                        tasks.append(Task("FERTILIZE_CROP", pos, utility=860.0))
                    elif crop == "STRAWBERRY":
                        tasks.append(Task("FERTILIZE_CROP", pos, utility=855.0))

        # 7. Livestock Care, Manure & Harvest
        for ax, ay, aname, atile in living_animals:
            pos = (ax, ay)
            if not atile.get("cared_today", False):
                tasks.append(Task("CARE", pos, utility=850.0, extra={"animal": aname}))
            if atile.get("yield_units", 0) > 0:
                tasks.append(Task("HARVEST_ANIMAL", pos, utility=840.0, extra={"animal": aname}))
            if atile.get("fertilizer_available", False):
                tasks.append(Task("COLLECT_FERTILIZER", pos, utility=820.0, extra={"animal": aname}))

        # 8. Planting Chores on Empty Tiles
        wheat_seeds_avail = seeds.get("WHEAT", 0)
        nw_wheat_empty = [p for p in NW_WHEAT_TILES if p in empty_tiles and p not in PASTURE_SET]
        if day <= 26 and wheat_seeds_avail > 0 and nw_wheat_empty:
            for p in nw_wheat_empty[:wheat_seeds_avail]:
                tasks.append(Task("PLANT_CROP", p, utility=810.0, extra="WHEAT"))

        straw_seeds_avail = seeds.get("STRAWBERRY", 0)
        straw_empty_tiles = [p for p in empty_tiles if p not in NW_WHEAT_TILES and p not in PASTURE_SET]
        if day <= 26 and straw_seeds_avail > 0 and straw_empty_tiles:
            for p in straw_empty_tiles[:straw_seeds_avail]:
                tasks.append(Task("PLANT_CROP", p, utility=800.0, extra="STRAWBERRY"))

        # 9. Digging Weeds
        for wx, wy in weed_tiles:
            tasks.append(Task("DIG_WEED", (wx, wy), utility=350.0))

        # ==========================================
        # 5. INVENTORY-AWARE HUNGARIAN MATCHING
        # ==========================================
        all_workers: List[Tuple[int, Tuple[int, int], Dict[str, int]]] = []
        all_workers.append((0, farmer_pos, inventories[0] if len(inventories) > 0 else {}))
        for h_idx, h_pos in enumerate(hands_pos):
            w_idx = h_idx + 1
            w_inv = inventories[w_idx] if w_idx < len(inventories) else {}
            all_workers.append((w_idx, h_pos, w_inv))

        num_workers = len(all_workers)
        num_tasks = len(tasks)
        worker_actions: List[List[Any]] = [["PASS"] for _ in range(num_workers)]

        if num_workers > 0 and num_tasks > 0:
            cost_matrix = np.zeros((num_workers, num_tasks), dtype=np.float64)

            for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
                is_at_shed = w_pos in SHED_TILES
                has_cow = w_inv.get("COW", 0) > 0
                has_sheep = w_inv.get("SHEEP", 0) > 0
                has_animal = has_cow or has_sheep
                has_wheat = w_inv.get("WHEAT", 0) > 0
                has_fert = w_inv.get("FERTILIZER", 0) > 0

                for j, task in enumerate(tasks):
                    t_pos = task.pos
                    raw_dist = get_dist(w_pos, t_pos)
                    effective_dist = float(raw_dist)

                    if has_animal:
                        if task.task_type == "PLACE_PASTURE_ANIMAL":
                            effective_dist = float(raw_dist) - 50.0
                        else:
                            cost_matrix[i, j] = 99999.0
                            continue

                    elif task.task_type == "PLACE_PASTURE_ANIMAL":
                        if not is_at_shed:
                            dist_to_shed = min(get_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 2.0

                    elif task.task_type == "FEED":
                        if has_wheat:
                            effective_dist = float(raw_dist) - 2.0
                        elif not is_at_shed:
                            dist_to_shed = min(get_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    elif task.task_type == "FERTILIZE_CROP":
                        if has_fert:
                            effective_dist = float(raw_dist) - 2.0
                        elif not is_at_shed:
                            dist_to_shed = min(get_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    cost_matrix[i, j] = (effective_dist * 4.0) - (task.utility * 0.1)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_task_map = {row: col for row, col in zip(row_ind, col_ind)}
        else:
            assigned_task_map = {}

        # ==========================================
        # 6. ACTION GENERATION PER WORKER
        # ==========================================
        for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
            wx, wy = w_pos
            w_tile = tiles[wy][wx] if wy < len(tiles) and wx < len(tiles[wy]) else None
            is_at_shed = w_pos in SHED_TILES
            assigned_task = tasks[assigned_task_map[i]] if i in assigned_task_map else None

            # 1. Shed Operations (Pickup / Drop)
            if is_at_shed:
                # Drop produce or unwanted items
                has_produce = any(w_inv.get(p, 0) > 0 for p in ["STRAWBERRY", "MELON", "MILK", "WOOL", "EGG"])
                has_unwanted_wheat = (w_inv.get("WHEAT", 0) > 0 and (not assigned_task or assigned_task.task_type != "FEED"))
                has_unwanted_fert = (w_inv.get("FERTILIZER", 0) > 0 and (not assigned_task or assigned_task.task_type != "FERTILIZE_CROP"))

                if has_produce or has_unwanted_wheat or has_unwanted_fert:
                    worker_actions[i] = ["DROP"]
                    continue

                # Pickup item ONLY if specifically assigned to that task
                if assigned_task:
                    if assigned_task.task_type == "PLACE_PASTURE_ANIMAL" and w_inv.get("SHEEP", 0) == 0 and w_inv.get("COW", 0) == 0:
                        if shed.get("SHEEP", 0) > 0:
                            worker_actions[i] = ["PICKUP", "SHEEP", 1]
                            continue
                        elif shed.get("COW", 0) > 0:
                            worker_actions[i] = ["PICKUP", "COW", 1]
                            continue

                    elif assigned_task.task_type == "FEED" and w_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                        worker_actions[i] = ["PICKUP", "WHEAT", min(4, shed.get("WHEAT", 0))]
                        continue

                    elif assigned_task.task_type == "FERTILIZE_CROP" and w_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                        worker_actions[i] = ["PICKUP", "FERTILIZER", min(4, shed.get("FERTILIZER", 0))]
                        continue

            # 2. Standing Tile Operations
            if isinstance(w_tile, dict):
                k = w_tile.get("kind")
                animal = w_tile.get("animal")

                if k in ["PASTURE", "COOP"]:
                    if animal is not None:
                        if not w_tile.get("fed_today", False) and w_inv.get("WHEAT", 0) > 0:
                            worker_actions[i] = ["FEED"]
                            continue
                        if not w_tile.get("cared_today", False):
                            worker_actions[i] = ["CARE"]
                            continue
                        if w_tile.get("fertilizer_available", False):
                            worker_actions[i] = ["COLLECT_FERTILIZER"]
                            continue
                        if w_tile.get("yield_units", 0) > 0:
                            worker_actions[i] = ["HARVEST"]
                            continue
                    else:
                        if assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "PLACE_PASTURE_ANIMAL":
                            if w_inv.get("SHEEP", 0) > 0:
                                worker_actions[i] = ["PLACE", "SHEEP"]
                                continue
                            elif w_inv.get("COW", 0) > 0:
                                worker_actions[i] = ["PLACE", "COW"]
                                continue

                elif k == "PLANT":
                    crop = w_tile.get("crop", "WHEAT")
                    age = day - w_tile.get("planted_day", day)
                    yield_u = w_tile.get("yield_units", 0)
                    fert_until = w_tile.get("fertilized_until_day", -1)

                    # 1. WATER FIRST (CRITICAL LIFE PROTECTION)
                    if not w_tile.get("watered_today", False):
                        worker_actions[i] = ["WATER"]
                        continue

                    # 2. FERTILIZE (Instant on-tile fertilization if unfertilized and worker has fertilizer)
                    if fert_until < day and w_inv.get("FERTILIZER", 0) > 0:
                        if crop == "MELON" and age <= 10:
                            worker_actions[i] = ["FERTILIZE"]
                            continue
                        elif crop == "STRAWBERRY" and day <= 27:
                            worker_actions[i] = ["FERTILIZE"]
                            continue

                    # 3. HARVEST
                    if crop == "WHEAT":
                        if age >= 4 or (day >= 28 and yield_u > 0):
                            worker_actions[i] = ["HARVEST"]
                            continue
                    elif crop == "MELON":
                        if age >= 10 or (day >= 28 and yield_u > 0):
                            worker_actions[i] = ["HARVEST"]
                            continue
                    elif crop == "STRAWBERRY":
                        if yield_u > 0:
                            worker_actions[i] = ["HARVEST"]
                            continue
                    else:
                        if yield_u > 0 or age >= 4:
                            worker_actions[i] = ["HARVEST"]
                            continue

                elif k == "WEED":
                    worker_actions[i] = ["DIG"]
                    continue

            elif w_tile is None:
                if assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "BUILD_PASTURE":
                    worker_actions[i] = ["BUILD_PASTURE"]
                    continue

                if (assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "PLANT_CROP") or (assigned_task is None and day <= 26):
                    # Never plant crops on dedicated pasture locations
                    if w_pos not in PASTURE_SET:
                        if day == 0:
                            if seeds.get("MELON", 0) > 0:
                                worker_actions[i] = ["PLANT", "MELON"]
                                seeds["MELON"] -= 1
                                continue
                            elif seeds.get("WHEAT", 0) > 0:
                                worker_actions[i] = ["PLANT", "WHEAT"]
                                seeds["WHEAT"] -= 1
                                continue
                        else:
                            if w_pos in NW_WHEAT_TILES:
                                if seeds.get("WHEAT", 0) > 0:
                                    worker_actions[i] = ["PLANT", "WHEAT"]
                                    seeds["WHEAT"] -= 1
                                    continue
                            else:
                                if seeds.get("STRAWBERRY", 0) > 0:
                                    worker_actions[i] = ["PLANT", "STRAWBERRY"]
                                    seeds["STRAWBERRY"] -= 1
                                    continue
                                elif seeds.get("WHEAT", 0) > 0:
                                    worker_actions[i] = ["PLANT", "WHEAT"]
                                    seeds["WHEAT"] -= 1
                                    continue

            # 3. Pathfinding to Target
            if assigned_task:
                target_pos = assigned_task.pos

                needs_shed = False
                if assigned_task.task_type == "FEED" and w_inv.get("WHEAT", 0) == 0:
                    needs_shed = True
                elif assigned_task.task_type == "FERTILIZE_CROP" and w_inv.get("FERTILIZER", 0) == 0:
                    needs_shed = True
                elif assigned_task.task_type == "PLACE_PASTURE_ANIMAL" and (w_inv.get("COW", 0) == 0 and w_inv.get("SHEEP", 0) == 0):
                    needs_shed = True

                if needs_shed and not is_at_shed:
                    nearest_shed = min(SHED_TILES, key=lambda s: get_dist(w_pos, s))
                    worker_actions[i] = [get_move(w_pos, nearest_shed)]
                else:
                    worker_actions[i] = [get_move(w_pos, target_pos)]
            else:
                if not is_at_shed:
                    nearest_shed = min(SHED_TILES, key=lambda s: get_dist(w_pos, s))
                    worker_actions[i] = [get_move(w_pos, nearest_shed)]
                else:
                    worker_actions[i] = ["PASS"]

        farmer_act = worker_actions[0] if worker_actions else ["PASS"]
        hands_act = worker_actions[1:] if len(worker_actions) > 1 else []

        return {
            "farmer": farmer_act,
            "hands": hands_act,
            "market": final_market_orders,
        }

# Global singleton
_CHAMPION_AGENT = Flagship150kEngine()

def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _CHAMPION_AGENT(obs, config)
