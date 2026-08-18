import os, sys
sys.path.insert(0, os.path.abspath("."))
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from kaggle_environments import make

# Game Constants
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]
SHED_TILES: List[Tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]

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

class ChoreTask:
    __slots__ = ("task_type", "pos", "utility", "extra")
    def __init__(self, task_type: str, pos: Tuple[int, int], utility: float, extra: Any = None):
        self.task_type = task_type
        self.pos = pos
        self.utility = utility
        self.extra = extra

class EnhancedHRL12WorkerDispatcher:
    def __init__(self, max_cows: int = 38, max_sheep: int = 38):
        self.max_cows = max_cows
        self.max_sheep = max_sheep

    def _get_target_hires(self, day: int) -> int:
        if day <= 6:
            return 5
        elif day <= 8:
            return 8
        elif day <= 24:
            return 12
        elif day <= 27:
            return 8
        elif day == 28:
            return 5
        else:
            return 0  # Day 29: zero hiring to maximize final cash balance

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        player = obs.get("player", 0)
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        private = obs.get("private", {}) or {}
        market = obs.get("market", {}) or {}

        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = day * 24 + hour
        money = float(my_farm.get("money", 0.0))
        farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
        hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
        tiles = my_farm.get("tiles", [])
        unlocked_quads = list(my_farm.get("unlocked_quadrants", ["NW"]))
        hires_today = int(my_farm.get("hires_today", 0))

        shed = private.get("shed", {}) or {}
        inventories = private.get("inventories", []) or []

        market_orders: List[List[Any]] = []

        # 1. SCAN FARM STRUCTURES, ANIMALS & TILES
        living_animals: List[Tuple[int, int, str, Dict[str, Any]]] = []
        living_cows = 0
        living_sheep = 0
        living_geese = 0
        empty_pastures: List[Tuple[int, int]] = []
        crop_tiles: List[Tuple[int, int, Dict[str, Any]]] = []
        empty_unlocked_tiles: List[Tuple[int, int]] = []
        weed_tiles: List[Tuple[int, int]] = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                pos = (c, r)
                t = tiles[r][c]
                if t is None:
                    if pos not in SHED_TILES:
                        empty_unlocked_tiles.append(pos)
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
                    elif k == "COOP":
                        if animal == "GOOSE":
                            living_geese += 1
                            living_animals.append((c, r, "GOOSE", t))
                    elif k == "PLANT":
                        crop_tiles.append((c, r, t))
                    elif k == "WEED":
                        weed_tiles.append(pos)

        total_living_animals = living_cows + living_sheep + living_geese
        total_structures = len(living_animals) + len(empty_pastures)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)
        unplaced_shed_animals = cows_in_shed + sheep_in_shed

        # 2. TOP-LEVEL STRATEGIC MACRO CONTROLLER

        # A. Priority 1: Immediate Market Selling (Produce, Wool, Milk, Fertilizer)
        for prod in ["WOOL", "MILK", "EGG", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "FERTILIZER"]:
            count = shed.get(prod, 0)
            if count > 0:
                market_orders.append(["SELL", prod, count])

        # B. Priority 2: Priority Land Expansion
        if "NE" not in unlocked_quads and money >= 1000 and day >= 6:
            market_orders.append(["BUY_LAND"])
            money -= 1000
            unlocked_quads.append("NE")
        if "SW" not in unlocked_quads and "NE" in unlocked_quads and money >= 2000 and day >= 7:
            market_orders.append(["BUY_LAND"])
            money -= 2000
            unlocked_quads.append("SW")
        if "SE" not in unlocked_quads and "SW" in unlocked_quads and money >= 4000 and day >= 9:
            market_orders.append(["BUY_LAND"])
            money -= 4000
            unlocked_quads.append("SE")

        # C. Priority 3: Lean Wheat Feed Management
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_needed = max(1, total_living_animals)
        if day >= 28:
            safe_wheat_reserve = 0
        elif day >= 27:
            safe_wheat_reserve = daily_feed_needed
        else:
            safe_wheat_reserve = max(6, int(daily_feed_needed * 1.1) + 2)

        if hour == 0:
            if wheat_in_shed > safe_wheat_reserve + 10 and day < 28:
                excess_wheat = wheat_in_shed - safe_wheat_reserve
                market_orders.append(["SELL", "WHEAT", excess_wheat])
            elif wheat_in_shed < daily_feed_needed and money >= 25 and day < 28:
                buy_wheat_qty = max(4, safe_wheat_reserve - wheat_in_shed)
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_wheat_qty])
                money -= buy_wheat_qty * 25
        elif day >= 28 and wheat_in_shed > 0:
            # Sell off remaining wheat in final days
            market_orders.append(["SELL", "WHEAT", wheat_in_shed])

        # D. Priority 4: Full-Scale Livestock Purchasing
        land_savings_required = 0.0
        if "NE" not in unlocked_quads and day >= 5:
            land_savings_required = 1000.0
        elif "SW" not in unlocked_quads and "NE" in unlocked_quads and day >= 6:
            land_savings_required = 2000.0
        elif "SE" not in unlocked_quads and "SW" in unlocked_quads and day >= 8:
            land_savings_required = 4000.0

        # CRITICAL CASH BUFFER: Guarantee daily feed ($25/animal) + safety buffer
        required_feed_cash = max(100.0, (total_living_animals + unplaced_shed_animals + 2) * 25.0)
        safe_cash_buffer = required_feed_cash + 150.0
        available_money_for_animals = max(0.0, money - land_savings_required - safe_cash_buffer)

        # Turn 1 Initial Reinvestment
        if day == 0 and hour == 0:
            if money >= 1500:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 3])
                money -= 1500
            if money >= 800:
                market_orders.append(["BUY_ANIMAL", "COW", 2])
                money -= 800
            if money >= 200:
                market_orders.append(["BUY_PRODUCT", "WHEAT", 8])
                money -= 200
        elif day <= 18 and hour <= 3:
            if unplaced_shed_animals <= 6 and available_money_for_animals >= 400:
                empty_or_buildable = len(empty_pastures) + len(empty_unlocked_tiles)
                remaining_sheep_capacity = max(0, self.max_sheep - (living_sheep + sheep_in_shed))
                remaining_cow_capacity = max(0, self.max_cows - (living_cows + cows_in_shed))

                if remaining_sheep_capacity > 0 and available_money_for_animals >= 500:
                    qty = min(12, int(available_money_for_animals // 500), empty_or_buildable, remaining_sheep_capacity)
                    if qty > 0:
                        market_orders.append(["BUY_ANIMAL", "SHEEP", qty])
                        money -= qty * 500
                        sheep_in_shed += qty
                elif remaining_cow_capacity > 0 and available_money_for_animals >= 400:
                    qty = min(12, int(available_money_for_animals // 400), empty_or_buildable, remaining_cow_capacity)
                    if qty > 0:
                        market_orders.append(["BUY_ANIMAL", "COW", qty])
                        money -= qty * 400
                        cows_in_shed += qty

        # E. Priority 5: Daily Labor Hiring
        target_hires = self._get_target_hires(day)
        if hour <= 1 and hires_today < target_hires and money >= 10:
            hires_needed = target_hires - hires_today
            for _ in range(hires_needed):
                market_orders.append(["HIRE"])
            hires_today = target_hires

        # 3. CHORE SCAN & TASK POOL GENERATION
        tasks: List[ChoreTask] = []

        # Livestock Chores
        for ax, ay, aname, atile in living_animals:
            pos = (ax, ay)
            if not atile.get("fed_today", False):
                tasks.append(ChoreTask("FEED", pos, utility=1000.0, extra={"animal": aname}))
            if not atile.get("cared_today", False):
                tasks.append(ChoreTask("CARE", pos, utility=950.0, extra={"animal": aname}))
            if atile.get("fertilizer_available", False):
                tasks.append(ChoreTask("COLLECT_FERTILIZER", pos, utility=900.0, extra={"animal": aname}))
            if atile.get("yield_units", 0) > 0:
                tasks.append(ChoreTask("HARVEST_ANIMAL", pos, utility=850.0, extra={"animal": aname, "yield": atile.get("yield_units", 0)}))

        # Place Animals into Empty Pastures
        for ep in empty_pastures:
            tasks.append(ChoreTask("PLACE_PASTURE_ANIMAL", ep, utility=920.0))

        # Building Pastures (Farmer Only)
        if day <= 23 and total_structures < (self.max_cows + self.max_sheep):
            sorted_empty = sorted(empty_unlocked_tiles, key=lambda p: get_manhattan_dist(p, (4, 4)))
            num_pastures_to_build = min(len(sorted_empty), (self.max_cows + self.max_sheep) - total_structures)
            build_utility = 980.0 if (day <= 14 and len(empty_pastures) < 20) else 800.0
            for p in sorted_empty[:num_pastures_to_build]:
                tasks.append(ChoreTask("BUILD_PASTURE", p, utility=build_utility))

        # Crop Chores
        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            age = day - ctile.get("planted_day", day)
            yield_units = ctile.get("yield_units", 0)
            if age >= 4 or (day >= 28 and yield_units > 0):
                tasks.append(ChoreTask("HARVEST_CROP", pos, utility=600.0))
            elif not ctile.get("watered_today", False):
                tasks.append(ChoreTask("WATER_CROP", pos, utility=550.0))
            elif ctile.get("fertilized_until_day", -1) < day:
                tasks.append(ChoreTask("FERTILIZE_CROP", pos, utility=400.0))

        for wx, wy in weed_tiles:
            tasks.append(ChoreTask("DIG_WEED", (wx, wy), utility=300.0))

        # 4. HUNGARIAN CHORE ENGINE
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
                for j, task in enumerate(tasks):
                    t_pos = task.pos
                    raw_dist = get_manhattan_dist(w_pos, t_pos)
                    effective_dist = float(raw_dist)

                    # Rules compliance: Farmhands cannot execute BUILD_PASTURE
                    if w_idx > 0 and task.task_type == "BUILD_PASTURE":
                        cost_matrix[i, j] = 99999.0
                        continue

                    if task.task_type == "FEED":
                        has_wheat = w_inv.get("WHEAT", 0) > 0
                        if not has_wheat:
                            if not is_at_shed:
                                dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                                dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                                effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    elif task.task_type == "PLACE_PASTURE_ANIMAL":
                        has_animal = (w_inv.get("COW", 0) > 0 or w_inv.get("SHEEP", 0) > 0)
                        if not has_animal:
                            if not is_at_shed:
                                dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                                dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                                effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    elif task.task_type == "FERTILIZE_CROP":
                        has_fert = w_inv.get("FERTILIZER", 0) > 0
                        if not has_fert:
                            if not is_at_shed:
                                dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                                dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                                effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    cost_matrix[i, j] = effective_dist - (task.utility * 0.1)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_task_map = {row: col for row, col in zip(row_ind, col_ind)}
        else:
            assigned_task_map = {}

        # 5. ACTION GENERATION PER WORKER
        for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
            wx, wy = w_pos
            w_tile = tiles[wy][wx] if wy < len(tiles) and wx < len(tiles[wy]) else None
            is_at_shed = w_pos in SHED_TILES

            # Shed Operations
            if is_at_shed:
                # Immediate DROP of produce on endgame days to guarantee market liquidation
                if day >= 27 and any(w_inv.get(prod, 0) > 0 for prod in ["WOOL", "MILK", "FERTILIZER", "EGG", "WHEAT"]):
                    worker_actions[i] = ["DROP"]
                    for k in list(w_inv.keys()):
                        del w_inv[k]
                    continue

                # Streamlined Animal Pickup
                can_expand = (empty_unlocked_tiles and total_structures < (self.max_cows + self.max_sheep))
                if (empty_pastures or can_expand) and w_inv.get("SHEEP", 0) == 0 and w_inv.get("COW", 0) == 0:
                    if shed.get("SHEEP", 0) > 0:
                        worker_actions[i] = ["PICKUP", "SHEEP", 1]
                        shed["SHEEP"] -= 1
                        w_inv["SHEEP"] = 1
                        continue
                    elif shed.get("COW", 0) > 0:
                        worker_actions[i] = ["PICKUP", "COW", 1]
                        shed["COW"] -= 1
                        w_inv["COW"] = 1
                        continue

                # Pickup Wheat if unfed animals exist
                unfed_count = sum(1 for _, _, _, at in living_animals if not at.get("fed_today", False))
                if unfed_count > 0 and w_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0 and day < 29:
                    take_qty = min(6, shed.get("WHEAT", 0), unfed_count)
                    if take_qty > 0:
                        worker_actions[i] = ["PICKUP", "WHEAT", take_qty]
                        shed["WHEAT"] -= take_qty
                        w_inv["WHEAT"] = w_inv.get("WHEAT", 0) + take_qty
                        continue

                # Pickup Fertilizer if crop fertilizing task is assigned
                if any(t.task_type == "FERTILIZE_CROP" for t in tasks) and w_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    take_fert = min(2, shed.get("FERTILIZER", 0))
                    if take_fert > 0:
                        worker_actions[i] = ["PICKUP", "FERTILIZER", take_fert]
                        shed["FERTILIZER"] -= take_fert
                        w_inv["FERTILIZER"] = w_inv.get("FERTILIZER", 0) + take_fert
                        continue

            # Standing Tile Operations
            if isinstance(w_tile, dict):
                k = w_tile.get("kind")
                animal = w_tile.get("animal")

                if k in ["PASTURE", "COOP"]:
                    if animal is not None:
                        # Feed
                        if not w_tile.get("fed_today", False) and w_inv.get("WHEAT", 0) > 0:
                            worker_actions[i] = ["FEED"]
                            w_inv["WHEAT"] -= 1
                            w_tile["fed_today"] = True
                            continue
                        # Care
                        if not w_tile.get("cared_today", False):
                            worker_actions[i] = ["CARE"]
                            w_tile["cared_today"] = True
                            continue
                        # Collect Fertilizer
                        if w_tile.get("fertilizer_available", False):
                            worker_actions[i] = ["COLLECT_FERTILIZER"]
                            w_tile["fertilizer_available"] = False
                            w_inv["FERTILIZER"] = w_inv.get("FERTILIZER", 0) + 1
                            continue
                        # Harvest Produce
                        if w_tile.get("yield_units", 0) > 0:
                            worker_actions[i] = ["HARVEST"]
                            w_tile["yield_units"] = 0
                            continue
                    else:
                        # Empty Pasture -> Place Animal Immediately
                        if k == "PASTURE":
                            if w_inv.get("SHEEP", 0) > 0:
                                worker_actions[i] = ["PLACE", "SHEEP"]
                                w_inv["SHEEP"] -= 1
                                continue
                            elif w_inv.get("COW", 0) > 0:
                                worker_actions[i] = ["PLACE", "COW"]
                                w_inv["COW"] -= 1
                                continue

                elif k == "PLANT":
                    age = day - w_tile.get("planted_day", day)
                    yield_u = w_tile.get("yield_units", 0)
                    if age >= 4 or (day >= 28 and yield_u > 0):
                        worker_actions[i] = ["HARVEST"]
                        continue
                    if not w_tile.get("watered_today", False):
                        worker_actions[i] = ["WATER"]
                        w_tile["watered_today"] = True
                        continue
                    if w_tile.get("fertilized_until_day", -1) < day and w_inv.get("FERTILIZER", 0) > 0:
                        worker_actions[i] = ["FERTILIZE"]
                        w_inv["FERTILIZER"] -= 1
                        continue

                elif k == "WEED":
                    worker_actions[i] = ["DIG"]
                    continue

            elif w_tile is None and w_pos not in SHED_TILES:
                # ONLY Worker 0 (Farmer) can execute BUILD_PASTURE
                if w_idx == 0 and day <= 23 and total_structures < (self.max_cows + self.max_sheep):
                    worker_actions[i] = ["BUILD_PASTURE"]
                    continue

            # Pathfinding to Assigned Task Target
            if i in assigned_task_map:
                task = tasks[assigned_task_map[i]]
                target_pos = task.pos

                needs_shed = False
                if task.task_type == "FEED" and w_inv.get("WHEAT", 0) == 0:
                    needs_shed = True
                elif task.task_type == "PLACE_PASTURE_ANIMAL" and (w_inv.get("COW", 0) == 0 and w_inv.get("SHEEP", 0) == 0):
                    needs_shed = True
                elif task.task_type == "FERTILIZE_CROP" and w_inv.get("FERTILIZER", 0) == 0:
                    needs_shed = True

                if needs_shed and not is_at_shed:
                    nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(w_pos, s))
                    worker_actions[i] = [get_step_towards(w_pos, nearest_shed)]
                else:
                    worker_actions[i] = [get_step_towards(w_pos, target_pos)]
            else:
                if not is_at_shed:
                    nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(w_pos, s))
                    worker_actions[i] = [get_step_towards(w_pos, nearest_shed)]
                else:
                    worker_actions[i] = ["PASS"]

        farmer_act = worker_actions[0] if worker_actions else ["PASS"]
        hands_act = worker_actions[1:] if len(worker_actions) > 1 else []

        return {
            "farmer": farmer_act,
            "hands": hands_act,
            "market": market_orders[:10],
        }

if __name__ == "__main__":
    from scripts.benchmark_seeds import evaluate_agent
    evaluate_agent(EnhancedHRL12WorkerDispatcher, num_seeds=5)
