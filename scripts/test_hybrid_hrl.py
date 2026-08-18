"""
Upgraded Grandmaster Hybrid Market Agent with Inventory-Aware Hungarian Dispatcher.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
import kaggle_environments

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

SHED_TILES: List[Tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]

# Designated NW Pasture locations around Shed (4 tiles)
NW_PASTURE_LOCS: List[Tuple[int, int]] = [(3, 3), (3, 4), (4, 3), (2, 3)]

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

    def __repr__(self):
        return f"Task({self.task_type}, pos={self.pos}, util={self.utility})"

class HybridMarketAgent:
    def __init__(
        self,
        max_cows: int = 10,
        max_sheep: int = 4,
        drip_wool_thresh: float = 180.0,
        drip_milk_thresh: float = 150.0,
        drip_strawberry_thresh: float = 120.0,
        drip_melon_thresh: float = 160.0,
    ):
        self.max_cows = max_cows
        self.max_sheep = max_sheep
        self.drip_wool_thresh = drip_wool_thresh
        self.drip_milk_thresh = drip_milk_thresh
        self.drip_strawberry_thresh = drip_strawberry_thresh
        self.drip_melon_thresh = drip_melon_thresh

    def _get_target_hires(self, day: int) -> int:
        if day == 0:
            return 5
        elif day <= 6:
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
            return 0  # Day 29: 0 hires to maximize final cash

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        player = obs["player"]
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
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
        unlocked_quads = list(my_farm.get("unlocked_quadrants", ["NW"]))
        hires_today = int(my_farm.get("hires_today", 0))

        shed = private.get("shed", {}) or {}
        seeds = private.get("seeds", {}) or {}
        inventories = private.get("inventories", []) or []
        market_prices = market.get("prices", {}) or {}

        market_orders: List[List[Any]] = []

        # 1. Scan Board
        living_animals: List[Tuple[int, int, str, Dict[str, Any]]] = []
        living_cows = 0
        living_sheep = 0
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
        total_animals_owned = total_living_animals + unplaced_shed_animals + cows_in_inv + sheep_in_inv

        # 2. Market Execution
        # A. Endgame Liquidation Flush (Days 28-29)
        if day >= 28:
            for prod in PRODUCTS:
                cnt = shed.get(prod, 0)
                if cnt > 0:
                    market_orders.append(["SELL", prod, cnt])
        else:
            # Synchronized Town Shop Drip-Feeding
            # WOOL: drip 2-4 units if price >= 180
            wool_cnt = shed.get("WOOL", 0)
            if wool_cnt > 0 and market_prices.get("WOOL", 0) >= self.drip_wool_thresh:
                drip_qty = min(wool_cnt, 4)
                market_orders.append(["SELL", "WOOL", drip_qty])

            # MILK: drip 2-4 units if price >= 150
            milk_cnt = shed.get("MILK", 0)
            if milk_cnt > 0 and market_prices.get("MILK", 0) >= self.drip_milk_thresh:
                drip_qty = min(milk_cnt, 4)
                market_orders.append(["SELL", "MILK", drip_qty])

            # STRAWBERRY: drip 4-8 units if price >= 120
            straw_cnt = shed.get("STRAWBERRY", 0)
            if straw_cnt > 0 and market_prices.get("STRAWBERRY", 0) >= self.drip_strawberry_thresh:
                drip_qty = min(straw_cnt, 8)
                market_orders.append(["SELL", "STRAWBERRY", drip_qty])

            # MELON: sell at >= 150 (or Day 10-12 flush)
            melon_cnt = shed.get("MELON", 0)
            if melon_cnt > 0 and (market_prices.get("MELON", 0) >= self.drip_melon_thresh or day >= 10):
                drip_qty = min(melon_cnt, 6)
                market_orders.append(["SELL", "MELON", drip_qty])

            # WHEAT: Safe reserve of 15-20 wheat for animal feed; sell surplus if price >= 24
            wheat_cnt = shed.get("WHEAT", 0)
            safe_wheat = max(15, total_living_animals + 5)
            if wheat_cnt > safe_wheat + 10 and market_prices.get("WHEAT", 0) >= 24:
                market_orders.append(["SELL", "WHEAT", wheat_cnt - safe_wheat])

            # FERTILIZER: NEVER sell for $1-$14! Hold for crop fertilizing.
            # Only sell if shed is overflowing (> 50 units) and price >= 60.
            fert_cnt = shed.get("FERTILIZER", 0)
            if fert_cnt > 50 and market_prices.get("FERTILIZER", 0) >= 60:
                market_orders.append(["SELL", "FERTILIZER", min(fert_cnt - 35, 5)])

            # Other minor produce
            for prod in ["CARROT", "TOMATO", "EGG"]:
                cnt = shed.get(prod, 0)
                if cnt > 0:
                    market_orders.append(["SELL", prod, cnt])

        # B. Priority Land Expansion
        if "NE" not in unlocked_quads and money >= 1000 and day >= 6:
            market_orders.append(["BUY_LAND"])
            money -= 1000
            unlocked_quads.append("NE")
        if "SW" not in unlocked_quads and "NE" in unlocked_quads and money >= 2000 and day >= 10:
            market_orders.append(["BUY_LAND"])
            money -= 2000
            unlocked_quads.append("SW")
        if "SE" not in unlocked_quads and "SW" in unlocked_quads and money >= 4000 and day >= 14:
            market_orders.append(["BUY_LAND"])
            money -= 4000
            unlocked_quads.append("SE")

        # C. Day 0 Turn 0 Initialization ($2,980 Budget)
        if day == 0 and hour == 0:
            if money >= 1000:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 2])
                money -= 1000
            if money >= 800:
                market_orders.append(["BUY_ANIMAL", "COW", 2])
                money -= 800
            if money >= 960:
                market_orders.append(["BUY_SEED", "MELON", 12])
                money -= 960
            if money >= 70:
                market_orders.append(["BUY_SEED", "WHEAT", 7])
                money -= 70
            if money >= 125:
                market_orders.append(["BUY_PRODUCT", "WHEAT", 5])
                money -= 125

        elif day <= 16 and hour == 0:
            rem_sheep = max(0, self.max_sheep - (living_sheep + sheep_in_shed + sheep_in_inv))
            rem_cows = max(0, self.max_cows - (living_cows + cows_in_shed + cows_in_inv))
            avail = max(0.0, money - 500.0)

            if rem_sheep > 0 and (living_sheep + sheep_in_shed) * 2 <= (living_cows + cows_in_shed) and avail >= 500:
                qty = min(rem_sheep, int(avail // 500), 2)
                if qty > 0:
                    market_orders.append(["BUY_ANIMAL", "SHEEP", qty])
                    money -= qty * 500
                    sheep_in_shed += qty
            elif rem_cows > 0 and avail >= 400:
                qty = min(rem_cows, int(avail // 400), 2)
                if qty > 0:
                    market_orders.append(["BUY_ANIMAL", "COW", qty])
                    money -= qty * 400
                    cows_in_shed += qty

        # Seed Purchasing
        if hour == 0 and day < 26:
            wheat_seeds = seeds.get("WHEAT", 0)
            if wheat_seeds < 8 and money >= 100:
                buy_w = max(4, 10 - wheat_seeds)
                market_orders.append(["BUY_SEED", "WHEAT", buy_w])
                money -= buy_w * 10

            straw_seeds = seeds.get("STRAWBERRY", 0)
            straw_tiles = sum(1 for _, _, ct in crop_tiles if ct.get("crop") == "STRAWBERRY")
            if "NE" in unlocked_quads and straw_seeds < 10 and (straw_tiles + straw_seeds) < 35 and money >= 1000:
                buy_s = min(15, int(money // 200))
                if buy_s > 0:
                    market_orders.append(["BUY_SEED", "STRAWBERRY", buy_s])
                    money -= buy_s * 100

        # Daily Labor Hiring
        target_hires = self._get_target_hires(day)
        if hour <= 1 and hires_today < target_hires and money >= 10:
            hires_needed = target_hires - hires_today
            for _ in range(hires_needed):
                market_orders.append(["HIRE"])
            hires_today = target_hires

        # 3. Chore Task Pool Generation
        tasks: List[ChoreTask] = []

        # Livestock Chores (Highest Priority: Feed, Care, Fertilizer Collection)
        for ax, ay, aname, atile in living_animals:
            pos = (ax, ay)
            if not atile.get("fed_today", False):
                tasks.append(ChoreTask("FEED", pos, utility=1000.0, extra={"animal": aname}))
            if not atile.get("cared_today", False):
                tasks.append(ChoreTask("CARE", pos, utility=950.0, extra={"animal": aname}))
            if atile.get("fertilizer_available", False):
                tasks.append(ChoreTask("COLLECT_FERTILIZER", pos, utility=920.0, extra={"animal": aname}))
            if atile.get("yield_units", 0) > 0:
                tasks.append(ChoreTask("HARVEST_ANIMAL", pos, utility=880.0, extra={"animal": aname}))

        # Place Animals in Empty Pastures
        if unplaced_shed_animals > 0 or cows_in_inv > 0 or sheep_in_inv > 0:
            for ep in empty_pastures:
                tasks.append(ChoreTask("PLACE_PASTURE_ANIMAL", ep, utility=890.0))

        # Pasture Construction: ONLY for owned animals needing structures
        pastures_needed = max(0, total_animals_owned - total_structures)
        if pastures_needed > 0 and len(empty_unlocked_tiles) > 0:
            if day == 0:
                candidate_spots = [p for p in NW_PASTURE_LOCS if p in empty_unlocked_tiles]
                for p in candidate_spots[:pastures_needed]:
                    tasks.append(ChoreTask("BUILD_PASTURE", p, utility=910.0))
            else:
                sorted_empty = sorted(empty_unlocked_tiles, key=lambda p: get_manhattan_dist(p, (4, 4)))
                for p in sorted_empty[:pastures_needed]:
                    tasks.append(ChoreTask("BUILD_PASTURE", p, utility=860.0))

        # Crop Chores & Internal Fertilizer Reinvestment
        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            crop = ctile.get("crop", "WHEAT")
            age = day - ctile.get("planted_day", day)
            yield_units = ctile.get("yield_units", 0)
            fert_until = ctile.get("fertilized_until_day", -1)

            if crop == "WHEAT":
                if age >= 4 or (day >= 28 and yield_units > 0):
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=680.0))
                elif not ctile.get("watered_today", False):
                    tasks.append(ChoreTask("WATER_CROP", pos, utility=600.0))
            elif crop == "MELON":
                if age >= 10 or (day >= 28 and yield_units > 0):
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=850.0))
                elif not ctile.get("watered_today", False):
                    tasks.append(ChoreTask("WATER_CROP", pos, utility=750.0))
                elif fert_until < day and age <= 10:
                    tasks.append(ChoreTask("FERTILIZE_CROP", pos, utility=780.0))
            elif crop == "STRAWBERRY":
                if yield_units > 0:
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=820.0))
                elif not ctile.get("watered_today", False):
                    tasks.append(ChoreTask("WATER_CROP", pos, utility=720.0))
                elif fert_until < day and day <= 27:
                    tasks.append(ChoreTask("FERTILIZE_CROP", pos, utility=760.0))
            else:
                if yield_units > 0 or age >= 4:
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=600.0))
                elif not ctile.get("watered_today", False):
                    tasks.append(ChoreTask("WATER_CROP", pos, utility=550.0))

        # Planting Chores
        seeds_available = sum(seeds.values())
        if day <= 26 and seeds_available > 0 and len(empty_unlocked_tiles) > 0:
            for p in empty_unlocked_tiles:
                if pastures_needed > 0 and p in NW_PASTURE_LOCS:
                    continue
                tasks.append(ChoreTask("PLANT_CROP", p, utility=650.0))

        # Dig Weeds
        for wx, wy in weed_tiles:
            tasks.append(ChoreTask("DIG_WEED", (wx, wy), utility=350.0))

        # 4. Inventory-Aware Hungarian Assignment
        all_workers = []
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
                    raw_dist = get_manhattan_dist(w_pos, t_pos)
                    effective_dist = float(raw_dist)

                    # Rules: Farmhands cannot build pastures
                    if w_idx > 0 and task.task_type == "BUILD_PASTURE":
                        cost_matrix[i, j] = 9999.0
                        continue

                    # Worker already holding animal -> must prioritize placing it
                    if has_animal:
                        if task.task_type == "PLACE_PASTURE_ANIMAL":
                            effective_dist = float(raw_dist) - 50.0
                        else:
                            effective_dist = float(raw_dist) + 200.0

                    elif task.task_type == "PLACE_PASTURE_ANIMAL":
                        # Needs to get animal from shed first
                        if not is_at_shed:
                            dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    elif task.task_type == "FEED":
                        if has_wheat:
                            effective_dist = float(raw_dist) - 20.0
                        elif not is_at_shed:
                            dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    elif task.task_type == "FERTILIZE_CROP":
                        if has_fert:
                            effective_dist = float(raw_dist) - 20.0
                        elif not is_at_shed:
                            dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    cost_matrix[i, j] = effective_dist - (task.utility * 0.1)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_task_map = {row: col for row, col in zip(row_ind, col_ind)}
        else:
            assigned_task_map = {}

        # 5. Worker Step Execution
        # Pre-count animals carried
        workers_carrying_animals = sum(1 for _, _, inv in all_workers if inv.get("COW", 0) > 0 or inv.get("SHEEP", 0) > 0)

        for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
            wx, wy = w_pos
            w_tile = tiles[wy][wx] if wy < len(tiles) and wx < len(tiles[wy]) else None
            is_at_shed = w_pos in SHED_TILES

            # 1. Shed Operations
            if is_at_shed:
                # Endgame DROP
                if day >= 27 and any(w_inv.get(p, 0) > 0 for p in PRODUCTS):
                    worker_actions[i] = ["DROP"]
                    for k in list(w_inv.keys()):
                        del w_inv[k]
                    continue

                # Pickup Animals if empty pastures exist and worker has no animal
                if empty_pastures and w_inv.get("SHEEP", 0) == 0 and w_inv.get("COW", 0) == 0:
                    if workers_carrying_animals < len(empty_pastures):
                        if shed.get("SHEEP", 0) > 0:
                            worker_actions[i] = ["PICKUP", "SHEEP", 1]
                            shed["SHEEP"] -= 1
                            w_inv["SHEEP"] = 1
                            workers_carrying_animals += 1
                            continue
                        elif shed.get("COW", 0) > 0:
                            worker_actions[i] = ["PICKUP", "COW", 1]
                            shed["COW"] -= 1
                            w_inv["COW"] = 1
                            workers_carrying_animals += 1
                            continue

                # Pickup Wheat for unfed animals
                unfed = sum(1 for _, _, _, at in living_animals if not at.get("fed_today", False))
                if unfed > 0 and w_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    take_qty = min(6, shed.get("WHEAT", 0), unfed)
                    if take_qty > 0:
                        worker_actions[i] = ["PICKUP", "WHEAT", take_qty]
                        shed["WHEAT"] -= take_qty
                        w_inv["WHEAT"] = w_inv.get("WHEAT", 0) + take_qty
                        continue

                # Pickup Fertilizer for crops
                fert_tasks = any(t.task_type == "FERTILIZE_CROP" for t in tasks)
                if fert_tasks and w_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    take_f = min(3, shed.get("FERTILIZER", 0))
                    if take_f > 0:
                        worker_actions[i] = ["PICKUP", "FERTILIZER", take_f]
                        shed["FERTILIZER"] -= take_f
                        w_inv["FERTILIZER"] = w_inv.get("FERTILIZER", 0) + take_f
                        continue

            # 2. Standing Tile Operations
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
                        # Harvest
                        if w_tile.get("yield_units", 0) > 0:
                            worker_actions[i] = ["HARVEST"]
                            w_tile["yield_units"] = 0
                            continue
                    else:
                        # Place Animal into empty pasture
                        if w_inv.get("SHEEP", 0) > 0:
                            worker_actions[i] = ["PLACE", "SHEEP"]
                            w_inv["SHEEP"] -= 1
                            continue
                        elif w_inv.get("COW", 0) > 0:
                            worker_actions[i] = ["PLACE", "COW"]
                            w_inv["COW"] -= 1
                            continue

                elif k == "PLANT":
                    crop = w_tile.get("crop", "WHEAT")
                    age = day - w_tile.get("planted_day", day)
                    yield_u = w_tile.get("yield_units", 0)
                    fert_until = w_tile.get("fertilized_until_day", -1)

                    if crop == "WHEAT":
                        if age >= 4 or (day >= 28 and yield_u > 0):
                            worker_actions[i] = ["HARVEST"]
                            continue
                        if not w_tile.get("watered_today", False):
                            worker_actions[i] = ["WATER"]
                            w_tile["watered_today"] = True
                            continue
                    elif crop == "MELON":
                        if age >= 10 or (day >= 28 and yield_u > 0):
                            worker_actions[i] = ["HARVEST"]
                            continue
                        if not w_tile.get("watered_today", False):
                            worker_actions[i] = ["WATER"]
                            w_tile["watered_today"] = True
                            continue
                        if fert_until < day and w_inv.get("FERTILIZER", 0) > 0 and age <= 10:
                            worker_actions[i] = ["FERTILIZE"]
                            w_inv["FERTILIZER"] -= 1
                            continue
                    elif crop == "STRAWBERRY":
                        if yield_u > 0:
                            worker_actions[i] = ["HARVEST"]
                            continue
                        if not w_tile.get("watered_today", False):
                            worker_actions[i] = ["WATER"]
                            w_tile["watered_today"] = True
                            continue
                        if fert_until < day and w_inv.get("FERTILIZER", 0) > 0 and day <= 27:
                            worker_actions[i] = ["FERTILIZE"]
                            w_inv["FERTILIZER"] -= 1
                            continue
                    else:
                        if yield_u > 0 or age >= 4:
                            worker_actions[i] = ["HARVEST"]
                            continue
                        if not w_tile.get("watered_today", False):
                            worker_actions[i] = ["WATER"]
                            w_tile["watered_today"] = True
                            continue

                elif k == "WEED":
                    worker_actions[i] = ["DIG"]
                    continue

            elif w_tile is None and w_pos not in SHED_TILES:
                assigned_task = tasks[assigned_task_map[i]] if i in assigned_task_map else None

                if assigned_task and assigned_task.task_type == "BUILD_PASTURE" and w_idx == 0:
                    worker_actions[i] = ["BUILD_PASTURE"]
                    total_structures += 1
                    continue

                if (assigned_task and assigned_task.task_type == "PLANT_CROP") or (assigned_task is None and day <= 26):
                    # Planting Logic
                    if day == 0:
                        # Day 0: Plant Melons (12) and Wheat (7)
                        if seeds.get("MELON", 0) > 0:
                            worker_actions[i] = ["PLANT", "MELON"]
                            seeds["MELON"] -= 1
                            continue
                        elif seeds.get("WHEAT", 0) > 0:
                            worker_actions[i] = ["PLANT", "WHEAT"]
                            seeds["WHEAT"] -= 1
                            continue
                    else:
                        is_nw = (wx < 5 and wy < 5)
                        if is_nw:
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
    bot = HybridMarketAgent()
    env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([bot, "starter"])
    print("Match finished! Final rewards:", [s.reward for s in env.steps[-1]])
