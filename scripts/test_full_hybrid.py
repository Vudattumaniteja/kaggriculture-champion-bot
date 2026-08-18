"""
Grandmaster Hybrid Market Agent (Full Abracadabra Blueprint).
1. Early Strawberry Inception (Day 6-9 in NE).
2. 100% Dedicated Fertilizer Reinvestment Chore Allocation (Doubling crop yield).
3. Paced 14-Pasture Livestock Expansion (10 Cows + 4 Sheep producing $2.4k/day).
4. NW Wheat Feed Autarky ($0 feed import cost).
5. 24/7 Synchronized Drip-Feeding (Wool >= $175, Milk >= $145, Strawberry >= $115) + Day 28-29 Endgame Flush.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
import kaggle_environments

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

SHED_TILES: Set[Tuple[int, int]] = {(4, 4), (5, 4), (4, 5), (5, 5)}

# Dedicated Pasture locations around Shed (14 tiles max)
PASTURE_LOCS_SET: Set[Tuple[int, int]] = {
    # NW Pastures (5)
    (3, 3), (3, 4), (4, 3), (2, 3), (2, 4),
    # NE Pastures (5)
    (5, 3), (6, 3), (7, 3), (5, 2), (6, 2),
    # SW Pastures (4)
    (3, 5), (3, 6), (2, 5), (2, 6),
}
PASTURE_LOCS_LIST: List[Tuple[int, int]] = list(PASTURE_LOCS_SET)

# Dedicated continuous NW Wheat feed tiles (7 tiles)
NW_WHEAT_TILES: Set[Tuple[int, int]] = {
    (0, 0), (1, 0), (2, 0), (3, 0),
    (0, 1), (1, 1), (2, 1),
}

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
        drip_wool_thresh: float = 175.0,
        drip_milk_thresh: float = 145.0,
        drip_strawberry_thresh: float = 115.0,
        drip_melon_thresh: float = 150.0,
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
        elif day <= 5:
            return 5
        elif day <= 8:
            return 8
        elif day <= 28:
            return 12  # Full 12-worker team
        else:
            return 8   # Day 29: 8 workers to harvest and drop

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
        unlocked_quads = list(my_farm.get("unlocked_quadrants", ["NW"]))
        hires_today = int(my_farm.get("hires_today", 0))

        shed = dict(private.get("shed", {}) or {})
        seeds = dict(private.get("seeds", {}) or {})
        inventories = [dict(inv) for inv in (private.get("inventories", []) or [])]
        market_prices = market.get("prices", {}) or {}

        market_orders: List[List[Any]] = []

        # ==========================================
        # 1. SCAN BOARD STRUCTURES, ANIMALS & TILES
        # ==========================================
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
        fert_in_inv = sum(inv.get("FERTILIZER", 0) for inv in inventories)
        total_animals_owned = total_living_animals + unplaced_shed_animals + cows_in_inv + sheep_in_inv

        # ==========================================
        # 2. STRATEGIC MARKET CONTROLLER
        # ==========================================

        # A. Endgame Liquidation Flush (Days 28-29 on EVERY turn)
        if day >= 28:
            for prod in PRODUCTS:
                cnt = shed.get(prod, 0)
                if cnt > 0:
                    market_orders.append(["SELL", prod, cnt])
        else:
            # 1. Melons: 100% IMMEDIATE LIQUIDATION ON HARVEST (Day 10+)
            melon_cnt = shed.get("MELON", 0)
            if melon_cnt > 0:
                market_orders.append(["SELL", "MELON", melon_cnt])
                money += melon_cnt * market_prices.get("MELON", 250)
                shed["MELON"] = 0

            # 2. Emergency Liquidity Refill (Guarantees money >= 100 for hiring and operations)
            if money < 100.0:
                if shed.get("WOOL", 0) > 0:
                    market_orders.append(["SELL", "WOOL", 1])
                    money += market_prices.get("WOOL", 200)
                    shed["WOOL"] -= 1
                elif shed.get("MILK", 0) > 0:
                    market_orders.append(["SELL", "MILK", 1])
                    money += market_prices.get("MILK", 160)
                    shed["MILK"] -= 1
                elif shed.get("STRAWBERRY", 0) > 0:
                    market_orders.append(["SELL", "STRAWBERRY", min(shed["STRAWBERRY"], 2)])
                    money += min(shed["STRAWBERRY"], 2) * market_prices.get("STRAWBERRY", 120)
                    shed["STRAWBERRY"] -= min(shed["STRAWBERRY"], 2)
                elif shed.get("WHEAT", 0) > 20:
                    market_orders.append(["SELL", "WHEAT", 5])
                    money += 5 * market_prices.get("WHEAT", 25)
                    shed["WHEAT"] -= 5

            # 3. Synchronized Town Shop Drip-Feeding (Triggered EVERY hour H00..H23)
            wool_cnt = shed.get("WOOL", 0)
            if wool_cnt > 0 and market_prices.get("WOOL", 0) >= self.drip_wool_thresh:
                drip_qty = min(wool_cnt, 4)
                market_orders.append(["SELL", "WOOL", drip_qty])

            milk_cnt = shed.get("MILK", 0)
            if milk_cnt > 0 and market_prices.get("MILK", 0) >= self.drip_milk_thresh:
                drip_qty = min(milk_cnt, 4)
                market_orders.append(["SELL", "MILK", drip_qty])

            straw_cnt = shed.get("STRAWBERRY", 0)
            if straw_cnt > 0 and market_prices.get("STRAWBERRY", 0) >= self.drip_strawberry_thresh:
                drip_qty = min(straw_cnt, 8)
                market_orders.append(["SELL", "STRAWBERRY", drip_qty])

            # Wheat: Safe reserve for animal feed; sell surplus if price >= 24
            wheat_cnt = shed.get("WHEAT", 0)
            safe_wheat = max(35, total_living_animals * 3)
            if wheat_cnt > safe_wheat + 15 and market_prices.get("WHEAT", 0) >= 24:
                market_orders.append(["SELL", "WHEAT", wheat_cnt - safe_wheat])

            # Fertilizer: NEVER sell for $1-$14! Reinvest 100% into crops.
            fert_cnt = shed.get("FERTILIZER", 0)
            if day >= 25 and fert_cnt > 15:
                market_orders.append(["SELL", "FERTILIZER", min(fert_cnt - 10, 10)])
            elif fert_cnt > 60 and market_prices.get("FERTILIZER", 0) >= 50:
                market_orders.append(["SELL", "FERTILIZER", min(fert_cnt - 40, 5)])

            for prod in ["CARROT", "TOMATO", "EGG"]:
                cnt = shed.get(prod, 0)
                if cnt > 0:
                    market_orders.append(["SELL", prod, cnt])

        # B. Priority Sequential Land Expansion (Exactly 1 BUY_LAND per turn)
        if "NE" not in unlocked_quads and money >= 1050 and day >= 6:
            market_orders.append(["BUY_LAND"])
            money -= 1000
            unlocked_quads.append("NE")
        elif "SW" not in unlocked_quads and "NE" in unlocked_quads and money >= 2050 and day >= 9:
            market_orders.append(["BUY_LAND"])
            money -= 2000
            unlocked_quads.append("SW")
        elif "SE" not in unlocked_quads and "SW" in unlocked_quads and money >= 4050 and day >= 11:
            market_orders.append(["BUY_LAND"])
            money -= 4000
            unlocked_quads.append("SE")

        # C. Day 0 Turn 0 Initialization ($2,690 spend, $310 buffer)
        if day == 0 and hour == 0:
            if money >= 500:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500
            if money >= 800:
                market_orders.append(["BUY_ANIMAL", "COW", 2])
                money -= 800
            if money >= 960:
                market_orders.append(["BUY_SEED", "MELON", 12])
                money -= 960
            if money >= 70:
                market_orders.append(["BUY_SEED", "WHEAT", 7])
                money -= 70
            if money >= 350:
                market_orders.append(["BUY_PRODUCT", "WHEAT", 14])
                money -= 350

        # Midgame Herd Scaling: Paced acquisition (max 2 animals per day, ONLY if shed is clear)
        elif day <= 18 and (hour == 0 or (day >= 11 and money >= 3000)) and unplaced_shed_animals == 0:
            rem_sheep = max(0, self.max_sheep - (living_sheep + sheep_in_shed + sheep_in_inv))
            rem_cows = max(0, self.max_cows - (living_cows + cows_in_shed + cows_in_inv))
            avail = max(0.0, money - 550.0)

            if total_animals_owned < 14:
                if rem_sheep > 0 and (living_sheep + sheep_in_shed) * 2 <= (living_cows + cows_in_shed) and avail >= 500:
                    market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                    money -= 500
                    sheep_in_shed += 1
                elif rem_cows > 0 and avail >= 400:
                    qty = min(rem_cows, int(avail // 400), 2)
                    if qty > 0:
                        market_orders.append(["BUY_ANIMAL", "COW", qty])
                        money -= qty * 400
                        cows_in_shed += qty

        # Seed Purchasing: Early Strawberry Deployment on Day 6-10, Mass Scaling on Day 11-15
        if day < 26 and (hour == 0 or (day >= 6 and money >= 300)):
            wheat_seeds = seeds.get("WHEAT", 0)
            if wheat_seeds < 15 and money >= 200:
                buy_w = max(6, 18 - wheat_seeds)
                market_orders.append(["BUY_SEED", "WHEAT", buy_w])
                money -= buy_w * 10

            straw_seeds = seeds.get("STRAWBERRY", 0)
            straw_tiles = sum(1 for _, _, ct in crop_tiles if ct.get("crop") == "STRAWBERRY")
            
            if day <= 8:
                target_strawberries = 10
            elif day <= 10:
                target_strawberries = 20
            elif day <= 16:
                target_strawberries = 42
            else:
                target_strawberries = 30

            if "NE" in unlocked_quads and (straw_tiles + straw_seeds) < target_strawberries and money >= 200:
                needed_s = target_strawberries - (straw_tiles + straw_seeds)
                buy_s = min(needed_s, int((money - 100) // 100))
                if buy_s > 0:
                    market_orders.append(["BUY_SEED", "STRAWBERRY", buy_s])
                    money -= buy_s * 100

        # Safety Wheat Feed Auto-Purchase (ONLY at hour 0)
        if hour == 0 and day <= 27:
            wheat_in_shed = shed.get("WHEAT", 0)
            if wheat_in_shed < 25 and money >= 250 and total_living_animals > 0:
                buy_feed = min(15, max(8, total_living_animals * 2 - wheat_in_shed))
                if buy_feed > 0:
                    market_orders.append(["BUY_PRODUCT", "WHEAT", buy_feed])
                    money -= buy_feed * 25

        # Daily Labor Hiring (ONLY at hour 0)
        target_hires = self._get_target_hires(day)
        if hour == 0 and hires_today < target_hires and money >= 10:
            hires_needed = target_hires - hires_today
            for _ in range(hires_needed):
                market_orders.append(["HIRE"])
            hires_today = target_hires

        # ==========================================
        # 3. CHORE SCAN & TASK POOL GENERATION
        # ==========================================
        tasks: List[ChoreTask] = []

        # 1. Livestock Critical Feed (TOP PRIORITY: UTILITY 1200.0)
        for ax, ay, aname, atile in living_animals:
            pos = (ax, ay)
            if not atile.get("fed_today", False):
                tasks.append(ChoreTask("FEED", pos, utility=1200.0, extra={"animal": aname}))

        # 2. Crop Critical Watering (TOP PRIORITY: UTILITY 1150.0)
        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            if not ctile.get("watered_today", False):
                tasks.append(ChoreTask("WATER_CROP", pos, utility=1150.0))

        # 3. Place Animals into Empty Pastures (UTILITY 1100.0)
        if unplaced_shed_animals > 0 or cows_in_inv > 0 or sheep_in_inv > 0:
            for ep in empty_pastures:
                tasks.append(ChoreTask("PLACE_PASTURE_ANIMAL", ep, utility=1100.0))

        # 4. Pasture Construction: ONLY on designated PASTURE_LOCS
        pastures_needed = max(0, total_animals_owned - total_structures)
        if pastures_needed > 0 and len(empty_unlocked_tiles) > 0:
            candidate_spots = [p for p in PASTURE_LOCS_LIST if p in empty_unlocked_tiles]
            if not candidate_spots:
                candidate_spots = sorted([p for p in empty_unlocked_tiles if p not in NW_WHEAT_TILES], key=lambda p: get_manhattan_dist(p, (4, 4)))
            for p in candidate_spots[:pastures_needed]:
                tasks.append(ChoreTask("BUILD_PASTURE", p, utility=1050.0))

        # 5. Crop Harvesting
        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            crop = ctile.get("crop", "WHEAT")
            age = day - ctile.get("planted_day", day)
            yield_units = ctile.get("yield_units", 0)

            if crop == "WHEAT":
                if age >= 4 or (day >= 28 and yield_units > 0):
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=850.0))
            elif crop == "MELON":
                if age >= 10 or (day >= 28 and yield_units > 0):
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=890.0))
            elif crop == "STRAWBERRY":
                if yield_units > 0:
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=880.0))
            else:
                if yield_units > 0 or age >= 4:
                    tasks.append(ChoreTask("HARVEST_CROP", pos, utility=700.0))

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
                        tasks.append(ChoreTask("FERTILIZE_CROP", pos, utility=860.0))
                    elif crop == "STRAWBERRY":
                        tasks.append(ChoreTask("FERTILIZE_CROP", pos, utility=855.0))

        # 7. Livestock Care, Manure & Harvest
        for ax, ay, aname, atile in living_animals:
            pos = (ax, ay)
            if not atile.get("cared_today", False):
                tasks.append(ChoreTask("CARE", pos, utility=850.0, extra={"animal": aname}))
            if atile.get("yield_units", 0) > 0:
                tasks.append(ChoreTask("HARVEST_ANIMAL", pos, utility=840.0, extra={"animal": aname}))
            if atile.get("fertilizer_available", False):
                tasks.append(ChoreTask("COLLECT_FERTILIZER", pos, utility=820.0, extra={"animal": aname}))

        # 8. Planting Chores on Empty Tiles (Excluding Pasture Zones)
        wheat_seeds_avail = seeds.get("WHEAT", 0)
        nw_wheat_empty = [p for p in NW_WHEAT_TILES if p in empty_unlocked_tiles]
        if day <= 26 and wheat_seeds_avail > 0 and nw_wheat_empty:
            for p in nw_wheat_empty[:wheat_seeds_avail]:
                tasks.append(ChoreTask("PLANT_CROP", p, utility=810.0, extra="WHEAT"))

        straw_seeds_avail = seeds.get("STRAWBERRY", 0)
        straw_empty_tiles = [p for p in empty_unlocked_tiles if p not in NW_WHEAT_TILES and p not in PASTURE_LOCS_SET]
        if day <= 26 and straw_seeds_avail > 0 and straw_empty_tiles:
            for p in straw_empty_tiles[:straw_seeds_avail]:
                tasks.append(ChoreTask("PLANT_CROP", p, utility=800.0, extra="STRAWBERRY"))

        # 9. Digging Weeds
        for wx, wy in weed_tiles:
            tasks.append(ChoreTask("DIG_WEED", (wx, wy), utility=350.0))

        # ==========================================
        # 4. INVENTORY-AWARE HUNGARIAN MATCHING
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
                    raw_dist = get_manhattan_dist(w_pos, t_pos)
                    effective_dist = float(raw_dist)

                    if has_animal:
                        if task.task_type == "PLACE_PASTURE_ANIMAL":
                            effective_dist = float(raw_dist) - 50.0
                        else:
                            cost_matrix[i, j] = 99999.0
                            continue

                    elif task.task_type == "PLACE_PASTURE_ANIMAL":
                        if not is_at_shed:
                            dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 2.0

                    elif task.task_type == "FEED":
                        if has_wheat:
                            effective_dist = float(raw_dist) - 2.0
                        elif not is_at_shed:
                            dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    elif task.task_type == "FERTILIZE_CROP":
                        if has_fert:
                            effective_dist = float(raw_dist) - 2.0
                        elif not is_at_shed:
                            dist_to_shed = min(get_manhattan_dist(w_pos, s) for s in SHED_TILES)
                            dist_shed_to_task = min(get_manhattan_dist(s, t_pos) for s in SHED_TILES)
                            effective_dist = dist_to_shed + dist_shed_to_task + 1.0

                    cost_matrix[i, j] = (effective_dist * 4.0) - (task.utility * 0.1)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_task_map = {row: col for row, col in zip(row_ind, col_ind)}
        else:
            assigned_task_map = {}

        # ==========================================
        # 5. ACTION GENERATION PER WORKER
        # ==========================================
        for i, (w_idx, w_pos, w_inv) in enumerate(all_workers):
            wx, wy = w_pos
            w_tile = tiles[wy][wx] if wy < len(tiles) and wx < len(tiles[wy]) else None
            is_at_shed = w_pos in SHED_TILES
            assigned_task = tasks[assigned_task_map[i]] if i in assigned_task_map else None

            # 1. Shed Operations (Pickup / Drop)
            if is_at_shed:
                # Drop harvested produce at shed
                if any(w_inv.get(p, 0) > 0 for p in ["STRAWBERRY", "MELON", "MILK", "WOOL", "EGG"]):
                    worker_actions[i] = ["DROP"]
                    continue

                if day >= 27 and any(w_inv.get(p, 0) > 0 for p in PRODUCTS):
                    worker_actions[i] = ["DROP"]
                    continue

                # Pickup Animals if assigned to PLACE_PASTURE_ANIMAL
                if assigned_task and assigned_task.task_type == "PLACE_PASTURE_ANIMAL" and w_inv.get("SHEEP", 0) == 0 and w_inv.get("COW", 0) == 0:
                    if shed.get("SHEEP", 0) > 0:
                        worker_actions[i] = ["PICKUP", "SHEEP", 1]
                        continue
                    elif shed.get("COW", 0) > 0:
                        worker_actions[i] = ["PICKUP", "COW", 1]
                        continue

                # Pickup Wheat for unfed animals
                unfed = sum(1 for _, _, _, at in living_animals if not at.get("fed_today", False))
                if unfed > 0 and w_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    take_qty = min(6, shed.get("WHEAT", 0), unfed)
                    if take_qty > 0:
                        worker_actions[i] = ["PICKUP", "WHEAT", take_qty]
                        continue

                # Batch pickup Fertilizer whenever leaving the shed
                if w_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    take_f = min(4, shed.get("FERTILIZER", 0))
                    if take_f > 0:
                        worker_actions[i] = ["PICKUP", "FERTILIZER", take_f]
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

            elif w_tile is None and w_pos not in SHED_TILES:
                if assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "BUILD_PASTURE":
                    worker_actions[i] = ["BUILD_PASTURE"]
                    continue

                if (assigned_task and assigned_task.pos == w_pos and assigned_task.task_type == "PLANT_CROP") or (assigned_task is None and day <= 26):
                    # Never plant crops on dedicated pasture locations
                    if w_pos not in PASTURE_LOCS_SET:
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
