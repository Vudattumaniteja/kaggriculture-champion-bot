"""
Two-Stage Market Guardrails and Neural-Weighted Hungarian Micro Assignment Solver.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

CROP_SPECS: Dict[str, Dict[str, Any]] = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
}

BASE_PRICES: Dict[str, float] = {
    "WHEAT": 25.0, "CARROT": 35.0, "TOMATO": 60.0, "STRAWBERRY": 120.0, "MELON": 250.0,
    "EGG": 50.0, "MILK": 160.0, "WOOL": 200.0, "FERTILIZER": 100.0,
}

SHED_TILES: List[Tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]


def get_manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_step_towards(curr: Tuple[int, int], target: Tuple[int, int], occupied: Optional[Set[Tuple[int, int]]] = None) -> Tuple[str, Tuple[int, int]]:
    cx, cy = curr
    tx, ty = target
    if cx == tx and cy == ty:
        return "PASS", curr

    occupied = occupied or set()
    moves: List[Tuple[str, Tuple[int, int]]] = []
    
    # Priority ordered moves
    dx = tx - cx
    dy = ty - cy
    if abs(dx) >= abs(dy):
        if dx > 0:
            moves.append(("EAST", (cx + 1, cy)))
        elif dx < 0:
            moves.append(("WEST", (cx - 1, cy)))
        if dy > 0:
            moves.append(("SOUTH", (cx, cy + 1)))
        elif dy < 0:
            moves.append(("NORTH", (cx, cy - 1)))
    else:
        if dy > 0:
            moves.append(("SOUTH", (cx, cy + 1)))
        elif dy < 0:
            moves.append(("NORTH", (cx, cy - 1)))
        if dx > 0:
            moves.append(("EAST", (cx + 1, cy)))
        elif dx < 0:
            moves.append(("WEST", (cx - 1, cy)))

    # Try preferred moves without stepping into occupied tiles
    for direction, next_pos in moves:
        nx, ny = next_pos
        if 0 <= nx < 10 and 0 <= ny < 10 and next_pos not in occupied:
            return direction, next_pos

    # If all preferred blocked, try any legal unoccupied move
    for direction, (nx, ny) in [("NORTH", (cx, cy - 1)), ("SOUTH", (cx, cy + 1)), ("EAST", (cx + 1, cy)), ("WEST", (cx - 1, cy))]:
        if 0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in occupied:
            return direction, (nx, ny)

    return "PASS", curr


def apply_market_guardrails(
    market_fractions: np.ndarray,
    obs: Dict[str, Any],
    seed_replenish_logits: Optional[np.ndarray] = None,
    land_expand_logit: float = 0.0,
    workforce_logit_idx: int = 0
) -> List[List[Any]]:
    """
    Applies deterministic safety guardrails on top of neural liquidation fractions:
    1. Cow feed wheat reservation (preserve >= cows * 2 in shed).
    2. Midnight shed overflow emergency liquidation (> 100 items dumped at t % 24 == 23).
    3. Land expansion and workforce recruiting filters.
    4. Autonomous seed replenishment.
    """
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}
    prices = market.get("prices", {}) or {}

    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    money = float(my_farm.get("money", 0.0))
    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    unlocked_quads = set(my_farm.get("unlocked_quadrants", ["NW"]))
    hires_today = int(my_farm.get("hires_today", 0))

    # Count living cows / animals
    tiles = my_farm.get("tiles", [])
    living_cows = 0
    living_animals = 0
    for r in range(len(tiles)):
        for c in range(len(tiles[r])):
            t = tiles[r][c]
            if isinstance(t, dict):
                an = t.get("animal")
                if an:
                    living_animals += 1
                    if an == "COW":
                        living_cows += 1

    market_orders: List[List[Any]] = []

    # 1. Stage 1 & Stage 2 Sell Orders
    cow_feed_reservation = max(4, living_cows * 2) if day < 28 else 0
    sold_counts: Dict[str, int] = {}

    for i, prod in enumerate(PRODUCTS):
        qty = shed.get(prod, 0)
        if qty <= 0:
            continue
        
        frac = float(market_fractions[i]) if i < len(market_fractions) else 0.0
        # On endgame days (day >= 28), force full liquidation
        if day >= 28:
            frac = 1.0

        if prod == "WHEAT":
            avail = max(0, qty - cow_feed_reservation)
        else:
            avail = qty

        sell_qty = int(math.floor(avail * frac)) if frac < 0.99 else avail
        if sell_qty > 0:
            market_orders.append(["SELL", prod, sell_qty])
            sold_counts[prod] = sell_qty
            shed[prod] -= sell_qty

    # 2. Midnight Shed Overflow Guardrail
    remaining_shed_total = sum(shed.values())
    if hour == 23 and remaining_shed_total > 100:
        excess = remaining_shed_total - 95
        # Dump highest spot price commodities first
        sorted_prods = sorted(shed.keys(), key=lambda p: float(prices.get(p, BASE_PRICES.get(p, 25.0))), reverse=True)
        for prod in sorted_prods:
            if excess <= 0:
                break
            can_dump = shed.get(prod, 0)
            if prod == "WHEAT" and day < 28:
                can_dump = max(0, can_dump - cow_feed_reservation)
            dump_qty = min(can_dump, excess)
            if dump_qty > 0:
                # Update existing SELL order or append new one
                existing = [o for o in market_orders if o[0] == "SELL" and o[1] == prod]
                if existing:
                    existing[0][2] += dump_qty
                else:
                    market_orders.append(["SELL", prod, dump_qty])
                shed[prod] -= dump_qty
                excess -= dump_qty

    # 3. Land Expansion
    if day <= 16 and land_expand_logit > 0.0:
        if "NE" not in unlocked_quads and money >= 1200:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked_quads and "NE" in unlocked_quads and money >= 2400 and day <= 10:
            market_orders.append(["BUY_LAND"])
            money -= 2000
        elif "SE" not in unlocked_quads and "SW" in unlocked_quads and money >= 4500 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 4000

    # 4. Workforce Recruiting
    num_hands = len(my_farm.get("hands", []))
    if day < 28 and hour <= 1 and hires_today < 2 and workforce_logit_idx > num_hands:
        if money >= (num_hands + 2) * 20:
            market_orders.append(["HIRE"])
            money -= 20

    # 5. Autonomous Seed Replenishment
    if seed_replenish_logits is not None and day < 27 and money >= 50:
        for i, crop in enumerate(CROPS):
            logit = float(seed_replenish_logits[i])
            cur_seeds = seeds.get(crop, 0)
            cost = CROP_SPECS[crop]["seed"]
            if logit > 0.0 and cur_seeds < 6 and money >= cost * 2:
                buy_qty = min(4, int(money // cost))
                if buy_qty > 0:
                    market_orders.append(["BUY_SEED", crop, buy_qty])
                    money -= buy_qty * cost

    return market_orders[:10]


def solve_micro_actions(
    obs: Dict[str, Any],
    crop_heatmaps: Optional[np.ndarray] = None,
    livestock_quotas: Optional[np.ndarray] = None
) -> Tuple[List[Any], List[List[Any]]]:
    """
    Formulates worker chore assignment as bipartite matching.
    Cost matrix: C_ij = Dist(w_i, t_j) - Priority(t_j)
    Solved via scipy.optimize.linear_sum_assignment with biological urgency overrides.
    """
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}
    town = obs.get("town", {}) or {}

    day = int(obs.get("day", 0))
    hour = int(obs.get("hour", 0))
    farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
    hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
    tiles = my_farm.get("tiles", [])
    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    inventories = private.get("inventories", []) or []
    unlocked_quads = set(my_farm.get("unlocked_quadrants", ["NW"]))

    units = [farmer_pos] + hands_pos
    num_units = len(units)

    # 1. Immediate in-place tile actions
    unit_actions: List[Optional[List[Any]]] = [None] * num_units
    occupied_destinations: Set[Tuple[int, int]] = set()

    for u_idx, u_pos in enumerate(units):
        ux, uy = u_pos
        u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
        u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
        is_at_shed = u_pos in SHED_TILES

        if isinstance(u_tile, dict):
            k = u_tile.get("kind")
            an = u_tile.get("animal")
            if k in ["COOP", "PASTURE"] and an:
                if not u_tile.get("fed_today", False):
                    unit_actions[u_idx] = ["FEED"]
                    occupied_destinations.add(u_pos)
                    continue
                if not u_tile.get("cared_today", False):
                    unit_actions[u_idx] = ["CARE"]
                    occupied_destinations.add(u_pos)
                    continue
                if u_tile.get("yield_units", 0) > 0:
                    unit_actions[u_idx] = ["HARVEST"]
                    occupied_destinations.add(u_pos)
                    continue
                if u_tile.get("fertilizer_available", False):
                    unit_actions[u_idx] = ["COLLECT_FERTILIZER"]
                    occupied_destinations.add(u_pos)
                    continue
            elif k == "PLANT":
                crop = u_tile.get("crop", "CARROT")
                cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                age = day - u_tile.get("planted_day", day)
                yield_u = u_tile.get("yield_units", 0)
                if (not cspec["ongoing"] and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                    cspec["ongoing"] and age >= cspec["first_yield_day"] and yield_u > 0
                ):
                    unit_actions[u_idx] = ["HARVEST"]
                    occupied_destinations.add(u_pos)
                    continue
                if not u_tile.get("watered_today", False):
                    unit_actions[u_idx] = ["WATER"]
                    occupied_destinations.add(u_pos)
                    continue
            elif k == "WEED":
                unit_actions[u_idx] = ["DIG"]
                occupied_destinations.add(u_pos)
                continue

    # 2. Collect pending tasks on the farm
    # Task tuple: (pos, priority_score, task_type, crop_name)
    tasks: List[Tuple[Tuple[int, int], float, str, Optional[str]]] = []

    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            pos = (c, r)

            if isinstance(t, dict):
                k = t.get("kind")
                an = t.get("animal")
                if k in ["COOP", "PASTURE"] and an:
                    if not t.get("fed_today", False):
                        tasks.append((pos, 25.0, "FEED", None))  # Biological urgency: Starving animal
                    elif not t.get("cared_today", False):
                        tasks.append((pos, 18.0, "CARE", None))
                    elif t.get("yield_units", 0) > 0:
                        tasks.append((pos, 20.0, "HARVEST_ANIMAL", None))
                    elif t.get("fertilizer_available", False):
                        tasks.append((pos, 10.0, "FERT_ANIMAL", None))
                elif k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                    age = day - t.get("planted_day", day)
                    yield_u = t.get("yield_units", 0)
                    if (not cspec["ongoing"] and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                        cspec["ongoing"] and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        tasks.append((pos, 22.0, "HARVEST_CROP", None))
                    elif not t.get("watered_today", False):
                        tasks.append((pos, 16.0, "WATER_CROP", None))
                elif k == "WEED":
                    tasks.append((pos, 8.0, "DIG_WEED", None))
            elif t is None and pos not in SHED_TILES and day < 27:
                # Empty planting tile
                best_crop = "CARROT"
                priority_bonus = 5.0
                if crop_heatmaps is not None:
                    # argmax crop at (r, c)
                    c_logits = crop_heatmaps[:, r, c]
                    best_crop_idx = int(np.argmax(c_logits))
                    best_crop = CROPS[best_crop_idx]
                    priority_bonus += float(c_logits[best_crop_idx])
                
                tasks.append((pos, priority_bonus, "PLANT_TILE", best_crop))

    # 3. Formulate and solve bipartite matching for unassigned units
    unassigned_unit_indices = [i for i, act in enumerate(unit_actions) if act is None]

    if unassigned_unit_indices and tasks:
        num_free = len(unassigned_unit_indices)
        num_tasks = len(tasks)
        cost_matrix = np.zeros((num_free, num_tasks), dtype=np.float32)

        for i_idx, u_idx in enumerate(unassigned_unit_indices):
            u_pos = units[u_idx]
            for j_idx, (t_pos, priority, _, _) in enumerate(tasks):
                dist = get_manhattan_dist(u_pos, t_pos)
                cost_matrix[i_idx, j_idx] = float(dist) - priority

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tasks: Set[int] = set()
        for i_idx, j_idx in zip(row_ind, col_ind):
            u_idx = unassigned_unit_indices[i_idx]
            u_pos = units[u_idx]
            t_pos, _, task_type, best_crop = tasks[j_idx]
            assigned_tasks.add(j_idx)

            if u_pos == t_pos:
                # Stand on task
                if task_type == "PLANT_TILE" and best_crop:
                    unit_actions[u_idx] = ["PLANT", best_crop]
                elif task_type == "FEED":
                    unit_actions[u_idx] = ["FEED"]
                elif task_type == "CARE":
                    unit_actions[u_idx] = ["CARE"]
                elif task_type in ["HARVEST_ANIMAL", "HARVEST_CROP"]:
                    unit_actions[u_idx] = ["HARVEST"]
                elif task_type == "DIG_WEED":
                    unit_actions[u_idx] = ["DIG"]
                else:
                    unit_actions[u_idx] = ["PASS"]
                occupied_destinations.add(u_pos)
            else:
                direction, next_pos = get_step_towards(u_pos, t_pos, occupied_destinations)
                unit_actions[u_idx] = [direction]
                occupied_destinations.add(next_pos)

    # 4. Fallback for remaining idle units -> walk to shed or PASS
    for u_idx in range(num_units):
        if unit_actions[u_idx] is None:
            u_pos = units[u_idx]
            if u_pos not in SHED_TILES:
                nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                direction, next_pos = get_step_towards(u_pos, nearest_shed, occupied_destinations)
                unit_actions[u_idx] = [direction]
                occupied_destinations.add(next_pos)
            else:
                unit_actions[u_idx] = ["PASS"]
                occupied_destinations.add(u_pos)

    farmer_action = unit_actions[0] if unit_actions else ["PASS"]
    hands_actions = unit_actions[1:] if len(unit_actions) > 1 else []

    return farmer_action, hands_actions
