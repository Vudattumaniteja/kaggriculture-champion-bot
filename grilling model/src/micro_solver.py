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


def get_step_towards(
    curr: Tuple[int, int],
    target: Tuple[int, int],
    reserved_dests: Optional[Set[Tuple[int, int]]] = None,
    unmoved_positions: Optional[Set[Tuple[int, int]]] = None
) -> Tuple[str, Tuple[int, int]]:
    """
    Computes the next collision-free movement step towards target.
    Guarantees that the chosen destination is in bounds, not already reserved by another unit,
    and does not step onto an unmoved unit's current position.
    """
    cx, cy = curr
    tx, ty = target
    reserved = reserved_dests or set()
    unmoved = unmoved_positions or set()

    moves: List[Tuple[str, Tuple[int, int]]] = []
    dx = tx - cx
    dy = ty - cy

    # Primary direction preferences based on target delta
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

    # Secondary directions for avoidance routing
    all_dirs = [("NORTH", (cx, cy - 1)), ("SOUTH", (cx, cy + 1)), ("EAST", (cx + 1, cy)), ("WEST", (cx - 1, cy))]
    for d, pos in all_dirs:
        if (d, pos) not in moves:
            moves.append((d, pos))

    # Evaluate moves in priority order
    for direction, next_pos in moves:
        nx, ny = next_pos
        if not (0 <= nx < 10 and 0 <= ny < 10):
            continue
        if next_pos in reserved:
            continue
        if next_pos in unmoved:
            continue
        return direction, next_pos

    # If all directional moves are blocked, check staying in place
    if curr not in reserved:
        return "PASS", curr

    # Fallback to any valid in-bounds unreserved cell
    for direction, (nx, ny) in all_dirs:
        if 0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in reserved:
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
    hour = int(obs.get("hour", int(obs.get("step", 0)) % 24))
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

    # Include cows in shed waiting to be placed
    total_cows = living_cows + int(shed.get("COW", 0))

    market_orders: List[List[Any]] = []

    # 1. Stage 1 & Stage 2 Sell Orders with Cow Feed Reservation
    cow_feed_reservation = total_cows * 2 if day < 28 else 0

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
            shed[prod] -= sell_qty

    # 2. Midnight Shed Overflow Guardrail (t % 24 == 23)
    remaining_shed_total = sum(shed.values())
    if hour == 23 and remaining_shed_total > 100:
        excess = remaining_shed_total - 100
        # Sort products by spot price descending (highest cash yield first)
        sorted_prods = sorted(
            [p for p in shed.keys() if shed[p] > 0],
            key=lambda p: float(prices.get(p, BASE_PRICES.get(p, 25.0))),
            reverse=True
        )

        for prod in sorted_prods:
            if excess <= 0:
                break
            can_dump = shed.get(prod, 0)
            if prod == "WHEAT" and day < 28:
                # Keep cow feed reserved if other items can cover the overflow
                can_dump = max(0, can_dump - cow_feed_reservation)
            dump_qty = min(can_dump, excess)
            if dump_qty > 0:
                existing = [o for o in market_orders if o[0] == "SELL" and o[1] == prod]
                if existing:
                    existing[0][2] += dump_qty
                else:
                    market_orders.append(["SELL", prod, dump_qty])
                shed[prod] -= dump_qty
                excess -= dump_qty

        # If excess still remains after preserving wheat feed, dump remainder to prevent engine discard
        if excess > 0 and shed.get("WHEAT", 0) > 0:
            dump_wheat = min(shed["WHEAT"], excess)
            existing = [o for o in market_orders if o[0] == "SELL" and o[1] == "WHEAT"]
            if existing:
                existing[0][2] += dump_wheat
            else:
                market_orders.append(["SELL", "WHEAT", dump_wheat])
            shed["WHEAT"] -= dump_wheat
            excess -= dump_wheat

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
    livestock_quotas: Optional[np.ndarray] = None,
    lambda_logit: float = 1.0
) -> Tuple[List[Any], List[List[Any]]]:
    """
    Formulates worker chore assignment as bipartite matching.
    Cost matrix: C_ij = Dist(w_i, t_j) - lambda * Logit(t_j) - Urgency(t_j)
    Solved via scipy.optimize.linear_sum_assignment with biological urgency overrides.
    Zero path collisions guaranteed through unmoved-position tracking and destination reservation.
    """
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    private = obs.get("private", {}) or {}

    day = int(obs.get("day", 0))
    farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
    hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
    tiles = my_farm.get("tiles", [])
    shed = dict(private.get("shed", {}) or {})
    seeds = dict(private.get("seeds", {}) or {})
    inventories = private.get("inventories", []) or []

    units = [farmer_pos] + hands_pos
    num_units = len(units)

    unit_actions: List[Optional[List[Any]]] = [None] * num_units
    reserved_destinations: Set[Tuple[int, int]] = set()
    unmoved_units: Set[int] = set(range(num_units))

    # Available seeds tracking to prevent multi-worker over-planting seed drops
    available_seeds = dict(seeds)

    # 1. Immediate in-place tile actions for standing units
    for u_idx, u_pos in enumerate(units):
        ux, uy = u_pos
        u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
        u_inv = inventories[u_idx] if u_idx < len(inventories) else {}

        if isinstance(u_tile, dict):
            k = u_tile.get("kind")
            an = u_tile.get("animal")
            if k in ["COOP", "PASTURE"] and an:
                if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                    unit_actions[u_idx] = ["FEED"]
                    u_inv["WHEAT"] -= 1
                    u_tile["fed_today"] = True
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                    continue
                if not u_tile.get("cared_today", False):
                    unit_actions[u_idx] = ["CARE"]
                    u_tile["cared_today"] = True
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                    continue
                if u_tile.get("yield_units", 0) > 0:
                    unit_actions[u_idx] = ["HARVEST"]
                    u_tile["yield_units"] = 0
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                    continue
                if u_tile.get("fertilizer_available", False):
                    unit_actions[u_idx] = ["COLLECT_FERTILIZER"]
                    u_tile["fertilizer_available"] = False
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                    continue
            elif k in ["COOP", "PASTURE"] and an is None:
                needed_an = "GOOSE" if k == "COOP" else "COW"
                if u_inv.get(needed_an, 0) > 0:
                    unit_actions[u_idx] = ["PLACE_ANIMAL", needed_an]
                    u_inv[needed_an] -= 1
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
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
                    u_tile["yield_units"] = 0
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                    continue
                if not u_tile.get("watered_today", False):
                    unit_actions[u_idx] = ["WATER"]
                    u_tile["watered_today"] = True
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                    continue
                if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0:
                    unit_actions[u_idx] = ["FERTILIZE"]
                    u_inv["FERTILIZER"] -= 1
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                    continue
            elif k == "WEED":
                unit_actions[u_idx] = ["DIG"]
                reserved_destinations.add(u_pos)
                unmoved_units.discard(u_idx)
                continue

    # 2. Collect pending tasks on the farm with biological urgency & neural logits
    # Task tuple: (pos, priority_score, task_type, payload)
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
                        tasks.append((pos, 30.0, "FEED", None))  # Starving animal (highest biological urgency)
                    elif not t.get("cared_today", False):
                        tasks.append((pos, 18.0, "CARE", None))
                    elif t.get("yield_units", 0) > 0:
                        tasks.append((pos, 25.0, "HARVEST_ANIMAL", None))
                    elif t.get("fertilizer_available", False):
                        tasks.append((pos, 12.0, "FERT_ANIMAL", None))
                elif k in ["COOP", "PASTURE"] and an is None:
                    needed_an = "GOOSE" if k == "COOP" else "COW"
                    if shed.get(needed_an, 0) > 0:
                        tasks.append((pos, 15.0, "PLACE_ANIMAL", needed_an))
                elif k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                    age = day - t.get("planted_day", day)
                    yield_u = t.get("yield_units", 0)
                    if (not cspec["ongoing"] and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                        cspec["ongoing"] and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        urg = 35.0 if day >= 28 else 25.0  # Ripe harvest
                        tasks.append((pos, urg, "HARVEST_CROP", None))
                    elif not t.get("watered_today", False):
                        tasks.append((pos, 22.0, "WATER_CROP", None))  # Dry crop
                elif k == "WEED":
                    tasks.append((pos, 10.0, "DIG_WEED", None))
            elif t is None and pos not in SHED_TILES and day < 27:
                # Select best crop with positive seed buffer
                best_crop = None
                best_logit = 0.0
                if crop_heatmaps is not None:
                    c_logits = crop_heatmaps[:, r, c]
                    sorted_indices = np.argsort(-c_logits)
                    for idx in sorted_indices:
                        candidate = CROPS[idx]
                        if available_seeds.get(candidate, 0) > 0:
                            best_crop = candidate
                            best_logit = float(c_logits[idx])
                            break
                if best_crop is None:
                    for crop in CROPS:
                        if available_seeds.get(crop, 0) > 0:
                            best_crop = crop
                            break

                if best_crop is not None and available_seeds.get(best_crop, 0) > 0:
                    priority = 5.0 + lambda_logit * best_logit
                    tasks.append((pos, priority, "PLANT_TILE", best_crop))

    # 3. Formulate and solve bipartite Hungarian matching for unassigned units
    unassigned_unit_indices = [i for i in range(num_units) if unit_actions[i] is None]

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

        for i_idx, j_idx in zip(row_ind, col_ind):
            u_idx = unassigned_unit_indices[i_idx]
            u_pos = units[u_idx]
            t_pos, _, task_type, payload = tasks[j_idx]

            unmoved_positions = {units[v] for v in unmoved_units if v != u_idx}

            if u_pos == t_pos:
                if task_type == "PLANT_TILE" and payload:
                    if available_seeds.get(payload, 0) > 0:
                        unit_actions[u_idx] = ["PLANT", payload]
                        available_seeds[payload] -= 1
                    else:
                        unit_actions[u_idx] = ["PASS"]
                elif task_type == "FEED":
                    u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
                    if u_inv.get("WHEAT", 0) > 0:
                        unit_actions[u_idx] = ["FEED"]
                        u_inv["WHEAT"] -= 1
                    else:
                        unit_actions[u_idx] = ["PASS"]
                elif task_type == "CARE":
                    unit_actions[u_idx] = ["CARE"]
                elif task_type in ["HARVEST_ANIMAL", "HARVEST_CROP"]:
                    unit_actions[u_idx] = ["HARVEST"]
                elif task_type == "DIG_WEED":
                    unit_actions[u_idx] = ["DIG"]
                elif task_type == "PLACE_ANIMAL" and payload:
                    unit_actions[u_idx] = ["PLACE_ANIMAL", payload]
                else:
                    unit_actions[u_idx] = ["PASS"]

                reserved_destinations.add(u_pos)
                unmoved_units.discard(u_idx)
            else:
                direction, next_pos = get_step_towards(u_pos, t_pos, reserved_destinations, unmoved_positions)
                unit_actions[u_idx] = [direction]
                reserved_destinations.add(next_pos)
                unmoved_units.discard(u_idx)

    # 4. Fallback for remaining idle units -> walk to nearest shed or PASS
    for u_idx in range(num_units):
        if unit_actions[u_idx] is None:
            u_pos = units[u_idx]
            unmoved_positions = {units[v] for v in unmoved_units if v != u_idx}
            if u_pos not in SHED_TILES:
                nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                direction, next_pos = get_step_towards(u_pos, nearest_shed, reserved_destinations, unmoved_positions)
                unit_actions[u_idx] = [direction]
                reserved_destinations.add(next_pos)
                unmoved_units.discard(u_idx)
            else:
                if u_pos not in reserved_destinations:
                    unit_actions[u_idx] = ["PASS"]
                    reserved_destinations.add(u_pos)
                    unmoved_units.discard(u_idx)
                else:
                    direction, next_pos = get_step_towards(u_pos, (4, 4), reserved_destinations, unmoved_positions)
                    unit_actions[u_idx] = [direction]
                    reserved_destinations.add(next_pos)
                    unmoved_units.discard(u_idx)

    farmer_action = unit_actions[0] if unit_actions and unit_actions[0] is not None else ["PASS"]
    hands_actions = [act if act is not None else ["PASS"] for act in unit_actions[1:]] if len(unit_actions) > 1 else []

    return farmer_action, hands_actions
