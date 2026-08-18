"""
Multi-Industry Crop Farming baseline agent.
Covers full crop repertoire: Carrots, Wheat, Tomatoes, Strawberries, Melons.
Orchestrates rotational planting, quadrant land expansions, farmhand chore division,
daily watering schedules, fertilizer application, and peak-maturity harvesting.
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


class CropFarmerAgent:
    def __init__(self, mode: str = "PORTFOLIO", target_active_tiles: int = 24):
        self.mode = mode  # "PORTFOLIO", "CARROT", "MELON", "STRAWBERRY", "TOMATO", "WHEAT"
        self.target_active_tiles = target_active_tiles

    def _select_target_crops(self, day: int, money: float) -> List[Tuple[str, int]]:
        """Returns a prioritized list of (crop_name, max_count) to plant based on current day and capital."""
        if self.mode == "CARROT":
            return [("CARROT", 30)] if day < 27 else []
        elif self.mode == "WHEAT":
            return [("WHEAT", 30)] if day < 26 else []
        elif self.mode == "MELON":
            if day <= 17:
                return [("MELON", 20), ("CARROT", 10)]
            elif day < 27:
                return [("CARROT", 30)]
            return []
        elif self.mode == "STRAWBERRY":
            if day <= 18:
                return [("STRAWBERRY", 15), ("CARROT", 10)]
            elif day < 27:
                return [("CARROT", 30)]
            return []
        elif self.mode == "TOMATO":
            if day <= 20:
                return [("TOMATO", 15), ("CARROT", 10)]
            elif day < 27:
                return [("CARROT", 30)]
            return []

        # Default: DIVERSIFIED / PORTFOLIO
        # Season opening (Days 0-6): Melons + Strawberries + Carrots for fast cash
        if day <= 6:
            return [("MELON", 8), ("STRAWBERRY", 6), ("CARROT", 10), ("WHEAT", 4)]
        # Early-Mid season (Days 7-14): Second wave Melons + Tomatoes + Carrots
        elif day <= 15:
            return [("MELON", 6), ("TOMATO", 8), ("CARROT", 12), ("WHEAT", 4)]
        # Mid-Late season (Days 16-23): Fast turnaround Carrots + Wheat
        elif day <= 23:
            return [("CARROT", 18), ("WHEAT", 6)]
        # Final sprint (Days 24-26): Fast 3-day Carrots
        elif day <= 26:
            return [("CARROT", 20)]
        # Liquidation (Days 27-29): No planting
        return []

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

        market_orders: List[List[Any]] = []

        # --- 1. MARKET STRATEGY ---
        # A. Sell all harvested goods from shed immediately
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                # Keep up to 3 fertilizer for our high-value plants if in early/mid season
                if item == "FERTILIZER" and day <= 22 and count > 3:
                    market_orders.append(["SELL", item, count - 3])
                elif item != "FERTILIZER":
                    market_orders.append(["SELL", item, count])

        # B. Land Expansion: Reinvest early profits to scale up farm capacity
        if day <= 16:
            if "NE" not in unlocked and money >= 1200:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and money >= 2400 and day <= 12:
                market_orders.append(["BUY_LAND"])
                money -= 2000

        # C. Labor Hiring: Recruit 2 farmhands daily for $2 total cost
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])
            money -= 2

        # D. Seed Purchasing Portfolio
        target_crop_alloc = self._select_target_crops(day, money)
        planted_counts: Dict[str, int] = {c: 0 for c in CROP_SPECS}
        empty_tiles_count = 0

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                if t is None:
                    empty_tiles_count += 1
                elif isinstance(t, dict) and t.get("kind") == "PLANT":
                    crop_name = t.get("crop", "CARROT")
                    planted_counts[crop_name] = planted_counts.get(crop_name, 0) + 1

        total_seeds_in_pocket = sum(seeds.values())
        max_desired_active = min(len(unlocked) * 16, self.target_active_tiles)

        for crop_name, max_cnt in target_crop_alloc:
            current_planted = planted_counts.get(crop_name, 0)
            current_seeds = seeds.get(crop_name, 0)
            needed = max(0, max_cnt - current_planted - current_seeds)
            seed_price = CROP_SPECS[crop_name]["seed"]

            if needed > 0 and money >= seed_price * 2 and day < 27:
                qty = min(needed, 4, int(money // seed_price))
                if qty > 0:
                    market_orders.append(["BUY_SEED", crop_name, qty])
                    money -= seed_price * qty
                    seeds[crop_name] = current_seeds + qty

        # --- 2. FIELD SCANNING & CHORE PLANNING ---
        water_tasks: List[Tuple[int, int]] = []
        harvest_tasks: List[Tuple[int, int]] = []
        fertilize_tasks: List[Tuple[int, int]] = []
        plant_tasks: List[Tuple[int, int]] = []
        weed_tasks: List[Tuple[int, int]] = []
        unlocked_empty_tiles: List[Tuple[int, int]] = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    unlocked_empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        crop = t.get("crop", "CARROT")
                        cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                        age = day - t.get("planted_day", day)
                        is_ongoing = cspec["ongoing"]
                        max_age = cspec["max_yield_day"]
                        first_yield = cspec["first_yield_day"]
                        yield_u = t.get("yield_units", 0)

                        # Harvest condition:
                        # Non-ongoing: age reaches max_yield_day, or day >= 28 with yield
                        # Ongoing: age >= first_yield_day and yield_units > 0
                        if not is_ongoing:
                            if age >= max_age or (day >= 28 and yield_u > 0):
                                harvest_tasks.append(pos)
                            elif not t.get("watered_today", False):
                                water_tasks.append(pos)
                            elif t.get("fertilized_until_day", -1) < day and crop in ["MELON", "TOMATO", "STRAWBERRY"]:
                                fertilize_tasks.append(pos)
                        else:
                            if age >= first_yield and yield_u > 0:
                                harvest_tasks.append(pos)
                            elif not t.get("watered_today", False):
                                water_tasks.append(pos)
                            elif t.get("fertilized_until_day", -1) < day:
                                fertilize_tasks.append(pos)

                    elif k == "WEED":
                        weed_tasks.append(pos)

        # Planting allocation on closest empty tiles to shed
        unlocked_empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        total_seeds_available = sum(seeds.values())
        if day < 27 and total_seeds_available > 0:
            for p in unlocked_empty_tiles[:max_desired_active]:
                plant_tasks.append(p)

        # Chore priority: WATER (urgent!) > HARVEST > FERTILIZE > PLANT > WEED
        priority_tasks = water_tasks + harvest_tasks + fertilize_tasks + plant_tasks + weed_tasks
        claimed_tasks: Set[Tuple[int, int]] = set()

        # --- 3. MULTI-WORKER EXECUTION ---
        units = [farmer_pos] + hands_pos
        unit_actions: List[List[Any]] = []

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in SHED_TILES

            # Shed pickup for fertilizer if crops need it
            if is_at_shed and fertilize_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                qty = min(2, shed.get("FERTILIZER", 0))
                unit_actions.append(["PICKUP", "FERTILIZER", qty])
                shed["FERTILIZER"] -= qty
                u_inv["FERTILIZER"] = qty
                continue

            # 1. Action on current standing tile
            if isinstance(u_tile, dict) and u_tile.get("kind") == "PLANT":
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

            elif isinstance(u_tile, dict) and u_tile.get("kind") == "WEED":
                unit_actions.append(["DIG"])
                continue

            elif u_tile is None and day < 27 and u_pos in plant_tasks:
                # Pick best available seed to plant
                for crop_candidate, _ in target_crop_alloc:
                    if seeds.get(crop_candidate, 0) > 0:
                        unit_actions.append(["PLANT", crop_candidate])
                        seeds[crop_candidate] -= 1
                        break
                else:
                    for crop_candidate in ["CARROT", "WHEAT", "TOMATO", "STRAWBERRY", "MELON"]:
                        if seeds.get(crop_candidate, 0) > 0:
                            unit_actions.append(["PLANT", crop_candidate])
                            seeds[crop_candidate] -= 1
                            break
                    else:
                        unit_actions.append(["PASS"])
                continue

            # 2. Pathfind to nearest unclaimed chore
            best_target = None
            best_dist = 999

            for task in priority_tasks:
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


# Default agent instances
crop_farmer_portfolio = CropFarmerAgent(mode="PORTFOLIO")
crop_farmer_carrot = CropFarmerAgent(mode="CARROT")
crop_farmer_melon = CropFarmerAgent(mode="MELON")
crop_farmer_strawberry = CropFarmerAgent(mode="STRAWBERRY")
crop_farmer_tomato = CropFarmerAgent(mode="TOMATO")
crop_farmer_wheat = CropFarmerAgent(mode="WHEAT")


def agent(obs, config=None):
    return crop_farmer_portfolio(obs, config)
