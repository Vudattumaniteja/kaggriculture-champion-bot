"""
Parameterized Heuristic Specialist Archetypes for Kaggriculture.

Implements the 5 specialist archetypes with high-performance chore management
and Manhattan pathfinding logic:
1. DeterministicGrandmaster (Balanced Grandmaster)
2. CarrotMonoculture (Carrot Cash Velocity)
3. MelonRusher (Melon Capital Payout)
4. DairySyndicate (Livestock & Dairy Tycoon)
5. TownShopSaturator (Town Shop Recipe Arbitrage)
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Set, Tuple

# Domain Constants & Catalogs
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

BASE_PRICES: Dict[str, int] = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
}

SHOPS_CATALOG: Dict[str, List[str]] = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

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


@dataclass
class SpecialistConfig:
    """Configuration parameters defining a specialist archetype."""
    name: str
    description: str
    target_geese: int = 2
    target_cows: int = 1
    target_sheep: int = 0
    max_coop_build_day: int = 10
    max_pasture_build_day: int = 12
    max_animal_buy_day: int = 18
    # Crop targets and preferences
    primary_crops: List[str] = field(default_factory=lambda: ["CARROT"])
    melon_cap: int = 6
    melon_max_day: int = 5
    wheat_cap: int = 8
    carrot_cap: int = 14
    tomato_cap: int = 0
    strawberry_cap: int = 0
    # Land & Labor
    unlock_ne_max_day: int = 16
    unlock_ne_min_money: int = 1200
    unlock_sw_max_day: int = 10
    unlock_sw_min_money: int = 2400
    hire_labor: bool = True
    hire_max_day: int = 28
    hire_daily_limit: int = 2
    # Shop Arbitrage & Reserves
    focus_town_shops: bool = True
    feed_reserve_multiplier: float = 3.0
    use_fertilizer: bool = True


# 1. Deterministic Grandmaster: Balanced Multi-Industry Farmer
ARCHETYPE_GRANDMASTER = SpecialistConfig(
    name="DeterministicGrandmaster",
    description="Balanced multi-industry grandmaster combining livestock, diversified crops, and town shop fulfillment.",
    target_geese=2,
    target_cows=1,
    target_sheep=0,
    max_coop_build_day=10,
    max_pasture_build_day=12,
    max_animal_buy_day=18,
    primary_crops=["MELON", "WHEAT", "CARROT"],
    melon_cap=6,
    melon_max_day=5,
    wheat_cap=8,
    carrot_cap=14,
    unlock_ne_max_day=16,
    unlock_ne_min_money=1200,
    unlock_sw_max_day=10,
    unlock_sw_min_money=2400,
    hire_labor=True,
    focus_town_shops=True,
    feed_reserve_multiplier=3.0,
    use_fertilizer=True,
)

# 2. Carrot Monoculture: High Cash-Velocity Crop Spammer
ARCHETYPE_CARROT_MONOCULTURE = SpecialistConfig(
    name="CarrotMonoculture",
    description="Fast-cycling monoculture specialist maximizing carrot planting acreage, fast harvesting, and quick market liquidation.",
    target_geese=0,
    target_cows=0,
    target_sheep=0,
    max_coop_build_day=0,
    max_pasture_build_day=0,
    max_animal_buy_day=0,
    primary_crops=["CARROT"],
    melon_cap=0,
    melon_max_day=0,
    wheat_cap=0,
    carrot_cap=24,
    tomato_cap=0,
    strawberry_cap=0,
    unlock_ne_max_day=18,
    unlock_ne_min_money=1100,
    unlock_sw_max_day=14,
    unlock_sw_min_money=2200,
    hire_labor=True,
    hire_max_day=28,
    hire_daily_limit=2,
    focus_town_shops=False,
    feed_reserve_multiplier=0.0,
    use_fertilizer=True,
)

# 3. Melon Rusher: High Payout Capital Accumulator
ARCHETYPE_MELON_RUSHER = SpecialistConfig(
    name="MelonRusher",
    description="Heavy melon investor dedicating tiles and labor to high-yield melon cycles for massive midgame and endgame cash spikes.",
    target_geese=0,
    target_cows=0,
    target_sheep=0,
    max_coop_build_day=0,
    max_pasture_build_day=0,
    max_animal_buy_day=0,
    primary_crops=["MELON", "CARROT"],
    melon_cap=16,
    melon_max_day=14,
    wheat_cap=0,
    carrot_cap=6,  # Small carrot buffer for early cash flow
    unlock_ne_max_day=16,
    unlock_ne_min_money=1150,
    unlock_sw_max_day=12,
    unlock_sw_min_money=2300,
    hire_labor=True,
    hire_max_day=28,
    hire_daily_limit=2,
    focus_town_shops=False,
    feed_reserve_multiplier=0.0,
    use_fertilizer=True,
)

# 4. Dairy Syndicate: Livestock & Milk Production Powerhouse
ARCHETYPE_DAIRY_SYNDICATE = SpecialistConfig(
    name="DairySyndicate",
    description="Livestock specialist prioritizing pastures, cow milking, sheep shearing, and dedicated wheat feed stockpiles.",
    target_geese=1,
    target_cows=3,
    target_sheep=1,
    max_coop_build_day=8,
    max_pasture_build_day=16,
    max_animal_buy_day=22,
    primary_crops=["WHEAT"],
    melon_cap=0,
    melon_max_day=0,
    wheat_cap=16,
    carrot_cap=4,
    unlock_ne_max_day=16,
    unlock_ne_min_money=1200,
    unlock_sw_max_day=12,
    unlock_sw_min_money=2400,
    hire_labor=True,
    hire_max_day=28,
    hire_daily_limit=2,
    focus_town_shops=True,
    feed_reserve_multiplier=4.0,
    use_fertilizer=True,
)

# 5. Town Shop Saturator: Dynamic Town Recipe Arbitrage
ARCHETYPE_TOWN_SHOP_SATURATOR = SpecialistConfig(
    name="TownShopSaturator",
    description="Town demand specialist dynamically tuning production to fill active shop recipes and harvest premium bonuses.",
    target_geese=1,
    target_cows=1,
    target_sheep=1,
    max_coop_build_day=10,
    max_pasture_build_day=14,
    max_animal_buy_day=20,
    primary_crops=["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
    melon_cap=2,
    melon_max_day=5,
    wheat_cap=10,
    carrot_cap=10,
    tomato_cap=4,
    strawberry_cap=4,
    unlock_ne_max_day=16,
    unlock_ne_min_money=1200,
    unlock_sw_max_day=10,
    unlock_sw_min_money=2400,
    hire_labor=True,
    hire_max_day=28,
    hire_daily_limit=2,
    focus_town_shops=True,
    feed_reserve_multiplier=3.0,
    use_fertilizer=True,
)

SPECIALIST_CONFIGS: Dict[str, SpecialistConfig] = {
    "DeterministicGrandmaster": ARCHETYPE_GRANDMASTER,
    "CarrotMonoculture": ARCHETYPE_CARROT_MONOCULTURE,
    "MelonRusher": ARCHETYPE_MELON_RUSHER,
    "DairySyndicate": ARCHETYPE_DAIRY_SYNDICATE,
    "TownShopSaturator": ARCHETYPE_TOWN_SHOP_SATURATOR,
}


class ParameterizedHeuristicFarmer:
    """
    High-performance heuristic bot parameterized by SpecialistConfig.
    Executes farm auditing, market dispatch, unit assignment, and pathfinding.
    """
    def __init__(self, config: Optional[SpecialistConfig] = None, personality: Optional[str] = None):
        if config is not None:
            self.cfg = config
        elif personality is not None and personality in SPECIALIST_CONFIGS:
            self.cfg = SPECIALIST_CONFIGS[personality]
        else:
            self.cfg = ARCHETYPE_GRANDMASTER

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        player = obs["player"]
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        private = obs.get("private", {}) or {}
        market = obs.get("market", {}) or {}
        town = obs.get("town", {}) or {}

        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        money = float(my_farm.get("money", 0.0))
        farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
        hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
        tiles = my_farm.get("tiles", [])
        unlocked = my_farm.get("unlocked_quadrants", ["NW"])
        hires_today = int(my_farm.get("hires_today", 0))
        shed = private.get("shed", {}) or {}
        seeds = private.get("seeds", {}) or {}
        inventories = private.get("inventories", []) or []
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

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                if isinstance(t, dict):
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

        # 2. Market Orders: Sell harvested goods
        for item in ["EGG", "MILK", "WOOL", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "FERTILIZER"]:
            count = shed.get(item, 0)
            if count > 0:
                if item == "FERTILIZER" and len(crop_tiles) > 0 and day <= 22 and count > 3 and self.cfg.use_fertilizer:
                    market_orders.append(["SELL", item, count - 3])
                elif item == "FERTILIZER" and not self.cfg.use_fertilizer:
                    market_orders.append(["SELL", item, count])
                elif item != "FERTILIZER":
                    market_orders.append(["SELL", item, count])

        # Wheat reserve management
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_req = total_living_animals
        wheat_safe_res = max(0, int(math.ceil(daily_feed_req * self.cfg.feed_reserve_multiplier)))
        if wheat_in_shed > wheat_safe_res + 8:
            market_orders.append(["SELL", "WHEAT", wheat_in_shed - wheat_safe_res])
        elif total_living_animals > 0 and wheat_in_shed < daily_feed_req and money >= 60 and day < 28:
            qty = min(6, max(1, wheat_safe_res - wheat_in_shed))
            market_orders.append(["BUY_PRODUCT", "WHEAT", qty])
            money -= qty * 25

        # Land Expansion
        if day <= self.cfg.unlock_ne_max_day and "NE" not in unlocked and money >= self.cfg.unlock_ne_min_money:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif day <= self.cfg.unlock_sw_max_day and "SW" not in unlocked and "NE" in unlocked and money >= self.cfg.unlock_sw_min_money:
            market_orders.append(["BUY_LAND"])
            money -= 2000

        # Labor Hiring
        if self.cfg.hire_labor and day < self.cfg.hire_max_day and hour <= 1 and hires_today < self.cfg.hire_daily_limit and money >= 40:
            market_orders.append(["HIRE"])
            money -= 2

        # Animal Purchasing
        if day <= self.cfg.max_animal_buy_day:
            if empty_coops > 0 and living_geese < self.cfg.target_geese and money >= 400:
                market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
                money -= 300
            elif empty_pastures > 0 and living_cows < self.cfg.target_cows and money >= 550:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400
            elif empty_pastures > 0 and living_sheep < self.cfg.target_sheep and money >= 450:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 350

        # Dynamic Seed Purchasing based on Specialist Config
        if day < 27 and money >= 50:
            seed_orders: List[Tuple[str, int]] = []

            # Melons
            if self.cfg.melon_cap > 0 and day <= self.cfg.melon_max_day:
                current_melon = seeds.get("MELON", 0) + planted_crops.get("MELON", 0)
                if current_melon < self.cfg.melon_cap and money >= 200:
                    needed = min(4, self.cfg.melon_cap - current_melon)
                    seed_orders.append(("MELON", needed))
                    money -= needed * 80

            # Wheat
            if self.cfg.wheat_cap > 0 and (total_living_animals > 0 or ("BAKERY" in unlocked_shops and self.cfg.focus_town_shops) or self.cfg.name == "DairySyndicate"):
                current_wheat = seeds.get("WHEAT", 0) + planted_crops.get("WHEAT", 0)
                if current_wheat < self.cfg.wheat_cap and money >= 60:
                    needed = min(6, self.cfg.wheat_cap - current_wheat)
                    seed_orders.append(("WHEAT", needed))
                    money -= needed * 10

            # Tomato (if town shop needs it)
            if self.cfg.tomato_cap > 0 and "PIZZA_SHOP" in unlocked_shops and day <= 15:
                current_tomato = seeds.get("TOMATO", 0) + planted_crops.get("TOMATO", 0)
                if current_tomato < self.cfg.tomato_cap and money >= 150:
                    needed = min(2, self.cfg.tomato_cap - current_tomato)
                    seed_orders.append(("TOMATO", needed))
                    money -= needed * 50

            # Strawberry (if town shop needs it)
            if self.cfg.strawberry_cap > 0 and any(s in unlocked_shops for s in ["SMOOTHIE_SHOP", "BRUNCH_SPOT", "ICE_CREAM_SHOP"]) and day <= 14:
                current_strawberry = seeds.get("STRAWBERRY", 0) + planted_crops.get("STRAWBERRY", 0)
                if current_strawberry < self.cfg.strawberry_cap and money >= 250:
                    needed = min(2, self.cfg.strawberry_cap - current_strawberry)
                    seed_orders.append(("STRAWBERRY", needed))
                    money -= needed * 100

            # Carrots (fill remaining capacity)
            if self.cfg.carrot_cap > 0:
                current_carrot = seeds.get("CARROT", 0) + planted_crops.get("CARROT", 0)
                if current_carrot < self.cfg.carrot_cap and money >= 80:
                    needed = min(8, self.cfg.carrot_cap - current_carrot)
                    seed_orders.append(("CARROT", needed))
                    money -= needed * 20

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
            elif self.cfg.use_fertilizer and ctile.get("fertilized_until_day", -1) < day:
                fertilize_crop_tasks.append(pos)

        # Structure and Planting task discovery
        total_structures = living_geese + empty_coops + living_cows + living_sheep + empty_pastures
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
                        elif shed.get("SHEEP", 0) > 0:
                            place_animal_tasks.append((c, r, "SHEEP"))
                elif t is None and pos not in SHED_TILES:
                    target_coops = self.cfg.target_geese
                    target_pastures = self.cfg.target_cows + self.cfg.target_sheep
                    if day <= self.cfg.max_coop_build_day and (living_geese + empty_coops) < target_coops and len(build_coop_tasks) == 0:
                        build_coop_tasks.append(pos)
                    elif day <= self.cfg.max_pasture_build_day and (living_cows + living_sheep + empty_pastures) < target_pastures and len(build_pasture_tasks) == 0:
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
                needed_an = "GOOSE" if u_tile.get("kind") == "COOP" else ("COW" if u_inv.get("COW", 0) > 0 else "SHEEP")
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
                if shed.get("SHEEP", 0) > 0 and empty_pastures > 0 and u_inv.get("SHEEP", 0) == 0:
                    unit_actions.append(["PICKUP", "SHEEP", 1])
                    shed["SHEEP"] -= 1
                    u_inv["SHEEP"] = 1
                    continue
                if self.cfg.use_fertilizer and fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    unit_actions.append(["PICKUP", "FERTILIZER", qty])
                    shed["FERTILIZER"] -= qty
                    u_inv["FERTILIZER"] = qty
                    continue

            # 3. Actions on current standing tile
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
                    if self.cfg.use_fertilizer and u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0:
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue
                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue
            elif u_tile is None and u_pos not in SHED_TILES:
                if u_pos in build_coop_tasks and day <= self.cfg.max_coop_build_day:
                    unit_actions.append(["BUILD_COOP"])
                    build_coop_tasks.remove(u_pos)
                    continue
                if u_pos in build_pasture_tasks and day <= self.cfg.max_pasture_build_day:
                    unit_actions.append(["BUILD_PASTURE"])
                    build_pasture_tasks.remove(u_pos)
                    continue
                if day < 27 and u_pos in plant_tasks:
                    best_crop = "CARROT"
                    # Priority order matching specialist archetype
                    if seeds.get("MELON", 0) > 0 and self.cfg.melon_cap > 0 and day <= self.cfg.melon_max_day:
                        best_crop = "MELON"
                    elif seeds.get("TOMATO", 0) > 0 and self.cfg.tomato_cap > 0:
                        best_crop = "TOMATO"
                    elif seeds.get("STRAWBERRY", 0) > 0 and self.cfg.strawberry_cap > 0:
                        best_crop = "STRAWBERRY"
                    elif seeds.get("WHEAT", 0) > 0 and (total_living_animals > 0 or ("BAKERY" in unlocked_shops and self.cfg.focus_town_shops) or self.cfg.name == "DairySyndicate"):
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


def get_specialist(personality: str) -> ParameterizedHeuristicFarmer:
    """Factory helper to obtain a configured specialist instance."""
    cfg = SPECIALIST_CONFIGS.get(personality, ARCHETYPE_GRANDMASTER)
    return ParameterizedHeuristicFarmer(config=cfg)


# Standalone Agent Callables
_agent_gm = ParameterizedHeuristicFarmer(ARCHETYPE_GRANDMASTER)
_agent_carrot = ParameterizedHeuristicFarmer(ARCHETYPE_CARROT_MONOCULTURE)
_agent_melon = ParameterizedHeuristicFarmer(ARCHETYPE_MELON_RUSHER)
_agent_dairy = ParameterizedHeuristicFarmer(ARCHETYPE_DAIRY_SYNDICATE)
_agent_town = ParameterizedHeuristicFarmer(ARCHETYPE_TOWN_SHOP_SATURATOR)

def agent_grandmaster(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _agent_gm(obs, config)

def agent_carrot_monoculture(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _agent_carrot(obs, config)

def agent_melon_rusher(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _agent_melon(obs, config)

def agent_dairy_syndicate(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _agent_dairy(obs, config)

def agent_town_shop_saturator(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _agent_town(obs, config)
