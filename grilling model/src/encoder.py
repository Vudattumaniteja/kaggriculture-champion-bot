"""
Observation State Feature Encoder for Kaggriculture Unified Champion Bot.
Extracts 24-channel spatial feature tensor and 72-dim scalar economic state vector.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import math
import numpy as np

# Canonical catalogs
CROPS: List[str] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS: List[str] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS: List[str] = ["GOOSE", "COW", "SHEEP"]

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

BASE_INVENTORY: Dict[str, float] = {
    "WHEAT": 100.0, "CARROT": 100.0, "TOMATO": 100.0, "STRAWBERRY": 100.0, "MELON": 100.0,
    "EGG": 100.0, "MILK": 100.0, "WOOL": 100.0, "FERTILIZER": 100.0,
}

SHOPS_CATALOG: List[str] = [
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"
]

SHED_TILES: List[Tuple[int, int]] = [(4, 4), (5, 4), (4, 5), (5, 5)]

SPATIAL_CHANNELS = 24
SCALAR_DIM = 72


def symlog(x: Any) -> float:
    """Symmetric logarithm transformation: sign(x) * ln(|x| + 1)."""
    try:
        x_f = float(x)
        if math.isnan(x_f) or math.isinf(x_f):
            return 0.0
        return math.copysign(math.log1p(abs(x_f)), x_f)
    except (ValueError, TypeError, OverflowError):
        return 0.0


def get_quadrant(r: int, c: int) -> str:
    """Returns NW, NE, SW, or SE for tile coordinates (r, c)."""
    if r < 5:
        return "NW" if c < 5 else "NE"
    else:
        return "SW" if c < 5 else "SE"


def encode_observation(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes raw Kaggriculture environment observation into:
    1. Spatial grid tensor: Shape (24, 10, 10), dtype float32
    2. Global economic scalar vector: Shape (72,), dtype float32
    """
    if not isinstance(obs, dict):
        obs = {}

    try:
        player = int(obs.get("player", 0) or 0)
    except (ValueError, TypeError):
        player = 0

    farms = obs.get("farms", [{}, {}])
    if not isinstance(farms, (list, tuple)):
        farms = [{}, {}]

    my_farm = farms[player] if (0 <= player < len(farms) and isinstance(farms[player], dict)) else {}
    opp_idx = 1 - player
    opp_farm = farms[opp_idx] if (0 <= opp_idx < len(farms) and isinstance(farms[opp_idx], dict)) else {}

    private = obs.get("private", {})
    if not isinstance(private, dict):
        private = {}

    market = obs.get("market", {})
    if not isinstance(market, dict):
        market = {}

    town = obs.get("town", {})
    if not isinstance(town, dict):
        town = {}

    try:
        step = float(obs.get("step", 0) or 0)
    except (ValueError, TypeError):
        step = 0.0

    try:
        day = float(obs.get("day", 0) or 0)
    except (ValueError, TypeError):
        day = 0.0

    try:
        hour = float(obs.get("hour", 0) or 0)
    except (ValueError, TypeError):
        hour = 0.0

    try:
        my_money = float(my_farm.get("money", 3000.0) if my_farm.get("money") is not None else 3000.0)
    except (ValueError, TypeError):
        my_money = 3000.0

    try:
        opp_money = float(opp_farm.get("money", 3000.0) if opp_farm.get("money") is not None else 3000.0)
    except (ValueError, TypeError):
        opp_money = 3000.0

    tiles = my_farm.get("tiles", [])
    if not isinstance(tiles, (list, tuple)):
        tiles = []

    unlocked_quads_raw = my_farm.get("unlocked_quadrants", ["NW"])
    unlocked_quads: Set[str] = set(unlocked_quads_raw) if isinstance(unlocked_quads_raw, (list, tuple, set)) else {"NW"}

    opp_unlocked_quads_raw = opp_farm.get("unlocked_quadrants", ["NW"])
    opp_unlocked_quads: Set[str] = set(opp_unlocked_quads_raw) if isinstance(opp_unlocked_quads_raw, (list, tuple, set)) else {"NW"}

    farmer_pos = my_farm.get("farmer", [4, 4])
    hands_pos = my_farm.get("hands", [])

    # 1. Build Spatial Tensor (24, 10, 10)
    spatial = np.zeros((SPATIAL_CHANNELS, 10, 10), dtype=np.float32)

    # Static invariant geometric fields
    for r in range(10):
        for c in range(10):
            # Channel 16: Shed footprint
            if (r in (4, 5)) and (c in (4, 5)):
                spatial[16, r, c] = 1.0

            # Channel 17: Town Delivery / Boundary Zone footprint
            if r in (0, 9) or c in (0, 9):
                spatial[17, r, c] = 0.5
            if (r in (4, 5)) and (c in (4, 5)):
                spatial[17, r, c] = 1.0

            # Channel 19: Quadrant Cost Map (NW: 0.0, NE: 0.25, SW: 0.50, SE: 1.00)
            q = get_quadrant(r, c)
            if q == "NW":
                spatial[19, r, c] = 0.0
            elif q == "NE":
                spatial[19, r, c] = 0.25
            elif q == "SW":
                spatial[19, r, c] = 0.50
            else:  # SE
                spatial[19, r, c] = 1.00

            # Channel 20: Distance to Shed (normalized)
            min_shed_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in [(4, 4), (4, 5), (5, 4), (5, 5)])
            spatial[20, r, c] = float(min_shed_dist) / 18.0

            # Channel 21: Distance to Shop / Town Center
            spatial[21, r, c] = (abs(r - 4.5) + abs(c - 4.5)) / 10.0

            # Channel 22: Shop Delta Row
            spatial[22, r, c] = (r - 4.5) / 5.0

            # Channel 23: Unlock Status Map
            spatial[23, r, c] = 1.0 if q in unlocked_quads else 0.0

    # Parse dynamic tile contents
    planted_counts: Dict[str, int] = {c: 0 for c in CROPS}
    animal_counts: Dict[str, int] = {a: 0 for a in ANIMALS}

    for r in range(min(10, len(tiles))):
        row = tiles[r]
        if not isinstance(row, (list, tuple)):
            continue
        for c in range(min(10, len(row))):
            t = row[c]
            if t == "LOCKED":
                continue

            q = get_quadrant(r, c)
            if t is None:
                if q in unlocked_quads and (r, c) not in [(4, 4), (5, 4), (4, 5), (5, 5)]:
                    spatial[10, r, c] = 1.0  # Empty unlocked tile
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    crop = t.get("crop", "CARROT")
                    if crop in CROPS:
                        crop_idx = CROPS.index(crop)
                        spatial[crop_idx, r, c] = 1.0  # Channels 0..4: One-hot crop
                        planted_counts[crop] = planted_counts.get(crop, 0) + 1

                    try:
                        planted_day = float(t.get("planted_day", day) or day)
                    except (ValueError, TypeError):
                        planted_day = day
                    age = max(0.0, day - planted_day)
                    spatial[5, r, c] = min(age / 12.0, 1.0)  # Channel 5: Growth ratio
                    spatial[6, r, c] = 1.0 if t.get("watered_today", False) else 0.0  # Channel 6: Moisture

                    try:
                        fert_until = float(t.get("fertilized_until_day", -1) or -1)
                    except (ValueError, TypeError):
                        fert_until = -1.0
                    spatial[7, r, c] = 1.0 if fert_until >= day else 0.0  # Channel 7: Fertilizer active

                elif k == "COOP":
                    spatial[8, r, c] = 1.0  # Channel 8: Coop
                    an = t.get("animal")
                    if an in ANIMALS:
                        spatial[11 + ANIMALS.index(an), r, c] = 1.0  # Channels 11..13
                        animal_counts[an] = animal_counts.get(an, 0) + 1
                        spatial[14, r, c] = 0.0 if t.get("fed_today", False) else 1.0  # Channel 14: Hunger
                        try:
                            y_units = float(t.get("yield_units", 0) or 0)
                        except (ValueError, TypeError):
                            y_units = 0.0
                        spatial[15, r, c] = min(max(y_units, 0.0) / 4.0, 1.0)  # Product ready
                    else:
                        spatial[10, r, c] = 0.5  # Empty coop

                elif k == "PASTURE":
                    spatial[9, r, c] = 1.0  # Channel 9: Pasture
                    an = t.get("animal")
                    if an in ANIMALS:
                        spatial[11 + ANIMALS.index(an), r, c] = 1.0  # Channels 11..13
                        animal_counts[an] = animal_counts.get(an, 0) + 1
                        spatial[14, r, c] = 0.0 if t.get("fed_today", False) else 1.0  # Channel 14: Hunger
                        try:
                            y_units = float(t.get("yield_units", 0) or 0)
                        except (ValueError, TypeError):
                            y_units = 0.0
                        spatial[15, r, c] = min(max(y_units, 0.0) / 4.0, 1.0)  # Product ready
                    else:
                        spatial[10, r, c] = 0.5  # Empty pasture

    # Worker density map (Channel 18)
    if isinstance(farmer_pos, (list, tuple)) and len(farmer_pos) >= 2:
        try:
            fx, fy = int(farmer_pos[0]), int(farmer_pos[1])
            if 0 <= fy < 10 and 0 <= fx < 10:
                spatial[18, fy, fx] += 1.0
        except (ValueError, TypeError):
            pass

    if isinstance(hands_pos, list):
        for h in hands_pos:
            if isinstance(h, (list, tuple)) and len(h) >= 2:
                try:
                    hx, hy = int(h[0]), int(h[1])
                    if 0 <= hy < 10 and 0 <= hx < 10:
                        spatial[18, hy, hx] += 1.0
                except (ValueError, TypeError):
                    pass

    # 2. Build Scalar Economic Vector (72 dims)
    scalar = np.zeros(SCALAR_DIM, dtype=np.float32)

    # 0..2: Turn Clocks
    scalar[0] = step / 720.0
    scalar[1] = hour / 24.0
    scalar[2] = day / 30.0

    # 3..5: Liquid Cash & Velocity
    scalar[3] = symlog(my_money) / 15.0
    scalar[4] = symlog(opp_money) / 15.0
    scalar[5] = float(np.clip((my_money - opp_money) / 10000.0, -10.0, 10.0))

    # 6..7: Labor Liabilities & Hires
    num_hands = len(hands_pos) if isinstance(hands_pos, list) else 0
    scalar[6] = ((num_hands + 1) * 20.0) / 260.0
    try:
        hires_today = float(my_farm.get("hires_today", 0) or 0)
    except (ValueError, TypeError):
        hires_today = 0.0
    scalar[7] = hires_today / 5.0

    # 8..10: Quadrant Status
    scalar[8] = float(len(unlocked_quads)) / 4.0
    scalar[9] = 1.0 if "NE" in unlocked_quads else 0.0
    scalar[10] = 1.0 if "SW" in unlocked_quads else 0.0

    # 11..19: Spot Prices (9 dims)
    prices = market.get("prices", {}) if isinstance(market.get("prices"), dict) else {}
    for i, prod in enumerate(PRODUCTS):
        try:
            p_val = float(prices.get(prod, BASE_PRICES.get(prod, 25.0)))
        except (ValueError, TypeError):
            p_val = BASE_PRICES.get(prod, 25.0)
        scalar[11 + i] = min(max(p_val, 0.0) / 500.0, 2.0)

    # 20..28: Price / Base Price Ratios (9 dims)
    for i, prod in enumerate(PRODUCTS):
        try:
            p_val = float(prices.get(prod, BASE_PRICES.get(prod, 25.0)))
        except (ValueError, TypeError):
            p_val = BASE_PRICES.get(prod, 25.0)
        p_base = BASE_PRICES.get(prod, 25.0)
        scalar[20 + i] = min(max(p_val, 0.0) / max(p_base, 1.0), 5.0)

    # 29..37: Market Inventory Elasticity / Crash Ratios (9 dims)
    mkt_inv = market.get("inventory", {}) if isinstance(market.get("inventory"), dict) else {}
    for i, prod in enumerate(PRODUCTS):
        try:
            cur_inv = float(mkt_inv.get(prod, BASE_INVENTORY.get(prod, 100.0)))
        except (ValueError, TypeError):
            cur_inv = BASE_INVENTORY.get(prod, 100.0)
        base_inv = BASE_INVENTORY.get(prod, 100.0)
        scalar[29 + i] = min(max(cur_inv, 0.0) / max(base_inv, 1.0), 5.0)

    # 38..45: Town Shop Demands / Active Shops (8 dims)
    unlocked_shops_raw = town.get("unlocked_shops", [])
    unlocked_shops = set(unlocked_shops_raw) if isinstance(unlocked_shops_raw, (list, tuple, set)) else set()
    for i, shop in enumerate(SHOPS_CATALOG):
        scalar[38 + i] = 1.0 if shop in unlocked_shops else 0.0

    # 46..54: Shed Inventory (9 dims)
    shed = private.get("shed", {}) if isinstance(private.get("shed"), dict) else {}
    total_shed = 0.0
    for i, prod in enumerate(PRODUCTS):
        try:
            s_count = float(shed.get(prod, 0) or 0)
        except (ValueError, TypeError):
            s_count = 0.0
        total_shed += max(s_count, 0.0)
        scalar[46 + i] = min(max(s_count, 0.0) / 25.0, 4.0)

    # 55: Shed Saturation Ratio
    scalar[55] = min(total_shed / 100.0, 2.0)

    # 56..60: Seed Inventory (5 dims)
    seeds = private.get("seeds", {}) if isinstance(private.get("seeds"), dict) else {}
    for i, crop in enumerate(CROPS):
        try:
            seed_count = float(seeds.get(crop, 0) or 0)
        except (ValueError, TypeError):
            seed_count = 0.0
        scalar[56 + i] = min(max(seed_count, 0.0) / 10.0, 5.0)

    # 61..65: Active Planted Crops (5 dims)
    for i, crop in enumerate(CROPS):
        scalar[61 + i] = min(float(planted_counts.get(crop, 0)) / 25.0, 4.0)

    # 66..68: Living Animals (3 dims)
    for i, an in enumerate(ANIMALS):
        scalar[66 + i] = min(float(animal_counts.get(an, 0)) / 10.0, 2.0)

    # 69..71: Opponent Public Profile (3 dims)
    opp_hands = opp_farm.get("hands", [])
    opp_num_hands = len(opp_hands) if isinstance(opp_hands, list) else 0
    scalar[69] = min(max(opp_money, 0.0) / 10000.0, 10.0)
    scalar[70] = float(len(opp_unlocked_quads)) / 4.0
    scalar[71] = float(opp_num_hands + 1) / 13.0

    # Ensure no NaN or Inf can escape under any circumstance
    spatial = np.nan_to_num(spatial, nan=0.0, posinf=1.0, neginf=-1.0)
    scalar = np.nan_to_num(scalar, nan=0.0, posinf=1.0, neginf=-1.0)

    return spatial, scalar
