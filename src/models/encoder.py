"""
Observation State Feature Encoder and Macro-Action Vocabulary for AlphaZero / MCTS.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

# Canonical item and crop ordering
CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

# Macro-Action Vocabulary for Policy Head
MACRO_ACTIONS = [
    "FARM_CARROTS_INTENSIVE",      # 0: Maximize carrot turnover
    "FARM_WHEAT_EXPANSION",        # 1: Build wheat reserves for animals / staples
    "FARM_DIVERSIFIED",            # 2: Mixed crops (Tomatoes, Strawberries, Melons)
    "BUY_LAND_EXPANSION",          # 3: Purchase next quadrant if affordable
    "HIRE_EXTRA_LABOR",            # 4: Recruit extra farmhands today
    "BUILD_GOOSE_COOP",            # 5: Build coop and purchase goose
    "BUILD_PASTURE_LIVESTOCK",     # 6: Build pasture and purchase cow/sheep
    "HARVEST_AND_LIQUIDATE_ALL",   # 7: Full harvest and dump shed inventory
    "MARKET_ARBITRAGE_TRADE",      # 8: Market trading (buy product/fertilizer, timed shop sales)
    "LIVESTOCK_CARE_FEED",         # 9: Animal care, feeding, and fertilizer handling
]

NUM_MACRO_ACTIONS = len(MACRO_ACTIONS)
SPATIAL_CHANNELS = 11
SCALAR_DIM = 32


def encode_observation(obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes the raw observation dict into:
    1. Spatial grid tensor: Shape (11, 10, 10)
    2. Global economic scalar vector: Shape (32,)
    """
    player = obs.get("player", 0)
    farms = obs.get("farms", [{}, {}])
    my_farm = farms[player] if player < len(farms) else {}
    opp_farm = farms[1 - player] if 1 - player < len(farms) else {}

    private = obs.get("private", {}) or {}
    market = obs.get("market", {}) or {}
    town = obs.get("town", {}) or {}

    day = float(obs.get("day", 0))
    hour = float(obs.get("hour", 0))
    step = float(obs.get("step", 0))

    # --- 1. SPATIAL TENSOR (11, 10, 10) ---
    grid = np.zeros((SPATIAL_CHANNELS, 10, 10), dtype=np.float32)

    tiles = my_farm.get("tiles", [])
    for r in range(min(10, len(tiles))):
        for c in range(min(10, len(tiles[r]))):
            t = tiles[r][c]
            if t == "LOCKED":
                grid[0, r, c] = 0.0  # Locked
            else:
                grid[0, r, c] = 1.0  # Unlocked

            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    grid[1, r, c] = 1.0  # Plant present
                    crop = t.get("crop", "CARROT")
                    if crop in CROPS:
                        grid[2, r, c] = (CROPS.index(crop) + 1) / len(CROPS)
                    age = max(0, day - t.get("planted_day", day))
                    grid[3, r, c] = min(age / 12.0, 1.0)
                    grid[4, r, c] = 1.0 if t.get("watered_today", False) else 0.0
                    grid[5, r, c] = min(float(t.get("yield_units", 0)) / 6.0, 1.0)
                elif k in ["COOP", "PASTURE"]:
                    grid[6, r, c] = 0.5 if k == "COOP" else 1.0
                    animal = t.get("animal")
                    if animal in ANIMALS:
                        grid[7, r, c] = (ANIMALS.index(animal) + 1) / len(ANIMALS)
                elif k == "WEED":
                    grid[8, r, c] = 1.0

    # Farmer and Hands positions
    fx, fy = my_farm.get("farmer", [4, 4])
    if 0 <= fy < 10 and 0 <= fx < 10:
        grid[9, fy, fx] = 1.0

    for hx, hy in my_farm.get("hands", []):
        if 0 <= hy < 10 and 0 <= hx < 10:
            grid[10, hy, hx] += 1.0

    # --- 2. GLOBAL SCALARS (32,) ---
    scalars = np.zeros(SCALAR_DIM, dtype=np.float32)

    money = float(my_farm.get("money", 3000.0))
    opp_money = float(opp_farm.get("money", 3000.0))

    scalars[0] = min(money / 10000.0, 1.0)
    scalars[1] = min(opp_money / 10000.0, 1.0)
    scalars[2] = (money - opp_money) / 5000.0
    scalars[3] = step / 720.0
    scalars[4] = day / 30.0
    scalars[5] = hour / 24.0
    scalars[6] = float(my_farm.get("hires_today", 0)) / 5.0

    unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
    scalars[7] = len(unlocked_quads) / 4.0

    # Shed inventory counts
    shed = private.get("shed", {})
    for idx, prod in enumerate(PRODUCTS):
        scalars[8 + idx] = min(float(shed.get(prod, 0)) / 25.0, 1.0)

    # Seed inventory counts
    seeds = private.get("seeds", {})
    for idx, crop in enumerate(CROPS):
        scalars[17 + idx] = min(float(seeds.get(crop, 0)) / 10.0, 1.0)

    # Market prices
    prices = market.get("prices", {})
    for idx, prod in enumerate(PRODUCTS):
        scalars[22 + idx] = min(float(prices.get(prod, 25)) / 250.0, 1.0)

    scalars[31] = 1.0 if day >= 27 else 0.0  # End-game liquidation indicator

    return grid, scalars
