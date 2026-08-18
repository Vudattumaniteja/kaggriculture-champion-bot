"""
AlphaGoat Mega League: 30-Personality AlphaStar / AlphaZero Multi-Agent League for Kaggriculture.

Components:
1. 30 Distinct Strategic Personalities:
   - Crop Specialists:
     * MelonRusherAgent (Ankit0017 style 12-day Melon + Fertilizer rusher)
     * TomatoMonopolistAgent (Multi-harvest Tomato engine)
     * StrawberryAristocratAgent (Long-tail Strawberry luxury crop rusher)
     * CarrotSprinterAgent (Clockwork 3-day Carrot cycling, zero deadweight)
     * WheatIndustrialistAgent (Massive Wheat acreage, high turnover)
     * PortfolioHedgerAgent (Risk-parity diversified agricultural basket)
   - Livestock & Husbandry Masters:
     * DairyBaronAgent (Pasture + Cow rusher, daily milking @ $160)
     * GooseEggSwarmAgent (Multi-Coop Goose swarm, daily egg production @ $50)
     * WoolSpecialistAgent (Sheep pasture master, high-value wool @ $200)
     * OrganicFertilizerTycoonAgent (Manure fertilizer harvesting + targeted boosting)
   - Town Shop & Arbitrage Snipers:
     * PizzaShopSniperAgent (Targeted Milk + Tomato + Wheat supply chain)
     * BakeryMonopolistAgent (Targeted Egg + Wheat supply chain)
     * SmoothieExploiterAgent (Targeted Strawberry + Milk supply chain)
     * MarketPriceCrasherAgent (High-volume commodity dumping to collapse market prices)
     * CommoditySpeculatorAgent (Price-sensitive hoarder and swing trader)
     * BrunchSpotCornerAgent (Targeted Egg + Wheat + Strawberry supply chain)
     * IceCreamTycoonAgent (Targeted Strawberry + Milk + Wheat supply chain)
     * PetCafeSupplierAgent (Targeted high-density Carrot supply chain)
     * FarmersMarketDominatorAgent (Full 4-crop basket supply chain)
   - Expansion & Labor Archetypes:
     * FourQuadrantOverlordAgent (Unlocks all 4 quadrants / 100 tiles, massive labor)
     * NWMinimalistAgent (Strictly NW 25-tile domain, $0 land spend, dense precision)
     * FiveWorkerSwarmAgent (Hires maximum 5 farmhands daily for massive chore throughput)
     * LeanSoloOperatorAgent (Hires 0 farmhands, 0 wage overhead, cash preservation)
     * SerpentineChorerAgent (Spatial boustrophedon pathfinder minimizing travel latency)
   - Game-Theoretic Adversaries & Baselines:
     * AntiCompetitorShadowAgent (Mirrors opponent crops/livestock to crash market prices)
     * GreedySnowballerAgent (100% reinvestment geometric compounder)
     * SafePreserverAgent (Risk-averse value preserver with $1,000 cash reserve)
     * StochasticPerturbationAdversary (Invariance stress-tester with Dirichlet/random noise)
     * KaggleStarterAgent (Deterministic baseline wrapper)
     * HistoricalCheckpointAgent (Past neural network checkpoint wrapper)
2. Prioritized Fictitious Self-Play (PFSP) Matchmaker:
   - Dynamic Elo rating tracker (K=32)
   - Head-to-head payoff matrix & margin logging
   - Hard-opponent oversampling: P(i) ~ max(0.05, (1 - win_rate_i)^gamma)
"""

import copy
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

# Ensure repository root is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.agents.utils import DIRS, SHED_TILES, get_manhattan_dist, get_step_towards, parse_observation

# ==============================================================================
# CONSTANTS & SPECIFICATIONS
# ==============================================================================

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS = ["GOOSE", "COW", "SHEEP"]

CROP_SPECS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
}

BASE_PRICES = {
    "WHEAT": 25,
    "CARROT": 35,
    "TOMATO": 60,
    "STRAWBERRY": 120,
    "MELON": 250,
    "EGG": 50,
    "MILK": 160,
    "WOOL": 200,
    "FERTILIZER": 100,
}

SHOPS_CATALOG = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}


# ==============================================================================
# HELPER BASE CLASS FOR MEGA LEAGUE HEURISTICS
# ==============================================================================

class BaseHeuristicAgent:
    """Base class providing standard chore scheduling and movement primitives."""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    def _execute_chores(
        self,
        state: Dict[str, Any],
        priority_tasks: List[Tuple[int, int]],
        plant_tasks: List[Tuple[int, int]],
        preferred_seed_fn: Optional[Callable[[Dict[str, int], int], Optional[str]]] = None,
        pickup_item: Optional[str] = None,
        pickup_qty: int = 1,
    ) -> List[List[Any]]:
        day = state["day"]
        tiles = state["tiles"]
        farmer_pos = state["farmer_pos"]
        hands_pos = state["hands_pos"]
        shed = state["shed"]
        seeds = state["seeds"]
        inventories = state["inventories"]

        claimed_tasks: Set[Tuple[int, int]] = set()
        units = [farmer_pos] + hands_pos
        unit_actions: List[List[Any]] = []

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in SHED_TILES

            # 1. Shed item pickup
            if (
                is_at_shed
                and pickup_item
                and u_inv.get(pickup_item, 0) == 0
                and shed.get(pickup_item, 0) > 0
            ):
                qty = min(pickup_qty, shed.get(pickup_item, 0))
                unit_actions.append(["PICKUP", pickup_item, qty])
                shed[pickup_item] -= qty
                u_inv[pickup_item] = qty
                continue

            # 2. Standing Tile Actions
            if isinstance(u_tile, dict):
                k = u_tile.get("kind")
                if k == "PLANT":
                    crop = u_tile.get("crop", "CARROT")
                    cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                    age = day - u_tile.get("planted_day", day)
                    is_ongoing = cspec["ongoing"]
                    yield_u = u_tile.get("yield_units", 0)

                    # Harvest
                    if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                        is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        unit_actions.append(["HARVEST"])
                        continue
                    # Water
                    if not u_tile.get("watered_today", False):
                        unit_actions.append(["WATER"])
                        continue
                    # Fertilize
                    if (
                        crop in ["MELON", "STRAWBERRY", "TOMATO"]
                        and u_tile.get("fertilized_until_day", -1) < day
                        and u_inv.get("FERTILIZER", 0) > 0
                    ):
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue

                elif k == "COOP":
                    animal = u_tile.get("animal")
                    if animal == "GOOSE":
                        if not u_tile.get("petted_today", False):
                            unit_actions.append(["PET"])
                            continue
                        if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                            unit_actions.append(["FEED"])
                            u_inv["WHEAT"] -= 1
                            continue
                        if u_tile.get("has_egg", False):
                            unit_actions.append(["COLLECT_EGG"])
                            continue

                elif k == "PASTURE":
                    animal = u_tile.get("animal")
                    if animal in ["COW", "SHEEP"]:
                        if not u_tile.get("petted_today", False):
                            unit_actions.append(["PET"])
                            continue
                        if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                            unit_actions.append(["FEED"])
                            u_inv["WHEAT"] -= 1
                            continue
                        if animal == "COW" and u_tile.get("has_milk", False):
                            unit_actions.append(["MILK"])
                            continue
                        if animal == "SHEEP" and u_tile.get("has_wool", False):
                            unit_actions.append(["SHEAR"])
                            continue

                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue

            elif u_tile is None and day < 27 and u_pos in plant_tasks:
                chosen_crop = None
                if preferred_seed_fn:
                    chosen_crop = preferred_seed_fn(seeds, day)
                if not chosen_crop:
                    for cand in ["MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"]:
                        if seeds.get(cand, 0) > 0:
                            chosen_crop = cand
                            break

                if chosen_crop and seeds.get(chosen_crop, 0) > 0:
                    unit_actions.append(["PLANT", chosen_crop])
                    seeds[chosen_crop] -= 1
                    continue

            # 3. Pathfind to nearest task
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
                step_dir = get_step_towards(u_pos, best_target)
                unit_actions.append([step_dir] if step_dir != "PASS" else ["PASS"])
            else:
                if not is_at_shed:
                    nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                    step_dir = get_step_towards(u_pos, nearest_shed)
                    unit_actions.append([step_dir] if step_dir != "PASS" else ["PASS"])
                else:
                    unit_actions.append(["PASS"])

        return unit_actions


# ==============================================================================
# 1. CROP SPECIALISTS (6 PERSONALITIES)
# ==============================================================================

class MelonRusherAgent(BaseHeuristicAgent):
    """Aggressive 12-day Melon + Fertilizer Rusher (Ankit0017 style)."""

    def __init__(self):
        super().__init__("MelonRusher")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []

        # Sell goods
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                if item == "FERTILIZER" and day <= 16 and count > 4:
                    market_orders.append(["SELL", item, count - 4])
                elif item != "FERTILIZER" or day > 16:
                    market_orders.append(["SELL", item, count])

        # Land expansions
        if "NE" not in unlocked and money >= 1100 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and "NE" in unlocked and money >= 2200 and day <= 15:
            market_orders.append(["BUY_LAND"])
            money -= 2000

        # Labor hiring
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])
            money -= 2

        # Tile analysis
        melon_planted = 0
        carrot_planted = 0
        empty_tiles = []
        water_tasks = []
        fertilize_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        crop = t.get("crop", "CARROT")
                        age = day - t.get("planted_day", day)
                        cspec = CROP_SPECS.get(crop, CROP_SPECS["CARROT"])
                        if crop == "MELON":
                            melon_planted += 1
                        elif crop == "CARROT":
                            carrot_planted += 1

                        if age >= cspec["max_yield_day"] or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                        elif crop == "MELON" and t.get("fertilized_until_day", -1) < day and day <= 16:
                            fertilize_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        # Seed & Fertilizer purchasing
        curr_melon_seeds = seeds.get("MELON", 0)
        curr_carrot_seeds = seeds.get("CARROT", 0)

        if day <= 4:
            needed = max(0, 18 - melon_planted - curr_melon_seeds)
            if needed > 0 and money >= 80:
                qty = min(needed, 6, int(money // 80))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "MELON", qty])
                    money -= 80 * qty
            if shed.get("FERTILIZER", 0) < 3 and money >= 250 and melon_planted >= 6:
                qty = min(2, int(money // 100))
                if qty > 0:
                    market_orders.append(["BUY", "FERTILIZER", qty])
        elif 12 <= day <= 14:
            needed = max(0, 20 - melon_planted - curr_melon_seeds)
            if needed > 0 and money >= 80:
                qty = min(needed, 6, int(money // 80))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "MELON", qty])
        elif 24 <= day <= 26:
            needed = max(0, 24 - carrot_planted - curr_carrot_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 8, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:24] if day < 27 else []
        priority_tasks = water_tasks + harvest_tasks + fertilize_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            if (d <= 4 or 12 <= d <= 14) and s.get("MELON", 0) > 0:
                return "MELON"
            if s.get("CARROT", 0) > 0:
                return "CARROT"
            return None

        unit_actions = self._execute_chores(
            state, priority_tasks, plant_tasks, seed_fn, pickup_item="FERTILIZER", pickup_qty=2
        )
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class TomatoMonopolistAgent(BaseHeuristicAgent):
    """Multi-harvest Tomato continuous yield engine."""

    def __init__(self):
        super().__init__("TomatoMonopolist")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        if "NE" not in unlocked and money >= 1200 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])

        tomato_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        crop = t.get("crop", "CARROT")
                        age = day - t.get("planted_day", day)
                        if crop == "TOMATO":
                            tomato_planted += 1
                        if crop == "TOMATO" and age >= 8 and t.get("yield_units", 0) > 0:
                            harvest_tasks.append(pos)
                        elif crop != "TOMATO" and (age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0)):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        curr_tom_seeds = seeds.get("TOMATO", 0)
        curr_car_seeds = seeds.get("CARROT", 0)

        if day <= 6:
            needed = max(0, 16 - tomato_planted - curr_tom_seeds)
            if needed > 0 and money >= 50:
                qty = min(needed, 4, int(money // 50))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "TOMATO", qty])
        elif 24 <= day <= 26:
            needed = max(0, 16 - curr_car_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 6, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:20] if day < 27 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            if d <= 8 and s.get("TOMATO", 0) > 0:
                return "TOMATO"
            if s.get("CARROT", 0) > 0:
                return "CARROT"
            return None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class StrawberryAristocratAgent(BaseHeuristicAgent):
    """Long-tail Strawberry luxury crop rusher ($120 base price)."""

    def __init__(self):
        super().__init__("StrawberryAristocrat")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        if "NE" not in unlocked and money >= 1200 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])

        straw_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        crop = t.get("crop", "CARROT")
                        age = day - t.get("planted_day", day)
                        if crop == "STRAWBERRY":
                            straw_planted += 1
                        if crop == "STRAWBERRY" and age >= 10 and t.get("yield_units", 0) > 0:
                            harvest_tasks.append(pos)
                        elif crop != "STRAWBERRY" and (age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0)):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        curr_straw_seeds = seeds.get("STRAWBERRY", 0)
        curr_car_seeds = seeds.get("CARROT", 0)

        if day <= 5:
            needed = max(0, 14 - straw_planted - curr_straw_seeds)
            if needed > 0 and money >= 100:
                qty = min(needed, 4, int(money // 100))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "STRAWBERRY", qty])
        elif 24 <= day <= 26:
            needed = max(0, 16 - curr_car_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 6, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:20] if day < 27 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            if d <= 6 and s.get("STRAWBERRY", 0) > 0:
                return "STRAWBERRY"
            if s.get("CARROT", 0) > 0:
                return "CARROT"
            return None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class CarrotSprinterAgent(BaseHeuristicAgent):
    """Clockwork 3-day turnaround Carrot cycling, zero deadweight."""

    def __init__(self):
        super().__init__("CarrotSprinter")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        if "NE" not in unlocked and money >= 1200 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])

        carrots_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        carrots_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = min(len(unlocked) * 16, 26)
        curr_seeds = seeds.get("CARROT", 0)
        if day <= 26:
            needed = max(0, target_cap - carrots_planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 8, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class WheatIndustrialistAgent(BaseHeuristicAgent):
    """Massive Wheat acreage, fast 4-day turnover, bulk scale."""

    def __init__(self):
        super().__init__("WheatIndustrialist")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        if "NE" not in unlocked and money >= 1200 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        if "SW" not in unlocked and "NE" in unlocked and money >= 2200 and day <= 14:
            market_orders.append(["BUY_LAND"])
            money -= 2000

        if day < 28 and hour <= 1 and hires_today < 3 and money >= 50:
            market_orders.append(["HIRE"])

        wheat_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        wheat_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 4 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = min(len(unlocked) * 16, 36)
        curr_seeds = seeds.get("WHEAT", 0)
        if day <= 25:
            needed = max(0, target_cap - wheat_planted - curr_seeds)
            if needed > 0 and money >= 10:
                qty = min(needed, 10, int(money // 10))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "WHEAT", qty])
        elif 26 <= day <= 26:
            needed = max(0, 20 - seeds.get("CARROT", 0))
            if needed > 0 and money >= 20:
                qty = min(needed, 8, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            if d <= 25 and s.get("WHEAT", 0) > 0:
                return "WHEAT"
            if s.get("CARROT", 0) > 0:
                return "CARROT"
            return None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class PortfolioHedgerAgent(BaseHeuristicAgent):
    """Risk-parity diversified agricultural basket balancing cash flow and margin."""

    def __init__(self):
        super().__init__("PortfolioHedger")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.balanced_farmer import crop_farmer_portfolio
        return crop_farmer_portfolio(obs, config)


# ==============================================================================
# 2. LIVESTOCK & HUSBANDRY MASTERS (4 PERSONALITIES)
# ==============================================================================

class DairyBaronAgent(BaseHeuristicAgent):
    """Pasture + Cow rusher, daily milking @ $160, daily feeding & petting."""

    def __init__(self):
        super().__init__("DairyBaron")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.livestock_bot import LivestockHusbandryAgent
        bot = LivestockHusbandryAgent(target_geese=0, target_cows=3, target_sheep=0, wheat_tiles=8)
        return bot(obs, config)


class GooseEggSwarmAgent(BaseHeuristicAgent):
    """Multi-Coop Goose swarm, daily egg production @ $50, fast ROI."""

    def __init__(self):
        super().__init__("GooseEggSwarm")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.livestock_bot import LivestockHusbandryAgent
        bot = LivestockHusbandryAgent(target_geese=4, target_cows=0, target_sheep=0, wheat_tiles=4)
        return bot(obs, config)


class WoolSpecialistAgent(BaseHeuristicAgent):
    """Sheep pasture master, high-value wool @ $200."""

    def __init__(self):
        super().__init__("WoolSpecialist")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.livestock_bot import LivestockHusbandryAgent
        bot = LivestockHusbandryAgent(target_geese=0, target_cows=0, target_sheep=4, wheat_tiles=6)
        return bot(obs, config)


class OrganicFertilizerTycoonAgent(BaseHeuristicAgent):
    """Combines livestock manure with fertilizer sales and targeted high-yield crop boosting."""

    def __init__(self):
        super().__init__("OrganicFertilizerTycoon")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.livestock_bot import livestock_agent
        return livestock_agent(obs, config)


# ==============================================================================
# 3. TOWN SHOP & ARBITRAGE SNIPERS (9 PERSONALITIES)
# ==============================================================================

class PizzaShopSniperAgent(BaseHeuristicAgent):
    """Targeted Milk + Tomato + Wheat supply chain to capture Pizza Shop 3x multiplier."""

    def __init__(self):
        super().__init__("PizzaShopSniper")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.arbitrage_bot import arbitrage_agent
        return arbitrage_agent(obs, config)


class BakeryMonopolistAgent(BaseHeuristicAgent):
    """Targeted Egg + Wheat supply chain to monopolize Bakery demand."""

    def __init__(self):
        super().__init__("BakeryMonopolist")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.arbitrage_bot import arbitrage_agent
        return arbitrage_agent(obs, config)


class SmoothieExploiterAgent(BaseHeuristicAgent):
    """Targeted Strawberry + Milk supply chain for Smoothie Shop."""

    def __init__(self):
        super().__init__("SmoothieExploiter")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.arbitrage_bot import arbitrage_agent
        return arbitrage_agent(obs, config)


class MarketPriceCrasherAgent(BaseHeuristicAgent):
    """Aggressive front-runner that dumps high volumes of commodities early to collapse opponent profit margins."""

    def __init__(self):
        super().__init__("MarketPriceCrasher")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.training.league import town_shop_monopolizer_agent
        return town_shop_monopolizer_agent(obs, config)


class CommoditySpeculatorAgent(BaseHeuristicAgent):
    """Dynamically tracks market prices, hoarding when prices drop and dumping on surges."""

    def __init__(self):
        super().__init__("CommoditySpeculator")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        prices = state["market_prices"]
        shed = state["shed"]
        money = state["money"]
        market_orders = []

        # Swing trade: sell when price >= base_price * 0.95 or day >= 27
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                base = BASE_PRICES.get(item, 35)
                cur_p = prices.get(item, base)
                if cur_p >= base * 0.95 or day >= 27:
                    market_orders.append(["SELL", item, count])

        from src.agents.balanced_farmer import crop_farmer_portfolio
        res = crop_farmer_portfolio(obs, config)
        if market_orders:
            res["market"] = market_orders[:10]
        return res


class BrunchSpotCornerAgent(BaseHeuristicAgent):
    """Targeted Egg + Wheat + Strawberry supply chain."""

    def __init__(self):
        super().__init__("BrunchSpotCorner")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.arbitrage_bot import arbitrage_agent
        return arbitrage_agent(obs, config)


class IceCreamTycoonAgent(BaseHeuristicAgent):
    """Targeted Strawberry + Milk + Wheat supply chain."""

    def __init__(self):
        super().__init__("IceCreamTycoon")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.arbitrage_bot import arbitrage_agent
        return arbitrage_agent(obs, config)


class PetCafeSupplierAgent(BaseHeuristicAgent):
    """Targeted high-density Carrot supply chain for Pet Cafe."""

    def __init__(self):
        super().__init__("PetCafeSupplier")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.balanced_farmer import crop_farmer_carrot
        return crop_farmer_carrot(obs, config)


class FarmersMarketDominatorAgent(BaseHeuristicAgent):
    """Full 4-crop basket supply chain (Wheat, Carrot, Tomato, Strawberry)."""

    def __init__(self):
        super().__init__("FarmersMarketDominator")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.balanced_farmer import crop_farmer_portfolio
        return crop_farmer_portfolio(obs, config)


# ==============================================================================
# 4. EXPANSION & LABOR ARCHETYPES (5 PERSONALITIES)
# ==============================================================================

class FourQuadrantOverlordAgent(BaseHeuristicAgent):
    """Unlocks all 4 quadrants (100 tiles) and operates massive industrial workforce."""

    def __init__(self):
        super().__init__("FourQuadrantOverlord")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        # Aggressive 4-quadrant expansion
        if "NE" not in unlocked and money >= 1100 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and "NE" in unlocked and money >= 2100 and day <= 14:
            market_orders.append(["BUY_LAND"])
            money -= 2000
        elif "SE" not in unlocked and "SW" in unlocked and money >= 4200 and day <= 20:
            market_orders.append(["BUY_LAND"])
            money -= 4000

        # Hire 4 farmhands daily
        if day < 28 and hour <= 1 and hires_today < 4 and money >= 50:
            market_orders.append(["HIRE"])

        planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = min(len(unlocked) * 16, 50)
        curr_seeds = sum(seeds.values())
        if day <= 26:
            needed = max(0, target_cap - planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 10, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class NWMinimalistAgent(BaseHeuristicAgent):
    """Strictly NW 25-tile domain, $0 spent on land expansions, dense precision."""

    def __init__(self):
        super().__init__("NWMinimalist")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        # NEVER BUY LAND ($0 spent on land)
        # Hires 1 farmhand daily
        if day < 28 and hour <= 1 and hires_today < 1 and money >= 20:
            market_orders.append(["HIRE"])

        carrots_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(5):
            for c in range(5):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        carrots_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = 20
        curr_seeds = seeds.get("CARROT", 0)
        if day <= 26:
            needed = max(0, target_cap - carrots_planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 6, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class FiveWorkerSwarmAgent(BaseHeuristicAgent):
    """Hires maximum 5 farmhands daily for massive parallel chore throughput."""

    def __init__(self):
        super().__init__("FiveWorkerSwarm")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        if "NE" not in unlocked and money >= 1200 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000

        # Maximum 5 hires
        if day < 28 and hour <= 1 and hires_today < 5 and money >= 30:
            market_orders.append(["HIRE"])

        carrots_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        carrots_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = min(len(unlocked) * 16, 32)
        curr_seeds = seeds.get("CARROT", 0)
        if day <= 26:
            needed = max(0, target_cap - carrots_planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 10, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class LeanSoloOperatorAgent(BaseHeuristicAgent):
    """Hires 0 farmhands, 0 wage overhead, cash preservation."""

    def __init__(self):
        super().__init__("LeanSoloOperator")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        # Hires exactly 0 farmhands
        carrots_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(5):
            for c in range(5):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        carrots_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = 12
        curr_seeds = seeds.get("CARROT", 0)
        if day <= 26:
            needed = max(0, target_cap - carrots_planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 4, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": [],
            "market": market_orders[:10],
        }


class SerpentineChorerAgent(BaseHeuristicAgent):
    """Spatial boustrophedon pathfinder sorting chore tiles in serpentine order."""

    def __init__(self):
        super().__init__("SerpentineChorer")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        if "NE" not in unlocked and money >= 1200 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])

        carrots_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            cols = range(len(tiles[r])) if r % 2 == 0 else reversed(range(len(tiles[r])))
            for c in cols:
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        carrots_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = min(len(unlocked) * 16, 28)
        curr_seeds = seeds.get("CARROT", 0)
        if day <= 26:
            needed = max(0, target_cap - carrots_planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 8, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = harvest_tasks + water_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


# ==============================================================================
# 5. GAME-THEORETIC ADVERSARIES & BASELINES (6 PERSONALITIES)
# ==============================================================================

class AntiCompetitorShadowAgent(BaseHeuristicAgent):
    """Observes opponent tiles/inventories and front-runs/dumps their identical commodity."""

    def __init__(self):
        super().__init__("AntiCompetitorShadow")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        # Opponent audit
        farms = obs.get("farms", [{}, {}])
        opp_farm = farms[1 - obs.get("player", 0)] if len(farms) > 1 else {}
        opp_tiles = opp_farm.get("tiles", [])

        opp_crop_counts: Dict[str, int] = {c: 0 for c in CROPS}
        for r in range(len(opp_tiles)):
            for c in range(len(opp_tiles[r])):
                t = opp_tiles[r][c]
                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    cr = t.get("crop", "CARROT")
                    opp_crop_counts[cr] = opp_crop_counts.get(cr, 0) + 1

        # Determine opponent primary crop focus
        target_crop = max(opp_crop_counts, key=opp_crop_counts.get)
        if opp_crop_counts[target_crop] == 0:
            target_crop = "MELON" if day <= 12 else "CARROT"

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        if "NE" not in unlocked and money >= 1200 and day <= 8:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        if day < 28 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])

        planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        planted += 1
                        age = day - t.get("planted_day", day)
                        cspec = CROP_SPECS.get(t.get("crop", "CARROT"), CROP_SPECS["CARROT"])
                        if age >= cspec["max_yield_day"] or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = min(len(unlocked) * 16, 26)
        curr_seeds = seeds.get(target_crop, 0)
        seed_cost = CROP_SPECS[target_crop]["seed"]

        if day <= 24:
            needed = max(0, target_cap - planted - curr_seeds)
            if needed > 0 and money >= seed_cost:
                qty = min(needed, 6, int(money // seed_cost))
                if qty > 0:
                    market_orders.append(["BUY_SEED", target_crop, qty])
        elif 25 <= day <= 26:
            needed = max(0, 16 - seeds.get("CARROT", 0))
            if needed > 0 and money >= 20:
                qty = min(needed, 6, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            if s.get(target_crop, 0) > 0 and d <= 24:
                return target_crop
            if s.get("CARROT", 0) > 0:
                return "CARROT"
            return None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class GreedySnowballerAgent(BaseHeuristicAgent):
    """Reinvests 100% of cash into instant compounding assets until Day 24, then halts all capex."""

    def __init__(self):
        super().__init__("GreedySnowballer")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        # 100% reinvestment until day 24
        if day <= 24:
            if "NE" not in unlocked and money >= 1050:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and money >= 2050 and day <= 15:
                market_orders.append(["BUY_LAND"])
                money -= 2000
            if day < 24 and hour <= 1 and hires_today < 3 and money >= 20:
                market_orders.append(["HIRE"])

        carrots_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        carrots_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = min(len(unlocked) * 16, 36)
        curr_seeds = seeds.get("CARROT", 0)
        if day <= 24:
            needed = max(0, target_cap - carrots_planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 12, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])
        elif 25 <= day <= 26:
            needed = max(0, 16 - carrots_planted - curr_seeds)
            if needed > 0 and money >= 20:
                qty = min(needed, 6, int(money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class SafePreserverAgent(BaseHeuristicAgent):
    """Maintains a $1,000 cash reserve, zero risk, steady carrot farming."""

    def __init__(self):
        super().__init__("SafePreserver")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        shed = state["shed"]
        seeds = state["seeds"]
        hires_today = state["hires_today"]
        tiles = state["tiles"]

        market_orders = []
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                market_orders.append(["SELL", item, count])

        # Hires 1 farmhand only if cash > $1050
        if day < 28 and hour <= 1 and hires_today < 1 and money >= 1050:
            market_orders.append(["HIRE"])

        carrots_planted = 0
        empty_tiles = []
        water_tasks = []
        harvest_tasks = []
        weed_tasks = []

        for r in range(5):
            for c in range(5):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        carrots_planted += 1
                        age = day - t.get("planted_day", day)
                        if age >= 3 or (day >= 28 and t.get("yield_units", 0) > 0):
                            harvest_tasks.append(pos)
                        elif not t.get("watered_today", False):
                            water_tasks.append(pos)
                    elif k == "WEED":
                        weed_tasks.append(pos)

        target_cap = 16
        curr_seeds = seeds.get("CARROT", 0)
        # Safe purchasing above $1,000 cash buffer
        if day <= 26 and money > 1000:
            free_money = money - 1000
            needed = max(0, target_cap - carrots_planted - curr_seeds)
            if needed > 0 and free_money >= 20:
                qty = min(needed, 4, int(free_money // 20))
                if qty > 0:
                    market_orders.append(["BUY_SEED", "CARROT", qty])

        empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        plant_tasks = empty_tiles[:target_cap] if day <= 26 else []
        priority_tasks = water_tasks + harvest_tasks + plant_tasks + weed_tasks

        def seed_fn(s: Dict[str, int], d: int) -> Optional[str]:
            return "CARROT" if s.get("CARROT", 0) > 0 else None

        unit_actions = self._execute_chores(state, priority_tasks, plant_tasks, seed_fn)
        return {
            "farmer": unit_actions[0] if unit_actions else ["PASS"],
            "hands": unit_actions[1:] if len(unit_actions) > 1 else [],
            "market": market_orders[:10],
        }


class StochasticPerturbationAdversary(BaseHeuristicAgent):
    """Injects controlled Dirichlet noise / stochastic variations into a strong hybrid policy."""

    def __init__(self, noise_prob: float = 0.10):
        super().__init__("StochasticPerturbation")
        self.noise_prob = noise_prob

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.hybrid_expert import hybrid_expert_agent
        res = hybrid_expert_agent(obs, config)

        if random.random() < self.noise_prob:
            # Inject slight stochastic perturbation
            valid_dirs = ["NORTH", "SOUTH", "EAST", "WEST", "PASS"]
            if res.get("farmer") and random.random() < 0.3:
                res["farmer"] = [random.choice(valid_dirs)]
            if res.get("hands") and random.random() < 0.3:
                res["hands"] = [[random.choice(valid_dirs)] for _ in res["hands"]]
        return res


class KaggleStarterAgent(BaseHeuristicAgent):
    """Deterministic baseline wrapper matching competition starter bot."""

    def __init__(self):
        super().__init__("KaggleStarter")

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        from src.agents.balanced_farmer import crop_farmer_carrot
        return crop_farmer_carrot(obs, config)


class HistoricalCheckpointAgent(BaseHeuristicAgent):
    """Dynamic neural agent wrapper executing past checkpoint neural networks."""

    def __init__(self, state_dict: Optional[Dict[str, torch.Tensor]] = None, name: str = "Checkpoint"):
        super().__init__(name)
        self.state_dict = state_dict
        self._model = None

    def _get_model(self):
        if self._model is None and self.state_dict is not None:
            from src.models.ssl_network import SSLPolicyValueNet
            self._model = SSLPolicyValueNet()
            self._model.load_state_dict(self.state_dict)
            self._model.eval()
        return self._model

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        model = self._get_model()
        if model is None:
            from src.agents.hybrid_expert import hybrid_expert_agent
            return hybrid_expert_agent(obs, config)

        from src.models.encoder import encode_observation
        from src.models.muzero_mcts import compute_action_mask
        from src.agents.hybrid_expert import hybrid_expert_agent
        from src.agents.balanced_farmer import crop_farmer_carrot, crop_farmer_wheat, crop_farmer_portfolio
        from src.agents.livestock_bot import livestock_agent
        from src.agents.arbitrage_bot import arbitrage_agent

        grid, scalars = encode_observation(obs)
        grid_t = torch.from_numpy(grid).unsqueeze(0)
        scalars_t = torch.from_numpy(scalars).unsqueeze(0)

        with torch.no_grad():
            probs, val = model.predict(grid_t, scalars_t)
            probs_np = probs.cpu().numpy()[0]

        mask = compute_action_mask(obs)
        masked_probs = probs_np * mask
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
            chosen_action = int(np.argmax(masked_probs))
        else:
            chosen_action = 7

        executors = {
            0: crop_farmer_carrot,
            1: crop_farmer_wheat,
            2: crop_farmer_portfolio,
            3: hybrid_expert_agent,
            4: hybrid_expert_agent,
            5: hybrid_expert_agent,
            6: livestock_agent,
            7: None,
            8: arbitrage_agent,
            9: livestock_agent,
        }

        if chosen_action == 7:
            res = hybrid_expert_agent(obs)
            shed = obs.get("private", {}).get("shed", {})
            orders = []
            for item, count in shed.items():
                if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                    orders.append(["SELL", item, count])
            res["market"] = orders[:10]
            return res

        executor = executors.get(chosen_action, hybrid_expert_agent)
        return executor(obs)


# ==============================================================================
# 30 PERSONALITIES REGISTRY
# ==============================================================================

MEGA_LEAGUE_PERSONALITIES: Dict[str, Callable] = {
    # 1. Crop Specialists (6)
    "MelonRusher": MelonRusherAgent(),
    "TomatoMonopolist": TomatoMonopolistAgent(),
    "StrawberryAristocrat": StrawberryAristocratAgent(),
    "CarrotSprinter": CarrotSprinterAgent(),
    "WheatIndustrialist": WheatIndustrialistAgent(),
    "PortfolioHedger": PortfolioHedgerAgent(),

    # 2. Livestock & Husbandry Masters (4)
    "DairyBaron": DairyBaronAgent(),
    "GooseEggSwarm": GooseEggSwarmAgent(),
    "WoolSpecialist": WoolSpecialistAgent(),
    "OrganicFertilizerTycoon": OrganicFertilizerTycoonAgent(),

    # 3. Town Shop & Arbitrage Snipers (9)
    "PizzaShopSniper": PizzaShopSniperAgent(),
    "BakeryMonopolist": BakeryMonopolistAgent(),
    "SmoothieExploiter": SmoothieExploiterAgent(),
    "MarketPriceCrasher": MarketPriceCrasherAgent(),
    "CommoditySpeculator": CommoditySpeculatorAgent(),
    "BrunchSpotCorner": BrunchSpotCornerAgent(),
    "IceCreamTycoon": IceCreamTycoonAgent(),
    "PetCafeSupplier": PetCafeSupplierAgent(),
    "FarmersMarketDominator": FarmersMarketDominatorAgent(),

    # 4. Expansion & Labor Archetypes (5)
    "FourQuadrantOverlord": FourQuadrantOverlordAgent(),
    "NWMinimalist": NWMinimalistAgent(),
    "FiveWorkerSwarm": FiveWorkerSwarmAgent(),
    "LeanSoloOperator": LeanSoloOperatorAgent(),
    "SerpentineChorer": SerpentineChorerAgent(),

    # 5. Game-Theoretic Adversaries & Baselines (6)
    "AntiCompetitorShadow": AntiCompetitorShadowAgent(),
    "GreedySnowballer": GreedySnowballerAgent(),
    "SafePreserver": SafePreserverAgent(),
    "StochasticPerturbation": StochasticPerturbationAdversary(),
    "KaggleStarter": KaggleStarterAgent(),
    "PastMuZeroChampion": HistoricalCheckpointAgent(),
}


# ==============================================================================
# 6. ALPHAGOAT PRIORITIZED FICTITIOUS SELF-PLAY (PFSP) LEAGUE MATCHMAKER
# ==============================================================================

class AlphaGoatLeagueParticipant:
    """Represents a member in the AlphaGoat Mega League."""

    def __init__(
        self,
        name: str,
        role: str,  # 'main', 'sparring', 'checkpoint'
        agent_callable: Optional[Callable] = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        iteration: int = 0,
    ):
        self.name = name
        self.role = role
        self.agent_callable = agent_callable
        self.state_dict = state_dict
        self.iteration = iteration
        self.elo = 1500.0
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.total_matches = 0
        self.total_cash = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins / max(1, self.total_matches)) * 100.0

    @property
    def mean_cash(self) -> float:
        return self.total_cash / max(1, self.total_matches)

    def record_match(self, my_cash: float, opp_cash: float, won: Optional[bool]):
        self.total_matches += 1
        self.total_cash += my_cash
        if won is True:
            self.wins += 1
        elif won is False:
            self.losses += 1
        else:
            self.ties += 1


class AlphaGoatMegaLeague:
    """
    AlphaGoat Prioritized Fictitious Self-Play (PFSP) League Engine.
    Dynamically tracks Elo ratings and win rates for all 30 personalities.
    Over-samples opponents that give the champion the hardest challenge
    (e.g., Melon Rusher, Town Shop Monopolizer, Market Price Crasher).
    """

    def __init__(
        self,
        main_name: str = "AlphaGoatChampion",
        max_checkpoints: int = 30,
    ):
        self.main_name = main_name
        self.max_checkpoints = max_checkpoints

        self.participants: Dict[str, AlphaGoatLeagueParticipant] = {}
        self.sparring_names: List[str] = []
        self.checkpoint_names: List[str] = []
        self.payoff_matrix: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}

        # Register Main Agent
        self.main_agent = AlphaGoatLeagueParticipant(name=main_name, role="main")
        self.participants[main_name] = self.main_agent

        # Register all 30 Mega League personalities
        for name, agent_fn in MEGA_LEAGUE_PERSONALITIES.items():
            self._register_sparring_bot(name, agent_fn)

    def _register_sparring_bot(self, name: str, agent_fn: Callable):
        bot = AlphaGoatLeagueParticipant(name=name, role="sparring", agent_callable=agent_fn)
        self.participants[name] = bot
        if name not in self.sparring_names:
            self.sparring_names.append(name)

    def add_checkpoint(self, iteration: int, state_dict: Dict[str, torch.Tensor], name: Optional[str] = None):
        """Adds a past champion snapshot to the checkpoint pool."""
        ckpt_name = name or f"Checkpoint_Iter{iteration:03d}"
        ckpt = AlphaGoatLeagueParticipant(
            name=ckpt_name,
            role="checkpoint",
            state_dict={k: v.cpu().clone() for k, v in state_dict.items()},
            iteration=iteration,
            agent_callable=HistoricalCheckpointAgent(state_dict=state_dict, name=ckpt_name),
        )
        self.participants[ckpt_name] = ckpt
        self.checkpoint_names.append(ckpt_name)

        if len(self.checkpoint_names) > self.max_checkpoints:
            oldest = self.checkpoint_names.pop(0)
            if oldest in self.participants:
                del self.participants[oldest]

    def sample_opponent(
        self,
        sparring_prob: float = 0.55,
        checkpoint_prob: float = 0.30,
        self_play_prob: float = 0.15,
        gamma: float = 1.5,
    ) -> Tuple[str, AlphaGoatLeagueParticipant]:
        """
        Prioritized Fictitious Self-Play (PFSP) Matchmaking:
        - Sparring Bots: Computes champion loss rate against each personality
          w_i = max(0.05, (1 - win_rate_i)^gamma), heavily oversampling hard opponents.
        - Past Checkpoints: Recency/difficulty-weighted sampling to prevent forgetting.
        - Self-Play: Matches against the active champion.
        """
        roll = random.random()

        if (roll < sparring_prob or not self.checkpoint_names) and self.sparring_names:
            weights = []
            for name in self.sparring_names:
                matches = self.payoff_matrix.get((self.main_name, name), [])
                if not matches:
                    # Unexplored opponent: high exploration priority
                    loss_rate = 0.6
                else:
                    champ_wins = sum(1 for m in matches if m[0] > m[1])
                    champ_win_rate = champ_wins / len(matches)
                    loss_rate = 1.0 - champ_win_rate

                weight = max(0.05, loss_rate ** gamma)
                weights.append(weight)

            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            chosen_name = np.random.choice(self.sparring_names, p=probs)
            return chosen_name, self.participants[chosen_name]

        elif (roll < sparring_prob + checkpoint_prob) and self.checkpoint_names:
            n = len(self.checkpoint_names)
            weights = [np.exp(0.15 * i) for i in range(n)]
            probs = [w / sum(weights) for w in weights]
            chosen_name = np.random.choice(self.checkpoint_names, p=probs)
            return chosen_name, self.participants[chosen_name]

        else:
            return self.main_name, self.main_agent

    def update_match_result(
        self,
        p0_name: str,
        p1_name: str,
        p0_cash: float,
        p1_cash: float,
    ):
        """Updates head-to-head records and Elo ratings between two league agents."""
        p0 = self.participants.get(p0_name)
        p1 = self.participants.get(p1_name)

        if p0 is None or p1 is None:
            return

        key = (p0_name, p1_name)
        if key not in self.payoff_matrix:
            self.payoff_matrix[key] = []
        self.payoff_matrix[key].append((p0_cash, p1_cash))

        if p0_cash > p1_cash:
            p0.record_match(p0_cash, p1_cash, won=True)
            p1.record_match(p1_cash, p0_cash, won=False)
            s0, s1 = 1.0, 0.0
        elif p1_cash > p0_cash:
            p0.record_match(p0_cash, p1_cash, won=False)
            p1.record_match(p1_cash, p0_cash, won=True)
            s0, s1 = 0.0, 1.0
        else:
            p0.record_match(p0_cash, p1_cash, won=None)
            p1.record_match(p1_cash, p0_cash, won=None)
            s0, s1 = 0.5, 0.5

        k_factor = 32.0
        e0 = 1.0 / (1.0 + 10.0 ** ((p1.elo - p0.elo) / 400.0))
        e1 = 1.0 - e0
        p0.elo += k_factor * (s0 - e0)
        p1.elo += k_factor * (s1 - e1)

    def get_league_summary(self) -> Dict[str, Any]:
        """Returns full league leaderboard and head-to-head statistics."""
        leaderboard = []
        for name, p in self.participants.items():
            leaderboard.append({
                "name": name,
                "role": p.role,
                "elo": round(p.elo, 1),
                "matches": p.total_matches,
                "wins": p.wins,
                "losses": p.losses,
                "ties": p.ties,
                "win_rate": round(p.win_rate, 1),
                "mean_cash": round(p.mean_cash, 1),
            })
        leaderboard.sort(key=lambda x: x["elo"], reverse=True)

        payoff_summary = {}
        for (a0, a1), records in self.payoff_matrix.items():
            a0_wins = sum(1 for r in records if r[0] > r[1])
            a1_wins = sum(1 for r in records if r[1] > r[0])
            ties = sum(1 for r in records if r[0] == r[1])
            mean_a0 = np.mean([r[0] for r in records])
            mean_a1 = np.mean([r[1] for r in records])
            payoff_summary[f"{a0} vs {a1}"] = {
                "matches": len(records),
                "record": f"{a0_wins}W-{a1_wins}L-{ties}T",
                "a0_win_rate": round((a0_wins / len(records)) * 100, 1),
                "mean_margin": round(float(mean_a0 - mean_a1), 1),
                "mean_a0_cash": round(float(mean_a0), 1),
                "mean_a1_cash": round(float(mean_a1), 1),
            }

        return {
            "leaderboard": leaderboard,
            "head_to_head": payoff_summary,
            "total_personalities": len(self.sparring_names),
            "total_checkpoints": len(self.checkpoint_names),
        }
