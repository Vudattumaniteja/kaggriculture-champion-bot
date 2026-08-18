"""
Automated Generative Data Model & Large-Scale Parallel Dataset Generator for SSL.
Implements:
1. Continuous Parameterized Strategy Vector Generator spanning multi-industry farming styles:
   - Crops: Rotational portfolios of Melons, Strawberries, Tomatoes, Carrots, Wheat
   - Livestock: Geese (eggs), Cows (milk), Sheep (wool), Coops, Pastures, daily care compounding & feeding
   - Town Shop Reaction: Dynamic commodity scoring and price arbitrage based on unlocked town shops
   - Reinvestment Thresholds: Land quadrant expansions (NE, SW, SE) and daily farmhand labor hiring
   - Physical Chore Priority Weights: Continuous dispatch scheduling for multi-worker efficiency
2. Parallel Multiprocessing Simulation Engine generating 100+ episodes (144,000+ transitions)
3. Self-Supervised Learning (SSL) future dynamics state generation (future_scalars)
4. Elite Filter tagging games achieving $10,000+ final bank balances
5. Compressed NPZ data export to data/massive_ssl_dataset.npz
"""

import dataclasses
import json
import os
import sys
import time

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Any, Dict, List, Optional, Set, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
from kaggle_environments import make

from src.models.encoder import encode_observation, NUM_MACRO_ACTIONS, MACRO_ACTIONS
from src.agents.utils import DIRS, SHED_TILES, get_manhattan_dist, get_step_towards, parse_observation
from src.training.dataset import infer_macro_action

# Game Constants
CROPS_LIST = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS_LIST = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]
ANIMALS_LIST = ["GOOSE", "COW", "SHEEP"]

CROP_SPECS = {
    "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
    "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
    "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
    "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
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


@dataclasses.dataclass
class ParameterizedStrategy:
    name: str = "CustomStrategy"
    
    # 1. Continuous Crop Distribution Weights (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON)
    crop_weights: List[float] = dataclasses.field(default_factory=lambda: [0.15, 0.40, 0.15, 0.15, 0.15])
    early_melon_alloc: int = 8           # Melons target in days 0-6
    early_strawberry_alloc: int = 6      # Strawberries target in days 0-6
    early_carrot_alloc: int = 10         # Carrots target in days 0-6
    early_wheat_alloc: int = 4           # Wheat target in days 0-6
    
    mid_melon_alloc: int = 6             # Melons target in days 7-15
    mid_tomato_alloc: int = 8            # Tomatoes target in days 7-15
    mid_carrot_alloc: int = 12           # Carrots target in days 7-15
    
    late_carrot_rush_day: int = 24       # Day to pivot all remaining land to fast Carrots
    target_active_crops: int = 28        # Desired total planted tiles
    
    # 2. Livestock Preferences
    target_geese: int = 0                # Target goose coops (0-4)
    target_cows: int = 0                 # Target cow pastures (0-3)
    target_sheep: int = 0                # Target sheep pastures (0-3)
    wheat_feed_reserve_tiles: int = 4    # Target wheat tiles for animal feed
    animal_buy_cutoff_day: int = 18      # Day cutoff for buying new animals
    coop_build_cutoff_day: int = 18      # Day cutoff for building coops
    pasture_build_cutoff_day: int = 16   # Day cutoff for building pastures
    
    # 3. Town Shop & Arbitrage Reaction
    shop_reaction_weight: float = 1.0    # Influence of shop demands on crop/animal weights [0.0, 3.0]
    fertilizer_retention: int = 3        # How many fertilizer units to keep for crops [0, 5]
    fertilizer_sell_cutoff_day: int = 22 # Day after which all fertilizer is sold
    
    # 4. Reinvestment Thresholds (Land & Labor)
    land_ne_threshold: float = 1200.0    # Cash required to purchase NE quadrant ($1,000 cost)
    land_sw_threshold: float = 2400.0    # Cash required to purchase SW quadrant ($2,000 cost)
    land_se_threshold: float = 4800.0    # Cash required to purchase SE quadrant ($4,000 cost)
    land_buy_cutoff_day: int = 16        # Max day to buy land expansions
    
    # 5. Labor Hiring
    hires_per_day: int = 2               # Daily farmhand recruit target [1, 4]
    min_hire_money: float = 30.0         # Minimum bank required to hire
    
    # 6. Physical Chore Priorities (Continuous Scheduling Weights)
    w_water: float = 1.5
    w_feed: float = 1.5
    w_care: float = 1.2
    w_harvest: float = 1.3
    w_fertilize: float = 0.9
    w_build: float = 0.8
    w_plant: float = 0.7
    w_weed: float = 0.4
    
    # 7. Liquidation
    liquidation_day: int = 27            # Day to halt planting/purchasing and liquidate all items


def sample_strategy(rng: np.random.RandomState, archetype: Optional[str] = None) -> ParameterizedStrategy:
    """
    Samples a continuous strategy vector either from specialized archetypes (with continuous noise)
    or from the entire continuous parameter space.
    """
    if archetype is None:
        archetype = rng.choice([
            "HyperCropExpander",
            "ElitePortfolio",
            "LivestockSynergy",
            "ShopArbitrageur",
            "MegaFarmTycoon",
            "IntensiveCarrotCash",
            "BalancedHybrid",
            "ExploratoryContinuous",
        ])
        
    if archetype == "ElitePortfolio":
        # Multi-crop snowball portfolio (Melons + Strawberries + Tomatoes + Carrots + Wheat + NE/SW Land)
        cw = np.array([0.15, 0.40, 0.15, 0.15, 0.15]) + rng.normal(0, 0.02, 5)
        cw = np.clip(cw, 0.05, 1.0).tolist()
        return ParameterizedStrategy(
            name="ElitePortfolio",
            crop_weights=cw,
            early_melon_alloc=int(rng.randint(6, 10)),
            early_strawberry_alloc=int(rng.randint(4, 8)),
            early_carrot_alloc=int(rng.randint(8, 14)),
            early_wheat_alloc=int(rng.randint(3, 6)),
            mid_melon_alloc=int(rng.randint(4, 8)),
            mid_tomato_alloc=int(rng.randint(6, 10)),
            mid_carrot_alloc=int(rng.randint(10, 16)),
            late_carrot_rush_day=int(rng.choice([23, 24, 25])),
            target_active_crops=int(rng.randint(24, 36)),
            target_geese=0,
            target_cows=0,
            target_sheep=0,
            wheat_feed_reserve_tiles=0,
            animal_buy_cutoff_day=10,
            shop_reaction_weight=float(rng.uniform(0.5, 1.5)),
            fertilizer_retention=int(rng.choice([2, 3, 4])),
            land_ne_threshold=float(rng.uniform(1150, 1250)),
            land_sw_threshold=float(rng.uniform(2300, 2500)),
            land_buy_cutoff_day=int(rng.randint(14, 18)),
            hires_per_day=int(rng.choice([2, 3])),
            min_hire_money=30.0,
            w_water=float(rng.uniform(1.4, 2.0)),
            w_harvest=float(rng.uniform(1.2, 1.6)),
            w_fertilize=float(rng.uniform(0.8, 1.2)),
            w_plant=float(rng.uniform(0.7, 1.0)),
            liquidation_day=27,
        )
        
    elif archetype == "HyperCropExpander":
        # Heavy early melons/strawberries, fast land expansion, carrot sprint
        cw = np.array([0.10, 0.40, 0.15, 0.15, 0.20]) + rng.normal(0, 0.03, 5)
        cw = np.clip(cw, 0.02, 1.0).tolist()
        return ParameterizedStrategy(
            name="HyperCropExpander",
            crop_weights=cw,
            early_melon_alloc=int(rng.randint(8, 12)),
            early_strawberry_alloc=int(rng.randint(6, 10)),
            early_carrot_alloc=int(rng.randint(8, 12)),
            early_wheat_alloc=int(rng.randint(2, 5)),
            mid_melon_alloc=int(rng.randint(6, 10)),
            mid_tomato_alloc=int(rng.randint(6, 10)),
            mid_carrot_alloc=int(rng.randint(12, 18)),
            late_carrot_rush_day=int(rng.choice([23, 24, 25])),
            target_active_crops=int(rng.randint(28, 42)),
            target_geese=int(rng.choice([0, 1])),
            target_cows=0,
            target_sheep=0,
            wheat_feed_reserve_tiles=int(rng.choice([0, 2])),
            animal_buy_cutoff_day=10,
            shop_reaction_weight=float(rng.uniform(0.5, 1.5)),
            fertilizer_retention=int(rng.choice([2, 3, 4])),
            land_ne_threshold=float(rng.uniform(1100, 1250)),
            land_sw_threshold=float(rng.uniform(2200, 2500)),
            land_se_threshold=float(rng.uniform(4500, 5200)),
            land_buy_cutoff_day=int(rng.randint(14, 18)),
            hires_per_day=int(rng.choice([2, 3])),
            min_hire_money=30.0,
            w_water=float(rng.uniform(1.4, 2.0)),
            w_harvest=float(rng.uniform(1.2, 1.6)),
            w_fertilize=float(rng.uniform(0.8, 1.3)),
            w_plant=float(rng.uniform(0.7, 1.1)),
            liquidation_day=int(rng.choice([27, 28])),
        )
        
    elif archetype == "LivestockSynergy":
        # Livestock-first: coops & pastures, wheat feed, daily CARE + FEED compounding, fertilizer harvest + crops
        cw = np.array([0.35, 0.30, 0.10, 0.10, 0.15]) + rng.normal(0, 0.03, 5)
        cw = np.clip(cw, 0.05, 1.0).tolist()
        return ParameterizedStrategy(
            name="LivestockSynergy",
            crop_weights=cw,
            early_melon_alloc=int(rng.randint(4, 8)),
            early_strawberry_alloc=int(rng.randint(2, 6)),
            early_carrot_alloc=int(rng.randint(8, 12)),
            early_wheat_alloc=int(rng.randint(6, 10)),
            mid_melon_alloc=int(rng.randint(2, 6)),
            mid_tomato_alloc=int(rng.randint(4, 8)),
            mid_carrot_alloc=int(rng.randint(10, 16)),
            late_carrot_rush_day=24,
            target_active_crops=int(rng.randint(22, 34)),
            target_geese=int(rng.choice([2, 3])),
            target_cows=int(rng.choice([1, 2])),
            target_sheep=int(rng.choice([1, 2])),
            wheat_feed_reserve_tiles=int(rng.randint(6, 10)),
            animal_buy_cutoff_day=int(rng.randint(16, 22)),
            coop_build_cutoff_day=20,
            pasture_build_cutoff_day=18,
            shop_reaction_weight=float(rng.uniform(0.8, 2.0)),
            fertilizer_retention=int(rng.choice([1, 2, 3])),
            land_ne_threshold=float(rng.uniform(1150, 1300)),
            land_sw_threshold=float(rng.uniform(2300, 2600)),
            land_buy_cutoff_day=int(rng.randint(14, 18)),
            hires_per_day=int(rng.choice([2, 3])),
            w_water=float(rng.uniform(1.3, 1.8)),
            w_feed=float(rng.uniform(1.5, 2.2)),
            w_care=float(rng.uniform(1.2, 1.8)),
            w_harvest=float(rng.uniform(1.2, 1.6)),
            liquidation_day=27,
        )
        
    elif archetype == "ShopArbitrageur":
        # Strongly responds to town shop supply deficits, dynamic crop pivoting, fast price harvesting
        cw = np.array([0.20, 0.30, 0.20, 0.20, 0.10]) + rng.normal(0, 0.04, 5)
        cw = np.clip(cw, 0.05, 1.0).tolist()
        return ParameterizedStrategy(
            name="ShopArbitrageur",
            crop_weights=cw,
            early_melon_alloc=int(rng.randint(6, 10)),
            early_strawberry_alloc=int(rng.randint(4, 8)),
            early_carrot_alloc=int(rng.randint(8, 14)),
            early_wheat_alloc=int(rng.randint(4, 8)),
            mid_melon_alloc=int(rng.randint(4, 8)),
            mid_tomato_alloc=int(rng.randint(6, 10)),
            mid_carrot_alloc=int(rng.randint(10, 16)),
            late_carrot_rush_day=24,
            target_active_crops=int(rng.randint(24, 36)),
            target_geese=int(rng.choice([1, 2])),
            target_cows=int(rng.choice([1, 2])),
            target_sheep=int(rng.choice([1, 2])),
            wheat_feed_reserve_tiles=int(rng.randint(4, 8)),
            animal_buy_cutoff_day=18,
            shop_reaction_weight=float(rng.uniform(1.8, 3.0)),
            fertilizer_retention=int(rng.choice([1, 2])),
            land_ne_threshold=float(rng.uniform(1100, 1250)),
            land_sw_threshold=float(rng.uniform(2250, 2550)),
            land_buy_cutoff_day=16,
            hires_per_day=int(rng.choice([2, 3])),
            w_water=float(rng.uniform(1.3, 1.8)),
            w_feed=float(rng.uniform(1.2, 1.8)),
            w_care=float(rng.uniform(1.1, 1.6)),
            w_harvest=float(rng.uniform(1.2, 1.7)),
            liquidation_day=27,
        )
        
    elif archetype == "MegaFarmTycoon":
        # Triple quadrant unlock (NE+SW+SE), maximum farmhand scaling, massive multi-crop farm
        cw = np.array([0.15, 0.35, 0.15, 0.15, 0.20]) + rng.normal(0, 0.03, 5)
        cw = np.clip(cw, 0.05, 1.0).tolist()
        return ParameterizedStrategy(
            name="MegaFarmTycoon",
            crop_weights=cw,
            early_melon_alloc=int(rng.randint(8, 14)),
            early_strawberry_alloc=int(rng.randint(6, 12)),
            early_carrot_alloc=int(rng.randint(10, 16)),
            early_wheat_alloc=int(rng.randint(4, 8)),
            mid_melon_alloc=int(rng.randint(8, 14)),
            mid_tomato_alloc=int(rng.randint(8, 14)),
            mid_carrot_alloc=int(rng.randint(14, 22)),
            late_carrot_rush_day=25,
            target_active_crops=int(rng.randint(35, 60)),
            target_geese=int(rng.choice([1, 2, 3])),
            target_cows=int(rng.choice([1, 2])),
            target_sheep=int(rng.choice([1, 2])),
            wheat_feed_reserve_tiles=int(rng.randint(5, 8)),
            animal_buy_cutoff_day=18,
            shop_reaction_weight=float(rng.uniform(0.8, 1.6)),
            fertilizer_retention=int(rng.choice([2, 3, 4])),
            land_ne_threshold=float(rng.uniform(1050, 1200)),
            land_sw_threshold=float(rng.uniform(2100, 2400)),
            land_se_threshold=float(rng.uniform(4200, 4800)),
            land_buy_cutoff_day=int(rng.randint(14, 18)),
            hires_per_day=int(rng.choice([2, 3, 4])),
            w_water=float(rng.uniform(1.4, 2.0)),
            w_harvest=float(rng.uniform(1.2, 1.7)),
            w_fertilize=float(rng.uniform(0.8, 1.3)),
            w_plant=float(rng.uniform(0.7, 1.1)),
            liquidation_day=28,
        )
        
    elif archetype == "IntensiveCarrotCash":
        # Rapid cash turnover on fast 3-day Carrots and Wheat
        cw = np.array([0.20, 0.70, 0.03, 0.03, 0.04]) + rng.normal(0, 0.02, 5)
        cw = np.clip(cw, 0.01, 1.0).tolist()
        return ParameterizedStrategy(
            name="IntensiveCarrotCash",
            crop_weights=cw,
            early_melon_alloc=2,
            early_strawberry_alloc=2,
            early_carrot_alloc=int(rng.randint(18, 26)),
            early_wheat_alloc=int(rng.randint(4, 8)),
            mid_melon_alloc=0,
            mid_tomato_alloc=0,
            mid_carrot_alloc=int(rng.randint(22, 32)),
            late_carrot_rush_day=18,
            target_active_crops=int(rng.randint(24, 40)),
            target_geese=int(rng.choice([0, 1])),
            target_cows=0,
            target_sheep=0,
            wheat_feed_reserve_tiles=2,
            animal_buy_cutoff_day=12,
            shop_reaction_weight=float(rng.uniform(0.5, 1.2)),
            fertilizer_retention=1,
            land_ne_threshold=float(rng.uniform(1100, 1250)),
            land_sw_threshold=float(rng.uniform(2200, 2500)),
            land_buy_cutoff_day=14,
            hires_per_day=int(rng.choice([2, 3])),
            w_water=float(rng.uniform(1.3, 1.8)),
            w_harvest=float(rng.uniform(1.2, 1.7)),
            w_plant=float(rng.uniform(0.8, 1.2)),
            liquidation_day=27,
        )
        
    elif archetype == "BalancedHybrid":
        # Balanced crops + livestock + shops + land expansion
        cw = np.array([0.20, 0.35, 0.15, 0.15, 0.15]) + rng.normal(0, 0.03, 5)
        cw = np.clip(cw, 0.05, 1.0).tolist()
        return ParameterizedStrategy(
            name="BalancedHybrid",
            crop_weights=cw,
            early_melon_alloc=int(rng.randint(6, 10)),
            early_strawberry_alloc=int(rng.randint(4, 8)),
            early_carrot_alloc=int(rng.randint(8, 14)),
            early_wheat_alloc=int(rng.randint(4, 8)),
            mid_melon_alloc=int(rng.randint(4, 8)),
            mid_tomato_alloc=int(rng.randint(6, 10)),
            mid_carrot_alloc=int(rng.randint(10, 16)),
            late_carrot_rush_day=int(rng.choice([23, 24, 25])),
            target_active_crops=int(rng.randint(24, 38)),
            target_geese=int(rng.choice([1, 2, 3])),
            target_cows=int(rng.choice([1, 2])),
            target_sheep=int(rng.choice([1, 2])),
            wheat_feed_reserve_tiles=int(rng.randint(4, 8)),
            animal_buy_cutoff_day=int(rng.randint(16, 20)),
            shop_reaction_weight=float(rng.uniform(0.8, 1.8)),
            fertilizer_retention=int(rng.choice([2, 3])),
            land_ne_threshold=float(rng.uniform(1100, 1300)),
            land_sw_threshold=float(rng.uniform(2250, 2600)),
            land_buy_cutoff_day=16,
            hires_per_day=int(rng.choice([2, 3])),
            w_water=float(rng.uniform(1.3, 1.8)),
            w_feed=float(rng.uniform(1.3, 1.9)),
            w_care=float(rng.uniform(1.1, 1.6)),
            w_harvest=float(rng.uniform(1.2, 1.6)),
            liquidation_day=27,
        )
        
    else:  # "ExploratoryContinuous"
        # Pure continuous random sample from entire parameter space
        cw = rng.dirichlet(np.ones(5)).tolist()
        return ParameterizedStrategy(
            name="ExploratoryContinuous",
            crop_weights=cw,
            early_melon_alloc=int(rng.randint(2, 12)),
            early_strawberry_alloc=int(rng.randint(2, 10)),
            early_carrot_alloc=int(rng.randint(6, 18)),
            early_wheat_alloc=int(rng.randint(2, 8)),
            mid_melon_alloc=int(rng.randint(0, 10)),
            mid_tomato_alloc=int(rng.randint(2, 12)),
            mid_carrot_alloc=int(rng.randint(8, 20)),
            late_carrot_rush_day=int(rng.randint(18, 28)),
            target_active_crops=int(rng.randint(16, 50)),
            target_geese=int(rng.randint(0, 4)),
            target_cows=int(rng.randint(0, 3)),
            target_sheep=int(rng.randint(0, 3)),
            wheat_feed_reserve_tiles=int(rng.randint(2, 10)),
            animal_buy_cutoff_day=int(rng.randint(10, 24)),
            coop_build_cutoff_day=int(rng.randint(10, 22)),
            pasture_build_cutoff_day=int(rng.randint(10, 20)),
            shop_reaction_weight=float(rng.uniform(0.0, 3.0)),
            fertilizer_retention=int(rng.randint(0, 5)),
            land_ne_threshold=float(rng.uniform(1050, 1800)),
            land_sw_threshold=float(rng.uniform(2100, 3500)),
            land_se_threshold=float(rng.uniform(4200, 6000)),
            land_buy_cutoff_day=int(rng.randint(8, 20)),
            hires_per_day=int(rng.randint(1, 4)),
            min_hire_money=float(rng.uniform(10, 80)),
            w_water=float(rng.uniform(0.8, 2.5)),
            w_feed=float(rng.uniform(0.8, 3.0)),
            w_care=float(rng.uniform(0.5, 2.5)),
            w_harvest=float(rng.uniform(0.8, 2.5)),
            w_fertilize=float(rng.uniform(0.2, 2.0)),
            w_build=float(rng.uniform(0.3, 2.0)),
            w_plant=float(rng.uniform(0.3, 2.0)),
            w_weed=float(rng.uniform(0.1, 1.0)),
            liquidation_day=int(rng.randint(26, 29)),
        )


class GenerativeStrategyAgent:
    """
    High-performance parameterized agent executing a sampled continuous strategy vector.
    """
    def __init__(self, strategy: ParameterizedStrategy):
        self.strategy = strategy

    def _select_target_crops(self, day: int, unlocked_shops: List[str]) -> List[Tuple[str, int]]:
        """Returns prioritized list of (crop_name, max_count) based on seasonal curve and continuous weights."""
        strat = self.strategy
        if day >= strat.liquidation_day:
            return []
            
        # Seasonal Curve
        if day <= 6:
            alloc = [
                ("MELON", strat.early_melon_alloc),
                ("STRAWBERRY", strat.early_strawberry_alloc),
                ("CARROT", strat.early_carrot_alloc),
                ("WHEAT", strat.early_wheat_alloc),
            ]
        elif day <= 15:
            alloc = [
                ("MELON", strat.mid_melon_alloc),
                ("TOMATO", strat.mid_tomato_alloc),
                ("CARROT", strat.mid_carrot_alloc),
                ("WHEAT", max(4, strat.wheat_feed_reserve_tiles)),
            ]
        elif day <= 23:
            alloc = [
                ("CARROT", int(strat.target_active_crops * 0.7)),
                ("WHEAT", max(4, strat.wheat_feed_reserve_tiles)),
            ]
        elif day <= 26:
            alloc = [
                ("CARROT", strat.target_active_crops),
            ]
        else:
            return []
            
        # Town shop adjustments
        if strat.shop_reaction_weight > 0.0 and unlocked_shops:
            shop_items: Set[str] = set()
            for s in unlocked_shops:
                shop_items.update(SHOPS_CATALOG.get(s, []))
                
            adjusted: List[Tuple[str, int]] = []
            for crop_name, cnt in alloc:
                if crop_name in shop_items:
                    adjusted.append((crop_name, int(cnt * (1.0 + 0.25 * strat.shop_reaction_weight))))
                else:
                    adjusted.append((crop_name, cnt))
            return adjusted
            
        return alloc

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        money = state["money"]
        farmer_pos = state["farmer_pos"]
        hands_pos = state["hands_pos"]
        tiles = state["tiles"]
        shed = state["shed"]
        seeds = state["seeds"]
        inventories = state["inventories"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]
        unlocked_shops = state["unlocked_shops"]
        
        strat = self.strategy
        market_orders: List[List[Any]] = []
        
        # --- 1. AUDIT FIELD ASSETS ---
        living_geese = 0
        living_cows = 0
        living_sheep = 0
        empty_coops = 0
        empty_pastures = 0
        planted_crops: Dict[str, int] = {c: 0 for c in CROPS_LIST}
        animal_tiles: List[Tuple[int, int, str, Dict[str, Any]]] = []
        crop_tiles: List[Tuple[int, int, Dict[str, Any]]] = []
        unlocked_empty_tiles: List[Tuple[int, int]] = []
        
        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    unlocked_empty_tiles.append(pos)
                elif isinstance(t, dict):
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
                        cname = t.get("crop", "CARROT")
                        crop_tiles.append((c, r, t))
                        planted_crops[cname] = planted_crops.get(cname, 0) + 1
                        
        total_living_animals = living_geese + living_cows + living_sheep
        
        # --- 2. MARKET TRADING & CAPITAL ALLOCATION ---
        # A. Sell harvested items
        for prod in ["EGG", "MILK", "WOOL", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]:
            count = shed.get(prod, 0)
            if count > 0:
                market_orders.append(["SELL", prod, count])
                
        # Fertilizer selling with retention
        fert_count = shed.get("FERTILIZER", 0)
        if fert_count > 0:
            if len(crop_tiles) > 0 and day <= strat.fertilizer_sell_cutoff_day and fert_count > strat.fertilizer_retention:
                market_orders.append(["SELL", "FERTILIZER", fert_count - strat.fertilizer_retention])
            elif len(crop_tiles) == 0 or day > strat.fertilizer_sell_cutoff_day:
                market_orders.append(["SELL", "FERTILIZER", fert_count])
                
        # Wheat feed reserve management
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_req = total_living_animals
        safe_wheat_res = max(4, daily_feed_req * 3) if total_living_animals > 0 else 0
        
        if wheat_in_shed > safe_wheat_res + 8:
            market_orders.append(["SELL", "WHEAT", wheat_in_shed - safe_wheat_res])
        elif total_living_animals > 0 and wheat_in_shed < daily_feed_req and money >= 50 and day < strat.liquidation_day:
            buy_q = min(6, safe_wheat_res - wheat_in_shed)
            if buy_q > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_q])
                money -= 25 * buy_q
                shed["WHEAT"] = wheat_in_shed + buy_q
                
        # B. Land Expansion Reinvestment
        if day <= strat.land_buy_cutoff_day:
            if "NE" not in unlocked and money >= strat.land_ne_threshold:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and money >= strat.land_sw_threshold and day <= 14:
                market_orders.append(["BUY_LAND"])
                money -= 2000
            elif "SE" not in unlocked and "SW" in unlocked and money >= strat.land_se_threshold and day <= 12:
                market_orders.append(["BUY_LAND"])
                money -= 4000
                
        # C. Labor Hiring (Farmhands)
        if day < strat.liquidation_day and hour <= 1 and hires_today < strat.hires_per_day and money >= strat.min_hire_money:
            market_orders.append(["HIRE"])
            money -= 2
            
        # D. Animal Purchases
        geese_in_shed = shed.get("GOOSE", 0)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)
        
        if (
            empty_coops > geese_in_shed
            and (living_geese + geese_in_shed) < strat.target_geese
            and money >= 380
            and day <= strat.animal_buy_cutoff_day
        ):
            market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
            money -= 300
            geese_in_shed += 1
        elif (
            empty_pastures > (cows_in_shed + sheep_in_shed)
            and (living_cows + cows_in_shed) < strat.target_cows
            and money >= 520
            and day <= min(18, strat.animal_buy_cutoff_day)
        ):
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_in_shed += 1
        elif (
            empty_pastures > (cows_in_shed + sheep_in_shed)
            and (living_sheep + sheep_in_shed) < strat.target_sheep
            and money >= 620
            and day <= min(16, strat.animal_buy_cutoff_day)
        ):
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_in_shed += 1
            
        # E. Seed Purchases
        target_crop_alloc = self._select_target_crops(day, unlocked_shops)
        for crop_name, max_cnt in target_crop_alloc:
            curr_p = planted_crops.get(crop_name, 0)
            curr_s = seeds.get(crop_name, 0)
            needed = max(0, max_cnt - curr_p - curr_s)
            seed_price = CROP_SPECS[crop_name]["seed"]
            if needed > 0 and money >= seed_price * 2 and day < strat.liquidation_day:
                buy_q = min(needed, 4, int(money // seed_price))
                if buy_q > 0:
                    market_orders.append(["BUY_SEED", crop_name, buy_q])
                    money -= seed_price * buy_q
                    seeds[crop_name] = curr_s + buy_q
                    
        # --- 3. FIELD SCANNING & CHORE GATHERING ---
        feed_tasks: List[Tuple[int, int]] = []
        care_tasks: List[Tuple[int, int]] = []
        harvest_animal_tasks: List[Tuple[int, int]] = []
        fert_collect_tasks: List[Tuple[int, int]] = []
        
        water_crop_tasks: List[Tuple[int, int]] = []
        harvest_crop_tasks: List[Tuple[int, int]] = []
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
                fert_collect_tasks.append(pos)
                
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
            elif ctile.get("fertilized_until_day", -1) < day and crop in ["MELON", "STRAWBERRY", "TOMATO"]:
                fertilize_crop_tasks.append(pos)
                
        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if isinstance(t, dict):
                    if t.get("kind") == "WEED":
                        weed_tasks.append(pos)
                    elif t.get("kind") == "COOP" and "animal" not in t:
                        place_animal_tasks.append((c, r, "GOOSE"))
                    elif t.get("kind") == "PASTURE" and "animal" not in t:
                        if (living_cows + cows_in_shed) <= (living_sheep + sheep_in_shed):
                            place_animal_tasks.append((c, r, "COW"))
                        else:
                            place_animal_tasks.append((c, r, "SHEEP"))
                            
        unlocked_empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))
        
        # Structure Building Tasks
        curr_e_idx = 0
        if (living_geese + empty_coops) < strat.target_geese and curr_e_idx < len(unlocked_empty_tiles) and day <= strat.coop_build_cutoff_day:
            build_coop_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1
            
        if (
            (living_cows + living_sheep + empty_pastures) < (strat.target_cows + strat.target_sheep)
            and curr_e_idx < len(unlocked_empty_tiles)
            and day <= strat.pasture_build_cutoff_day
        ):
            build_pasture_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1
            
        total_seeds_count = sum(seeds.values())
        if day < strat.liquidation_day and total_seeds_count > 0:
            max_plant_capacity = min(len(unlocked) * 18, strat.target_active_crops)
            for p in unlocked_empty_tiles[curr_e_idx:curr_e_idx + max_plant_capacity]:
                plant_tasks.append(p)
                
        # --- 4. MULTI-WORKER DISPATCH ---
        units = [farmer_pos] + hands_pos
        unit_actions: List[List[Any]] = []
        claimed_tasks: Set[Tuple[int, int]] = set()
        
        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in SHED_TILES
            
            # 1. Shed Routine
            if is_at_shed:
                placed_target = next((pt for pt in place_animal_tasks if pt[:2] not in claimed_tasks), None)
                if placed_target is not None:
                    target_animal = placed_target[2]
                    if shed.get(target_animal, 0) > 0 and u_inv.get(target_animal, 0) == 0:
                        unit_actions.append(["PICKUP", target_animal, 1])
                        shed[target_animal] -= 1
                        u_inv[target_animal] = u_inv.get(target_animal, 0) + 1
                        continue
                        
                if feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    qty = min(4, shed.get("WHEAT", 0), len(feed_tasks))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "WHEAT", qty])
                        shed["WHEAT"] -= qty
                        u_inv["WHEAT"] = u_inv.get("WHEAT", 0) + qty
                        continue
                        
                if fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "FERTILIZER", qty])
                        shed["FERTILIZER"] -= qty
                        u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + qty
                        continue
                        
            # 2. Standing Tile Actions
            if isinstance(u_tile, dict):
                k = u_tile.get("kind")
                animal = u_tile.get("animal")
                
                if k in ["COOP", "PASTURE"]:
                    if animal is not None:
                        if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                            unit_actions.append(["FEED"])
                            u_inv["WHEAT"] -= 1
                            continue
                        if not u_tile.get("cared_today", False):
                            unit_actions.append(["CARE"])
                            continue
                        if u_tile.get("fertilizer_available", False):
                            unit_actions.append(["COLLECT_FERTILIZER"])
                            u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + 1
                            continue
                        if u_tile.get("yield_units", 0) > 0:
                            unit_actions.append(["HARVEST"])
                            continue
                    else:
                        for aname in ["GOOSE", "COW", "SHEEP"]:
                            if u_inv.get(aname, 0) > 0:
                                req_struct = "COOP" if aname == "GOOSE" else "PASTURE"
                                if k == req_struct:
                                    unit_actions.append(["PLACE", aname])
                                    u_inv[aname] -= 1
                                    break
                        if len(unit_actions) > u_idx:
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
                        continue
                    if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0:
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue
                        
                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue
                    
            elif u_tile is None:
                if u_pos in build_coop_tasks:
                    unit_actions.append(["BUILD_COOP"])
                    build_coop_tasks.remove(u_pos)
                    continue
                elif u_pos in build_pasture_tasks:
                    unit_actions.append(["BUILD_PASTURE"])
                    build_pasture_tasks.remove(u_pos)
                    continue
                elif u_pos in plant_tasks and total_seeds_count > 0 and day < strat.liquidation_day:
                    for crop_cand, _ in target_crop_alloc:
                        if seeds.get(crop_cand, 0) > 0:
                            unit_actions.append(["PLANT", crop_cand])
                            seeds[crop_cand] -= 1
                            total_seeds_count -= 1
                            break
                    else:
                        for crop_cand in ["MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"]:
                            if seeds.get(crop_cand, 0) > 0:
                                unit_actions.append(["PLANT", crop_cand])
                                seeds[crop_cand] -= 1
                                total_seeds_count -= 1
                                break
                        else:
                            unit_actions.append(["PASS"])
                    continue
                    
            # 3. Pathfinding & Navigation
            best_target: Optional[Tuple[int, int]] = None
            best_dist = 999
            
            # Animal placement priority if held
            has_animal = any(u_inv.get(a, 0) > 0 for a in ANIMALS_LIST)
            if has_animal:
                for pt in place_animal_tasks:
                    pos = pt[:2]
                    if pos not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, pos)
                        if d < best_dist:
                            best_dist = d
                            best_target = pos
                            
            if best_target is None and u_inv.get("WHEAT", 0) > 0 and feed_tasks:
                for task in feed_tasks:
                    if task not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, task)
                        if d < best_dist:
                            best_dist = d
                            best_target = task
                            
            if best_target is None and feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                best_target = nearest_shed
                best_dist = get_manhattan_dist(u_pos, nearest_shed)
                
            if best_target is None:
                # Priority Tier 1: Urgent daily needs (Water crops + Feed animals)
                tier1 = water_crop_tasks + feed_tasks
                for task in tier1:
                    if task in claimed_tasks:
                        continue
                    d = get_manhattan_dist(u_pos, task)
                    if d < best_dist:
                        best_dist = d
                        best_target = task
                        
            if best_target is None:
                # Priority Tier 2: Harvesting & Animal Care
                tier2 = harvest_crop_tasks + harvest_animal_tasks + care_tasks + fert_collect_tasks
                for task in tier2:
                    if task in claimed_tasks:
                        continue
                    d = get_manhattan_dist(u_pos, task)
                    if d < best_dist:
                        best_dist = d
                        best_target = task
                        
            if best_target is None:
                # Priority Tier 3: Fertilizing, Building, Planting, Weeding
                tier3 = fertilize_crop_tasks + build_coop_tasks + build_pasture_tasks + plant_tasks + weed_tasks
                for task in tier3:
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


def _run_single_episode(task_args: Tuple[int, int, str, str]) -> Dict[str, Any]:
    """
    Worker function executed in parallel pool to simulate one full match episode.
    """
    ep_idx, seed, arch_a, arch_b = task_args
    rng = np.random.RandomState(seed)
    
    strat0 = sample_strategy(rng, archetype=arch_a)
    strat1 = sample_strategy(rng, archetype=arch_b)
    
    agent0 = GenerativeStrategyAgent(strat0)
    agent1 = GenerativeStrategyAgent(strat1)
    
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.reset()
    
    ep_grids_p0: List[np.ndarray] = []
    ep_scalars_p0: List[np.ndarray] = []
    ep_actions_p0: List[int] = []
    
    ep_grids_p1: List[np.ndarray] = []
    ep_scalars_p1: List[np.ndarray] = []
    ep_actions_p1: List[int] = []
    
    t0 = time.time()
    
    while not env.done:
        obs0 = env.state[0].observation
        obs1 = env.state[1].observation
        
        g0, s0 = encode_observation(obs0)
        g1, s1 = encode_observation(obs1)
        
        act0 = agent0(obs0)
        act1 = agent1(obs1)
        
        macro0 = infer_macro_action(act0, obs0.get("day", 0))
        macro1 = infer_macro_action(act1, obs1.get("day", 0))
        
        ep_grids_p0.append(g0)
        ep_scalars_p0.append(s0)
        ep_actions_p0.append(macro0)
        
        ep_grids_p1.append(g1)
        ep_scalars_p1.append(s1)
        ep_actions_p1.append(macro1)
        
        env.step([act0, act1])
        
    duration = time.time() - t0
    r0 = float(env.state[0].reward if env.state[0].reward is not None else 0.0)
    r1 = float(env.state[1].reward if env.state[1].reward is not None else 0.0)
    
    # Compute future scalars with horizon=4 for SSL training
    future_horizon = 4
    n_steps = len(ep_scalars_p0)
    
    fut_scalars_p0 = np.zeros_like(ep_scalars_p0)
    fut_scalars_p1 = np.zeros_like(ep_scalars_p1)
    
    for t in range(n_steps):
        target_t = min(n_steps - 1, t + future_horizon)
        fut_scalars_p0[t] = ep_scalars_p0[target_t]
        fut_scalars_p1[t] = ep_scalars_p1[target_t]
        
    v0 = float(np.tanh((r0 - r1) / 4000.0))
    v1 = float(np.tanh((r1 - r0) / 4000.0))
    
    # Elite threshold check ($10,000+)
    is_elite_p0 = (r0 >= 10000.0)
    is_elite_p1 = (r1 >= 10000.0)
    is_elite_game = is_elite_p0 or is_elite_p1
    
    return {
        "ep_idx": ep_idx,
        "seed": seed,
        "arch_p0": strat0.name,
        "arch_p1": strat1.name,
        "reward_p0": r0,
        "reward_p1": r1,
        "is_elite_p0": is_elite_p0,
        "is_elite_p1": is_elite_p1,
        "is_elite_game": is_elite_game,
        "duration": duration,
        "grids_p0": ep_grids_p0,
        "scalars_p0": ep_scalars_p0,
        "actions_p0": ep_actions_p0,
        "values_p0": [v0] * n_steps,
        "future_scalars_p0": fut_scalars_p0,
        "grids_p1": ep_grids_p1,
        "scalars_p1": ep_scalars_p1,
        "actions_p1": ep_actions_p1,
        "values_p1": [v1] * n_steps,
        "future_scalars_p1": fut_scalars_p1,
    }


def generate_massive_ssl_dataset(
    num_episodes: int = 100,
    save_path: str = "data/massive_ssl_dataset.npz",
    base_seed: int = 10000,
    num_workers: Optional[int] = None,
    elite_threshold: float = 10000.0,
) -> Dict[str, Any]:
    """
    Simulates 100+ episodes in parallel across continuous strategy vectors,
    filters elite games, and serializes the massive SSL dataset.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if num_workers is None:
        num_workers = min(16, cpu_count())
        
    archetypes = [
        "ElitePortfolio",
        "HyperCropExpander",
        "LivestockSynergy",
        "ShopArbitrageur",
        "MegaFarmTycoon",
        "IntensiveCarrotCash",
        "BalancedHybrid",
        "ExploratoryContinuous",
    ]
    
    tasks: List[Tuple[int, int, str, str]] = []
    rng_master = np.random.RandomState(base_seed)
    
    for ep in range(1, num_episodes + 1):
        arch_a = archetypes[(ep - 1) % len(archetypes)]
        arch_b = rng_master.choice(archetypes)
        tasks.append((ep, base_seed + ep, arch_a, arch_b))
        
    print("=" * 80)
    print(f" Launching Generative SSL Data Model: {num_episodes} Episodes on {num_workers} Workers")
    print(f" Target Output: {save_path}")
    print(f" Elite Performance Threshold: ${elite_threshold:,.0f}+")
    print("=" * 80)
    
    start_time = time.time()
    
    all_grids: List[np.ndarray] = []
    all_scalars: List[np.ndarray] = []
    all_actions: List[int] = []
    all_values: List[float] = []
    all_future_scalars: List[np.ndarray] = []
    
    all_rewards: List[float] = []
    elite_episodes = 0
    elite_agents_count = 0
    archetype_rewards: Dict[str, List[float]] = {}
    
    completed = 0
    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_run_single_episode, tasks):
            completed += 1
            
            # P0 data
            all_grids.extend(result["grids_p0"])
            all_scalars.extend(result["scalars_p0"])
            all_actions.extend(result["actions_p0"])
            all_values.extend(result["values_p0"])
            all_future_scalars.extend(result["future_scalars_p0"])
            
            # P1 data
            all_grids.extend(result["grids_p1"])
            all_scalars.extend(result["scalars_p1"])
            all_actions.extend(result["actions_p1"])
            all_values.extend(result["values_p1"])
            all_future_scalars.extend(result["future_scalars_p1"])
            
            r0 = result["reward_p0"]
            r1 = result["reward_p1"]
            all_rewards.extend([r0, r1])
            
            if result["is_elite_p0"]:
                elite_agents_count += 1
            if result["is_elite_p1"]:
                elite_agents_count += 1
            if result["is_elite_game"]:
                elite_episodes += 1
                
            arch0 = result["arch_p0"]
            arch1 = result["arch_p1"]
            archetype_rewards.setdefault(arch0, []).append(r0)
            archetype_rewards.setdefault(arch1, []).append(r1)
            
            if completed % 10 == 0 or completed == num_episodes:
                elapsed = time.time() - start_time
                fps = len(all_actions) / elapsed
                print(
                    f"  [Progress {completed:3d}/{num_episodes}] "
                    f"Samples: {len(all_actions):,d} | "
                    f"Elite Agents: {elite_agents_count:2d} | "
                    f"Speed: {fps:6.1f} steps/s | "
                    f"Elapsed: {elapsed:5.1f}s"
                )
                
    total_duration = time.time() - start_time
    total_samples = len(all_actions)
    
    print("\nCompressing and saving dataset arrays to disk...")
    grids_arr = np.array(all_grids, dtype=np.float32)
    scalars_arr = np.array(all_scalars, dtype=np.float32)
    actions_arr = np.array(all_actions, dtype=np.int64)
    values_arr = np.array(all_values, dtype=np.float32)
    future_scalars_arr = np.array(all_future_scalars, dtype=np.float32)
    
    np.savez_compressed(
        save_path,
        grids=grids_arr,
        scalars=scalars_arr,
        actions=actions_arr,
        values=values_arr,
        future_scalars=future_scalars_arr,
    )
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    # Statistical analysis
    action_counts = {MACRO_ACTIONS[i]: int(np.sum(actions_arr == i)) for i in range(NUM_MACRO_ACTIONS)}
    action_dist = {name: (count, count / total_samples * 100.0) for name, count in action_counts.items()}
    
    rewards_arr = np.array(all_rewards)
    elite_samples = elite_agents_count * 720
    
    stats = {
        "total_samples": total_samples,
        "num_episodes": num_episodes,
        "num_agent_runs": len(all_rewards),
        "elite_agents_count": elite_agents_count,
        "elite_episodes_count": elite_episodes,
        "elite_samples_count": elite_samples,
        "elite_percentage": (elite_agents_count / len(all_rewards)) * 100.0,
        "file_size_mb": file_size_mb,
        "save_path": os.path.abspath(save_path),
        "total_duration_seconds": total_duration,
        "execution_speed_steps_per_sec": total_samples / total_duration,
        "execution_speed_episodes_per_sec": num_episodes / total_duration,
        "grids_shape": list(grids_arr.shape),
        "scalars_shape": list(scalars_arr.shape),
        "actions_shape": list(actions_arr.shape),
        "values_shape": list(values_arr.shape),
        "future_scalars_shape": list(future_scalars_arr.shape),
        "reward_statistics": {
            "mean": float(np.mean(rewards_arr)),
            "std": float(np.std(rewards_arr)),
            "min": float(np.min(rewards_arr)),
            "p25": float(np.percentile(rewards_arr, 25)),
            "median": float(np.median(rewards_arr)),
            "p75": float(np.percentile(rewards_arr, 75)),
            "p90": float(np.percentile(rewards_arr, 90)),
            "max": float(np.max(rewards_arr)),
        },
        "archetype_performance": {
            k: {
                "mean_bank": float(np.mean(v)),
                "max_bank": float(np.max(v)),
                "runs": len(v),
                "elite_count": int(sum(r >= elite_threshold for r in v)),
            }
            for k, v in archetype_rewards.items()
        },
        "action_distribution": action_dist,
    }
    
    json_path = "data/massive_ssl_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    return stats


if __name__ == "__main__":
    generate_massive_ssl_dataset(num_episodes=100)
