import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
from typing import Dict, Any

from src.micro_solver import solve_micro_actions, apply_market_guardrails
from src.agent import ChampionAgent, agent


class TestMicroAssignment(unittest.TestCase):
    def setUp(self):
        self.mock_obs: Dict[str, Any] = {
            "player": 0,
            "step": 23,
            "day": 0,
            "hour": 23,
            "farms": [
                {
                    "money": 2500.0,
                    "hires_today": 0,
                    "unlocked_quadrants": ["NW"],
                    "farmer": [4, 4],
                    "hands": [[4, 5], [3, 2]],
                    "tiles": [
                        [None for _ in range(10)] for _ in range(10)
                    ]
                },
                {
                    "money": 2800.0,
                    "hires_today": 0,
                    "unlocked_quadrants": ["NW"],
                    "farmer": [4, 4],
                    "hands": [],
                    "tiles": [
                        [None for _ in range(10)] for _ in range(10)
                    ]
                }
            ],
            "private": {
                "shed": {"WHEAT": 20, "CARROT": 90, "MILK": 10},  # Total shed = 120 items (>100 overflow)
                "seeds": {"WHEAT": 4, "CARROT": 6},
                "inventories": [{}, {}, {}]
            },
            "market": {
                "prices": {
                    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
                    "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100
                },
                "inventory": {
                    "WHEAT": 100, "CARROT": 100, "TOMATO": 100, "STRAWBERRY": 100, "MELON": 100,
                    "EGG": 100, "MILK": 100, "WOOL": 100, "FERTILIZER": 100
                }
            },
            "town": {
                "unlocked_shops": ["BAKERY"]
            }
        }

        # Add 2 cows on tiles (1, 1) and (1, 2)
        self.mock_obs["farms"][0]["tiles"][1][1] = {
            "kind": "PASTURE",
            "animal": "COW",
            "fed_today": False,
            "cared_today": True,
            "yield_units": 0,
            "fertilizer_available": False
        }
        self.mock_obs["farms"][0]["tiles"][1][2] = {
            "kind": "PASTURE",
            "animal": "COW",
            "fed_today": False,
            "cared_today": True,
            "yield_units": 0,
            "fertilizer_available": False
        }

    def test_cow_feed_reservation_guardrail(self):
        # Policy requests 100% liquidation (fraction = 1.0) on all items
        raw_fractions = np.ones(9, dtype=np.float32)
        market_orders = apply_market_guardrails(
            market_fractions=raw_fractions,
            obs=self.mock_obs,
            seed_replenish_logits=np.zeros(5, dtype=np.float32),
            land_expand_logit=0.0,
            workforce_logit_idx=0
        )
        
        # Total wheat in shed = 20. With 2 cows, reservation = 2 * 2 = 4 (or safe buffer 6)
        # Sell order for wheat must leave at least reserved wheat in shed
        wheat_sells = [order for order in market_orders if order[0] == "SELL" and order[1] == "WHEAT"]
        self.assertTrue(len(wheat_sells) > 0)
        sold_wheat_qty = wheat_sells[0][2]
        self.assertLessEqual(sold_wheat_qty, 20 - 4)

    def test_midnight_shed_overflow_guardrail(self):
        # Total shed items = 20 + 90 + 10 = 120 (>100), at hour=23
        # Guardrail must emit sell orders to bring shed under 100
        raw_fractions = np.zeros(9, dtype=np.float32)  # Policy wants 0 sales
        market_orders = apply_market_guardrails(
            market_fractions=raw_fractions,
            obs=self.mock_obs,
            seed_replenish_logits=np.zeros(5, dtype=np.float32),
            land_expand_logit=0.0,
            workforce_logit_idx=0
        )
        
        total_sold = sum(order[2] for order in market_orders if order[0] == "SELL")
        self.assertGreaterEqual(total_sold, 20, "Should liquidate at least 20 items to prevent overflow discard")

    def test_hungarian_micro_solver_dispatch(self):
        crop_heatmaps = np.zeros((5, 10, 10), dtype=np.float32)
        crop_heatmaps[1, 0, 0] = 5.0  # High preference for carrot at (0, 0)
        
        farmer_act, hands_acts = solve_micro_actions(
            obs=self.mock_obs,
            crop_heatmaps=crop_heatmaps,
            livestock_quotas=np.array([0.0, 0.0, 2.0])
        )
        
        self.assertIsInstance(farmer_act, list)
        self.assertEqual(len(hands_acts), 2)
        # Verify valid action strings/ops
        valid_ops = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "FEED", "CARE", "HARVEST", "WATER", "PLANT", "COLLECT_FERTILIZER", "BUILD_COOP", "BUILD_PASTURE", "PICKUP", "PLACE_ANIMAL", "DIG", "FERTILIZE"}
        self.assertIn(farmer_act[0], valid_ops)
        for h_act in hands_acts:
            self.assertIn(h_act[0], valid_ops)

    def test_agent_callable_contract(self):
        champ_agent = ChampionAgent()
        res = champ_agent(self.mock_obs)
        self.assertIn("farmer", res)
        self.assertIn("hands", res)
        self.assertIn("market", res)
        self.assertIsInstance(res["farmer"], list)
        self.assertIsInstance(res["hands"], list)
        self.assertIsInstance(res["market"], list)
        self.assertLessEqual(len(res["market"]), 10)


if __name__ == "__main__":
    unittest.main()
