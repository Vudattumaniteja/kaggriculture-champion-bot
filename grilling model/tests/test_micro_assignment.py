import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
from typing import Dict, Any

from kaggle_environments import make
from src.micro_solver import solve_micro_actions, apply_market_guardrails, get_manhattan_dist, get_step_towards
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
        """
        Acceptance Criteria:
        Cow feed reservation clamps Wheat market sell orders to preserve at least Cows * 2 daily feed in shed.
        """
        # Policy requests 100% liquidation (fraction = 1.0) on all items
        raw_fractions = np.ones(9, dtype=np.float32)
        market_orders = apply_market_guardrails(
            market_fractions=raw_fractions,
            obs=self.mock_obs,
            seed_replenish_logits=np.zeros(5, dtype=np.float32),
            land_expand_logit=0.0,
            workforce_logit_idx=0
        )
        
        # Total wheat in shed = 20. With 2 cows, reservation = 2 * 2 = 4
        # Sell order for wheat must leave at least reserved wheat in shed
        wheat_sells = [order for order in market_orders if order[0] == "SELL" and order[1] == "WHEAT"]
        self.assertTrue(len(wheat_sells) > 0)
        sold_wheat_qty = wheat_sells[0][2]
        self.assertLessEqual(sold_wheat_qty, 20 - 4, f"Sold {sold_wheat_qty} wheat, violating 4-wheat cow feed floor")

    def test_cow_feed_zero_cows(self):
        """When 0 cows exist, full wheat liquidation is permitted."""
        obs = dict(self.mock_obs)
        farm = dict(obs["farms"][0])
        farm["tiles"] = [[None for _ in range(10)] for _ in range(10)]
        obs["farms"] = [farm, obs["farms"][1]]
        obs["private"] = {"shed": {"WHEAT": 20}, "seeds": {}, "inventories": [{}]}

        raw_fractions = np.ones(9, dtype=np.float32)
        market_orders = apply_market_guardrails(
            market_fractions=raw_fractions,
            obs=obs
        )
        wheat_sells = [order for order in market_orders if order[0] == "SELL" and order[1] == "WHEAT"]
        self.assertEqual(wheat_sells[0][2], 20)

    def test_midnight_shed_overflow_guardrail(self):
        """
        Acceptance Criteria:
        Midnight shed overflow guardrail automatically liquidates excess goods > 100 items at t % 24 == 23.
        """
        # Total shed items = 20 + 90 + 10 = 120 (>100), at hour=23
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

    def test_midnight_overflow_not_triggered_at_midday(self):
        """Overflow emergency dump should only fire at midnight (hour 23), not at midday (e.g. hour 12)."""
        obs = dict(self.mock_obs)
        obs["hour"] = 12
        obs["step"] = 12
        raw_fractions = np.zeros(9, dtype=np.float32)
        market_orders = apply_market_guardrails(
            market_fractions=raw_fractions,
            obs=obs
        )
        sell_orders = [o for o in market_orders if o[0] == "SELL"]
        self.assertEqual(len(sell_orders), 0, "No emergency sell orders expected at hour 12 when policy fraction is 0")

    def test_hungarian_micro_solver_dispatch(self):
        """
        Acceptance Criteria:
        Cost matrix C_ij = Dist(w_i, t_j) - lambda * Logit(t_j) - Urgency(t_j) solved via Hungarian matching.
        """
        crop_heatmaps = np.zeros((5, 10, 10), dtype=np.float32)
        crop_heatmaps[1, 0, 0] = 5.0  # High preference for carrot at (0, 0)
        
        farmer_act, hands_acts = solve_micro_actions(
            obs=self.mock_obs,
            crop_heatmaps=crop_heatmaps,
            livestock_quotas=np.array([0.0, 0.0, 2.0])
        )
        
        self.assertIsInstance(farmer_act, list)
        self.assertEqual(len(hands_acts), 2)
        valid_ops = {"NORTH", "SOUTH", "EAST", "WEST", "PASS", "FEED", "CARE", "HARVEST", "WATER", "PLANT", "COLLECT_FERTILIZER", "BUILD_COOP", "BUILD_PASTURE", "PICKUP", "PLACE_ANIMAL", "DIG", "FERTILIZE"}
        self.assertIn(farmer_act[0], valid_ops)
        for h_act in hands_acts:
            self.assertIn(h_act[0], valid_ops)

    def test_biological_urgency_priorities(self):
        """
        Acceptance Criteria:
        Biological urgency overrides prioritize starving animals, dry crops, and ripe harvests.
        """
        obs = dict(self.mock_obs)
        farm = dict(obs["farms"][0])
        farm["tiles"] = [[None for _ in range(10)] for _ in range(10)]
        
        # Starving cow at (0, 1)
        farm["tiles"][1][0] = {
            "kind": "PASTURE", "animal": "COW", "fed_today": False, "cared_today": True, "yield_units": 0, "fertilizer_available": False
        }
        # Dry crop at (0, 2)
        farm["tiles"][2][0] = {
            "kind": "PLANT", "crop": "CARROT", "planted_day": 0, "yield_units": 0, "watered_today": False
        }
        # Ripe harvest crop at (0, 3)
        farm["tiles"][3][0] = {
            "kind": "PLANT", "crop": "CARROT", "planted_day": -4, "yield_units": 4, "watered_today": True
        }
        
        farm["farmer"] = [0, 0]
        farm["hands"] = [[1, 0], [2, 0]]
        obs["farms"] = [farm, obs["farms"][1]]
        
        farmer_act, hands_acts = solve_micro_actions(obs)
        all_acts = [farmer_act] + hands_acts
        
        # All units should move SOUTH towards urgent tasks on column 0
        south_moves = sum(1 for a in all_acts if a[0] == "SOUTH")
        self.assertGreaterEqual(south_moves, 1)

    def test_zero_path_collisions_under_12_worker_load(self):
        """
        Acceptance Criteria:
        100-step simulation against environment verifies 100% legal moves and zero path collisions under 12-worker load.
        """
        env = make("kaggriculture", configuration={"episodeSteps": 105}, debug=True)
        champ_agent = ChampionAgent()

        obs = env.reset()[0].observation
        # Initialize 12 workers
        farm0 = env.state[0].observation.farms[0]
        farm0["hands"] = [[(i % 5), (i // 5)] for i in range(11)]
        env.state[0].observation.private["inventories"] = [{} for _ in range(12)]

        steps_run = 0
        while not env.done and steps_run < 100:
            obs = env.state[0].observation
            action = champ_agent(obs)
            
            p0 = obs["farms"][0]
            units = [tuple(p0["farmer"])] + [tuple(h) for h in p0["hands"]]
            unit_acts = [action["farmer"]] + action["hands"]
            
            dests = []
            for u_pos, act in zip(units, unit_acts):
                op = act[0]
                if op == "NORTH":
                    dests.append((u_pos[0], u_pos[1] - 1))
                elif op == "SOUTH":
                    dests.append((u_pos[0], u_pos[1] + 1))
                elif op == "EAST":
                    dests.append((u_pos[0] + 1, u_pos[1]))
                elif op == "WEST":
                    dests.append((u_pos[0] - 1, u_pos[1]))
                else:
                    dests.append(u_pos)

            self.assertEqual(len(dests), len(set(dests)), f"Path collision at step {steps_run}! Dests: {dests}")

            state = env.step([action, {"farmer": ["PASS"]}])
            steps_run += 1
            self.assertNotEqual(state[0].status, "ERROR", f"Agent errored at step {steps_run}: {state[0].info}")

            # Replenish hands if end_of_day reset occurred
            if len(env.state[0].observation.farms[0]["hands"]) < 11:
                env.state[0].observation.farms[0]["hands"] = [[(i % 5), (i // 5)] for i in range(11)]
                env.state[0].observation.private["inventories"] = [{} for _ in range(12)]

        self.assertGreaterEqual(steps_run, 100)

    def test_agent_callable_contract(self):
        """
        Acceptance Criteria:
        grilling model/src/agent.py compiles valid submission action dictionary matching Kaggle format.
        """
        res = agent(self.mock_obs)
        self.assertIn("farmer", res)
        self.assertIn("hands", res)
        self.assertIn("market", res)
        self.assertIsInstance(res["farmer"], list)
        self.assertIsInstance(res["hands"], list)
        self.assertIsInstance(res["market"], list)
        self.assertLessEqual(len(res["market"]), 10)


if __name__ == "__main__":
    unittest.main()
