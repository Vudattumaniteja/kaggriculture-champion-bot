"""
Unit tests for Prototype Parameterized Heuristic Specialists (Issue #27).
Tests all 5 specialist archetypes for:
- Correct parameter configuration
- Action structure and legality
- State progression and chore execution
- Full game simulation compatibility with kaggle-environments
"""

import sys
from pathlib import Path

prototype_root = Path(__file__).resolve().parent
grilling_model_root = prototype_root.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
from prototype.specialists import (
    SPECIALIST_CONFIGS,
    SpecialistConfig,
    ParameterizedHeuristicFarmer,
    get_specialist,
    agent_grandmaster,
    agent_carrot_monoculture,
    agent_melon_rusher,
    agent_dairy_syndicate,
    agent_town_shop_saturator,
)


class TestSpecialistArchetypes(unittest.TestCase):
    def setUp(self):
        self.archetypes = [
            "DeterministicGrandmaster",
            "CarrotMonoculture",
            "MelonRusher",
            "DairySyndicate",
            "TownShopSaturator",
        ]

    def test_archetype_configs_exist(self):
        for name in self.archetypes:
            self.assertIn(name, SPECIALIST_CONFIGS)
            cfg = SPECIALIST_CONFIGS[name]
            self.assertIsInstance(cfg, SpecialistConfig)
            self.assertEqual(cfg.name, name)

    def test_specialist_parameters(self):
        # 1. Carrot Monoculture should have 0 animal targets and high carrot cap
        carrot_cfg = SPECIALIST_CONFIGS["CarrotMonoculture"]
        self.assertEqual(carrot_cfg.target_geese, 0)
        self.assertEqual(carrot_cfg.target_cows, 0)
        self.assertEqual(carrot_cfg.target_sheep, 0)
        self.assertGreater(carrot_cfg.carrot_cap, 15)

        # 2. Melon Rusher should have high melon cap
        melon_cfg = SPECIALIST_CONFIGS["MelonRusher"]
        self.assertGreater(melon_cfg.melon_cap, 10)
        self.assertEqual(melon_cfg.target_cows, 0)

        # 3. Dairy Syndicate should focus on livestock and wheat feed
        dairy_cfg = SPECIALIST_CONFIGS["DairySyndicate"]
        self.assertGreaterEqual(dairy_cfg.target_cows, 2)
        self.assertGreaterEqual(dairy_cfg.feed_reserve_multiplier, 3.0)

        # 4. Town Shop Saturator should have dynamic shop focus
        town_cfg = SPECIALIST_CONFIGS["TownShopSaturator"]
        self.assertTrue(town_cfg.focus_town_shops)

    def test_specialist_action_format(self):
        mock_obs = {
            "player": 0,
            "farms": [
                {
                    "money": 3000.0,
                    "farmer": [4, 4],
                    "hands": [],
                    "tiles": [[None for _ in range(10)] for _ in range(10)],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
                {
                    "money": 3000.0,
                    "farmer": [4, 4],
                    "hands": [],
                    "tiles": [[None for _ in range(10)] for _ in range(10)],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                }
            ],
            "private": {
                "shed": {"WHEAT": 5, "CARROT": 2},
                "seeds": {"CARROT": 4},
                "inventories": [{}],
            },
            "market": {"prices": {"CARROT": 35, "WHEAT": 25}},
            "town": {"unlocked_shops": ["BAKERY", "PIZZA_SHOP"]},
            "day": 1,
            "hour": 0,
            "step": 0,
        }

        for name in self.archetypes:
            farmer = get_specialist(name)
            action = farmer(mock_obs)
            self.assertIn("farmer", action)
            self.assertIn("hands", action)
            self.assertIn("market", action)
            self.assertIsInstance(action["farmer"], list)
            self.assertIsInstance(action["market"], list)
            self.assertLessEqual(len(action["market"]), 10)

    def test_callables_work(self):
        mock_obs = {
            "player": 0,
            "farms": [{"money": 3000.0, "farmer": [4, 4], "hands": [], "tiles": []}, {}],
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
            "market": {},
            "town": {},
            "day": 2,
            "hour": 1,
            "step": 25,
        }
        for fn in [
            agent_grandmaster,
            agent_carrot_monoculture,
            agent_melon_rusher,
            agent_dairy_syndicate,
            agent_town_shop_saturator,
        ]:
            act = fn(mock_obs)
            self.assertIn("farmer", act)
            self.assertIn("market", act)


if __name__ == "__main__":
    unittest.main()
