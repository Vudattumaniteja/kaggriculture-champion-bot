import sys
import os
import random
from pathlib import Path

# Add grilling model root to sys.path
grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
from typing import Dict, Any, List

from src.encoder import (
    encode_observation,
    symlog,
    get_quadrant,
    SPATIAL_CHANNELS,
    SCALAR_DIM,
    CROPS,
    PRODUCTS,
    ANIMALS,
    BASE_PRICES,
    BASE_INVENTORY,
    SHOPS_CATALOG,
    SHED_TILES
)


class TestObservationEncoder(unittest.TestCase):
    def setUp(self):
        # Construct synthetic observation
        self.mock_obs: Dict[str, Any] = {
            "player": 0,
            "step": 120,
            "day": 5,
            "hour": 0,
            "farms": [
                {
                    "money": 3500.0,
                    "hires_today": 1,
                    "unlocked_quadrants": ["NW", "NE"],
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
                    "hands": [[4, 5]],
                    "tiles": [
                        [None for _ in range(10)] for _ in range(10)
                    ]
                }
            ],
            "private": {
                "shed": {"WHEAT": 10, "CARROT": 5, "MILK": 2},
                "seeds": {"WHEAT": 4, "CARROT": 6},
                "inventories": [{}, {}, {}]
            },
            "market": {
                "prices": {
                    "WHEAT": 28, "CARROT": 36, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
                    "EGG": 55, "MILK": 170, "WOOL": 200, "FERTILIZER": 100
                },
                "inventory": {
                    "WHEAT": 100, "CARROT": 80, "TOMATO": 50, "STRAWBERRY": 40, "MELON": 20,
                    "EGG": 60, "MILK": 40, "WOOL": 30, "FERTILIZER": 50
                }
            },
            "town": {
                "unlocked_shops": ["BAKERY", "PIZZA_SHOP"]
            }
        }

        # Populate some tiles on player farm
        # Planted carrot at row 1, col 1
        self.mock_obs["farms"][0]["tiles"][1][1] = {
            "kind": "PLANT",
            "crop": "CARROT",
            "planted_day": 3,
            "watered_today": True,
            "fertilized_until_day": 6,
            "yield_units": 2
        }
        # Coop with Goose at row 2, col 2
        self.mock_obs["farms"][0]["tiles"][2][2] = {
            "kind": "COOP",
            "animal": "GOOSE",
            "fed_today": False,
            "cared_today": True,
            "yield_units": 1,
            "fertilizer_available": True
        }
        # Pasture with Cow at row 3, col 3
        self.mock_obs["farms"][0]["tiles"][3][3] = {
            "kind": "PASTURE",
            "animal": "COW",
            "fed_today": True,
            "cared_today": True,
            "yield_units": 0,
            "fertilizer_available": False
        }
        # Locked quadrants in SE (rows 5-9, cols 5-9)
        for r in range(5, 10):
            for c in range(5, 10):
                self.mock_obs["farms"][0]["tiles"][r][c] = "LOCKED"

    def test_output_shapes_and_types(self):
        spatial, scalar = encode_observation(self.mock_obs)
        self.assertEqual(spatial.shape, (SPATIAL_CHANNELS, 10, 10))
        self.assertEqual(scalar.shape, (SCALAR_DIM,))
        self.assertEqual(spatial.dtype, np.float32)
        self.assertEqual(scalar.dtype, np.float32)

    def test_no_nans_or_infs(self):
        spatial, scalar = encode_observation(self.mock_obs)
        self.assertFalse(np.isnan(spatial).any(), "Spatial tensor contains NaNs")
        self.assertFalse(np.isinf(spatial).any(), "Spatial tensor contains Infs")
        self.assertFalse(np.isnan(scalar).any(), "Scalar vector contains NaNs")
        self.assertFalse(np.isinf(scalar).any(), "Scalar vector contains Infs")

    def test_geometric_channels(self):
        spatial, _ = encode_observation(self.mock_obs)
        # Check Quadrant Cost Map (Channel 19)
        # NW = 0.0, NE = 0.25, SW = 0.50, SE = 1.00
        self.assertAlmostEqual(spatial[19, 0, 0], 0.0)
        self.assertAlmostEqual(spatial[19, 0, 9], 0.25)
        self.assertAlmostEqual(spatial[19, 9, 0], 0.50)
        self.assertAlmostEqual(spatial[19, 9, 9], 1.00)

        # Check Shed Footprint (Channel 16)
        self.assertEqual(spatial[16, 4, 4], 1.0)
        self.assertEqual(spatial[16, 4, 5], 1.0)
        self.assertEqual(spatial[16, 5, 4], 1.0)
        self.assertEqual(spatial[16, 5, 5], 1.0)
        self.assertEqual(spatial[16, 0, 0], 0.0)

        # Check Delivery Zone (Channel 17)
        self.assertEqual(spatial[17, 0, 0], 0.5)
        self.assertEqual(spatial[17, 9, 9], 0.5)
        self.assertEqual(spatial[17, 4, 4], 1.0)

        # Check Shed Distance Transform (Channel 20)
        self.assertEqual(spatial[20, 4, 4], 0.0)
        self.assertGreater(spatial[20, 0, 0], 0.0)
        self.assertLessEqual(spatial[20].max(), 1.0)
        self.assertGreaterEqual(spatial[20].min(), 0.0)

        # Check Shop Distance Transform (Channel 21) & Shop Delta Row (Channel 22)
        self.assertLessEqual(spatial[21].max(), 1.0)
        self.assertGreaterEqual(spatial[21].min(), 0.0)
        self.assertLessEqual(spatial[22].max(), 1.0)
        self.assertGreaterEqual(spatial[22].min(), -1.0)

        # Check Unlock Status Map (Channel 23)
        self.assertEqual(spatial[23, 0, 0], 1.0)  # NW unlocked
        self.assertEqual(spatial[23, 0, 9], 1.0)  # NE unlocked
        self.assertEqual(spatial[23, 9, 9], 0.0)  # SE locked

    def test_crop_channels(self):
        spatial, _ = encode_observation(self.mock_obs)
        # Carrot is index 1 in CROPS
        self.assertEqual(spatial[1, 1, 1], 1.0)  # One-hot CARROT channel
        self.assertEqual(spatial[0, 1, 1], 0.0)  # Not WHEAT
        self.assertEqual(spatial[2, 1, 1], 0.0)  # Not TOMATO
        self.assertEqual(spatial[3, 1, 1], 0.0)  # Not STRAWBERRY
        self.assertEqual(spatial[4, 1, 1], 0.0)  # Not MELON
        self.assertGreater(spatial[5, 1, 1], 0.0)  # Growth ratio > 0
        self.assertEqual(spatial[6, 1, 1], 1.0)  # Watered today
        self.assertEqual(spatial[7, 1, 1], 1.0)  # Fertilized active

    def test_animal_channels(self):
        spatial, _ = encode_observation(self.mock_obs)
        # Coop is channel 8, Goose is channel 11
        self.assertEqual(spatial[8, 2, 2], 1.0)   # Coop
        self.assertEqual(spatial[11, 2, 2], 1.0)  # Goose
        self.assertEqual(spatial[14, 2, 2], 1.0)  # Hunger = 1.0 because fed_today is False
        self.assertAlmostEqual(spatial[15, 2, 2], 0.25)  # Product ready = 1/4

        # Pasture is channel 9, Cow is channel 12
        self.assertEqual(spatial[9, 3, 3], 1.0)   # Pasture
        self.assertEqual(spatial[12, 3, 3], 1.0)  # Cow (COW is index 1 in ANIMALS -> 11 + 1 = 12)
        self.assertEqual(spatial[14, 3, 3], 0.0)  # Hunger = 0.0 because fed_today is True

    def test_worker_density_channel(self):
        spatial, _ = encode_observation(self.mock_obs)
        self.assertGreater(spatial[18].sum(), 0.0)
        # Farmer at [4, 4], hands at [4, 5] and [3, 2]
        self.assertGreaterEqual(spatial[18, 4, 4], 1.0)
        self.assertGreaterEqual(spatial[18, 5, 4], 1.0)
        self.assertGreaterEqual(spatial[18, 2, 3], 1.0)

    def test_scalar_economic_features(self):
        _, scalar = encode_observation(self.mock_obs)
        # Turn clocks
        self.assertAlmostEqual(scalar[0], 120.0 / 720.0)
        self.assertAlmostEqual(scalar[1], 0.0)
        self.assertAlmostEqual(scalar[2], 5.0 / 30.0)

        # Cash symlog
        self.assertAlmostEqual(scalar[3], symlog(3500.0) / 15.0)
        self.assertAlmostEqual(scalar[4], symlog(2800.0) / 15.0)

        # Labor & Quadrants
        self.assertAlmostEqual(scalar[6], (3 * 20.0) / 260.0)  # 2 hands + 1 farmer = 3
        self.assertAlmostEqual(scalar[7], 1.0 / 5.0)  # hires today
        self.assertAlmostEqual(scalar[8], 2.0 / 4.0)  # 2 unlocked quads
        self.assertEqual(scalar[9], 1.0)  # NE unlocked
        self.assertEqual(scalar[10], 0.0)  # SW not unlocked

        # Spot prices, baseline ratios, crash indicators
        self.assertGreater(scalar[11], 0.0)
        self.assertGreater(scalar[20], 0.0)
        self.assertGreater(scalar[29], 0.0)

        # Town shop demands
        self.assertEqual(scalar[38], 1.0)  # BAKERY unlocked (index 0)
        self.assertEqual(scalar[39], 1.0)  # PIZZA_SHOP unlocked (index 1)
        self.assertEqual(scalar[40], 0.0)  # BRUNCH_SPOT locked (index 2)

        # Shed inventory and saturation
        self.assertGreater(scalar[46], 0.0)  # WHEAT in shed
        self.assertGreater(scalar[55], 0.0)  # Shed saturation

        # Opponent public telemetry
        self.assertAlmostEqual(scalar[69], 2800.0 / 10000.0)
        self.assertAlmostEqual(scalar[70], 1.0 / 4.0)
        self.assertAlmostEqual(scalar[71], 2.0 / 13.0)  # 1 opp hand + 1 opp farmer = 2

    def test_edge_case_bankrupt_and_negative_cash(self):
        bankrupt_obs = {
            "player": 0,
            "farms": [
                {"money": -500.0, "unlocked_quadrants": ["NW"], "farmer": [4, 4], "hands": []},
                {"money": 0.0, "unlocked_quadrants": ["NW"], "farmer": [4, 4], "hands": []}
            ],
            "private": {"shed": {}, "seeds": {}},
            "market": {"prices": {}, "inventory": {}},
            "town": {"unlocked_shops": []}
        }
        spatial, scalar = encode_observation(bankrupt_obs)
        self.assertEqual(spatial.shape, (24, 10, 10))
        self.assertEqual(scalar.shape, (72,))
        self.assertFalse(np.isnan(spatial).any())
        self.assertFalse(np.isnan(scalar).any())
        self.assertFalse(np.isinf(spatial).any())
        self.assertFalse(np.isinf(scalar).any())

    def test_edge_case_fully_unlocked_and_max_capacity(self):
        full_obs = {
            "player": 1,
            "step": 719,
            "day": 29,
            "hour": 23,
            "farms": [
                {"money": 1000000.0, "unlocked_quadrants": ["NW", "NE", "SW", "SE"], "farmer": [4, 4], "hands": [[i, i] for i in range(12)]},
                {"money": 500000.0, "unlocked_quadrants": ["NW", "NE", "SW", "SE"], "farmer": [5, 5], "hands": [[i, 9 - i] for i in range(12)]}
            ],
            "private": {
                "shed": {prod: 500 for prod in PRODUCTS},
                "seeds": {crop: 200 for crop in CROPS}
            },
            "market": {
                "prices": {prod: 10000.0 for prod in PRODUCTS},
                "inventory": {prod: 50000.0 for prod in PRODUCTS}
            },
            "town": {
                "unlocked_shops": SHOPS_CATALOG
            }
        }
        spatial, scalar = encode_observation(full_obs)
        self.assertEqual(spatial.shape, (24, 10, 10))
        self.assertEqual(scalar.shape, (72,))
        self.assertFalse(np.isnan(spatial).any())
        self.assertFalse(np.isnan(scalar).any())
        self.assertFalse(np.isinf(spatial).any())
        self.assertFalse(np.isinf(scalar).any())
        self.assertTrue((spatial >= -1.0).all() and (spatial <= 12.0).all())
        self.assertTrue((scalar >= -10.0).all() and (scalar <= 10.0).all())

    def test_edge_case_empty_and_corrupted_obs(self):
        empty_obs: Dict[str, Any] = {"player": 0}
        spatial, scalar = encode_observation(empty_obs)
        self.assertEqual(spatial.shape, (24, 10, 10))
        self.assertEqual(scalar.shape, (72,))
        self.assertFalse(np.isnan(spatial).any())
        self.assertFalse(np.isnan(scalar).any())

        corrupted_obs = {
            "player": "invalid",
            "farms": "not_a_list",
            "private": None,
            "market": {"prices": "invalid", "inventory": None},
            "town": None,
            "step": None,
            "day": "five",
            "hour": None
        }
        spatial, scalar = encode_observation(corrupted_obs)
        self.assertEqual(spatial.shape, (24, 10, 10))
        self.assertEqual(scalar.shape, (72,))
        self.assertFalse(np.isnan(spatial).any())
        self.assertFalse(np.isnan(scalar).any())

    def test_100_sample_game_steps_simulation(self):
        """Validates tensor shapes, channel bounds, and normalization across 100 sample game steps."""
        random.seed(42)
        np.random.seed(42)

        for step in range(100):
            day = step // 24
            hour = step % 24
            unlocked = ["NW"]
            if day >= 2:
                unlocked.append("NE")
            if day >= 5:
                unlocked.append("SW")
            if day >= 8:
                unlocked.append("SE")

            tiles: List[List[Any]] = [[None for _ in range(10)] for _ in range(10)]
            for r in range(10):
                for c in range(10):
                    q = get_quadrant(r, c)
                    if q not in unlocked:
                        tiles[r][c] = "LOCKED"
                    elif (r, c) in SHED_TILES:
                        tiles[r][c] = None
                    elif (r + c + step) % 5 == 0:
                        crop_name = random.choice(CROPS)
                        tiles[r][c] = {
                            "kind": "PLANT",
                            "crop": crop_name,
                            "planted_day": max(0, day - random.randint(0, 4)),
                            "watered_today": bool(random.getrandbits(1)),
                            "fertilized_until_day": day + random.randint(0, 2),
                            "yield_units": random.randint(0, 4)
                        }
                    elif (r + c + step) % 7 == 0:
                        an_name = random.choice(ANIMALS)
                        pen_kind = "COOP" if an_name == "GOOSE" else "PASTURE"
                        tiles[r][c] = {
                            "kind": pen_kind,
                            "animal": an_name,
                            "fed_today": bool(random.getrandbits(1)),
                            "cared_today": True,
                            "yield_units": random.randint(0, 3)
                        }

            num_hands = min(step // 10, 10)
            hands_pos = [[random.randint(0, 9), random.randint(0, 9)] for _ in range(num_hands)]

            sim_obs: Dict[str, Any] = {
                "player": step % 2,
                "step": step,
                "day": day,
                "hour": hour,
                "farms": [
                    {
                        "money": 3000.0 + step * 25.0 - num_hands * 20.0,
                        "hires_today": 1 if hour == 0 else 0,
                        "unlocked_quadrants": unlocked,
                        "farmer": [random.randint(0, 9), random.randint(0, 9)],
                        "hands": hands_pos,
                        "tiles": tiles
                    },
                    {
                        "money": 3000.0 + step * 20.0,
                        "hires_today": 0,
                        "unlocked_quadrants": ["NW"],
                        "farmer": [4, 4],
                        "hands": [],
                        "tiles": [[None for _ in range(10)] for _ in range(10)]
                    }
                ],
                "private": {
                    "shed": {p: random.randint(0, 20) for p in PRODUCTS},
                    "seeds": {c: random.randint(0, 10) for c in CROPS}
                },
                "market": {
                    "prices": {p: BASE_PRICES[p] * (0.8 + 0.4 * random.random()) for p in PRODUCTS},
                    "inventory": {p: BASE_INVENTORY[p] * (0.5 + random.random()) for p in PRODUCTS}
                },
                "town": {
                    "unlocked_shops": random.sample(SHOPS_CATALOG, k=min(len(SHOPS_CATALOG), 1 + step // 15))
                }
            }

            spatial, scalar = encode_observation(sim_obs)

            # Assert tensor dimensions and types
            self.assertEqual(spatial.shape, (24, 10, 10))
            self.assertEqual(scalar.shape, (72,))
            self.assertEqual(spatial.dtype, np.float32)
            self.assertEqual(scalar.dtype, np.float32)

            # Assert no NaNs or Infs
            self.assertFalse(np.isnan(spatial).any(), f"NaN in spatial tensor at step {step}")
            self.assertFalse(np.isinf(spatial).any(), f"Inf in spatial tensor at step {step}")
            self.assertFalse(np.isnan(scalar).any(), f"NaN in scalar vector at step {step}")
            self.assertFalse(np.isinf(scalar).any(), f"Inf in scalar vector at step {step}")

            # Assert bounds
            self.assertTrue((spatial >= -1.0).all() and (spatial <= 20.0).all(), f"Spatial bounds violated at step {step}")
            self.assertTrue((scalar >= -10.0).all() and (scalar <= 10.0).all(), f"Scalar bounds violated at step {step}")


if __name__ == "__main__":
    unittest.main()
