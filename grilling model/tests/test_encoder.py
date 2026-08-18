import sys
import os
from pathlib import Path

# Add grilling model root to sys.path
grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import numpy as np
from typing import Dict, Any

from src.encoder import encode_observation, SPATIAL_CHANNELS, SCALAR_DIM, CROPS, PRODUCTS, ANIMALS, SHOPS_CATALOG


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
        # Locked quadrants in SE (rows 5-9, cols 5-9)
        for r in range(5, 10):
            for c in range(5, 10):
                self.mock_obs["farms"][0]["tiles"][r][c] = "LOCKED"

    def test_output_shapes_and_types(self):
        spatial, scalar = encode_observation(self.mock_obs)
        self.assertEqual(spatial.shape, (24, 10, 10))
        self.assertEqual(scalar.shape, (72,))
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
        self.assertEqual(spatial[16, 5, 5], 1.0)
        self.assertEqual(spatial[16, 0, 0], 0.0)

        # Check Unlock Status Map (Channel 23)
        self.assertEqual(spatial[23, 0, 0], 1.0)  # NW unlocked
        self.assertEqual(spatial[23, 0, 9], 1.0)  # NE unlocked
        self.assertEqual(spatial[23, 9, 9], 0.0)  # SE locked

    def test_crop_channels(self):
        spatial, _ = encode_observation(self.mock_obs)
        # Carrot is index 1 in CROPS
        self.assertEqual(spatial[1, 1, 1], 1.0)  # One-hot CARROT channel
        self.assertEqual(spatial[0, 1, 1], 0.0)  # Not WHEAT
        self.assertEqual(spatial[6, 1, 1], 1.0)  # Watered today
        self.assertEqual(spatial[7, 1, 1], 1.0)  # Fertilized active

    def test_animal_channels(self):
        spatial, _ = encode_observation(self.mock_obs)
        # Coop is channel 8, Goose is channel 11
        self.assertEqual(spatial[8, 2, 2], 1.0)   # Coop
        self.assertEqual(spatial[11, 2, 2], 1.0)  # Goose
        self.assertEqual(spatial[14, 2, 2], 1.0)  # Hunger = 1.0 because fed_today is False

    def test_worker_density_channel(self):
        spatial, _ = encode_observation(self.mock_obs)
        self.assertGreater(spatial[18].sum(), 0.0)

    def test_edge_case_empty_obs(self):
        empty_obs: Dict[str, Any] = {"player": 0}
        spatial, scalar = encode_observation(empty_obs)
        self.assertEqual(spatial.shape, (24, 10, 10))
        self.assertEqual(scalar.shape, (72,))
        self.assertFalse(np.isnan(spatial).any())
        self.assertFalse(np.isnan(scalar).any())


if __name__ == "__main__":
    unittest.main()
