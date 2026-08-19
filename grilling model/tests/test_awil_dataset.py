import sys
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.awil_dataset import (
    apply_d4_augmentation,
    transform_coordinates_d4,
    compute_awil_sample_weights,
    get_phase_bucket,
    parse_replay_transitions,
    AWILDataset,
    Stage1ValueBaseline,
    train_stage1_baseline,
    validate_dataset_integrity,
)


class TestAWILDataset(unittest.TestCase):
    def setUp(self):
        self.spatial = np.zeros((24, 10, 10), dtype=np.float32)
        self.spatial[1, 2, 3] = 1.0  # Tile at row 2, col 3 has Carrot
        self.crop_target = np.zeros((5, 10, 10), dtype=np.float32)
        self.crop_target[1, 2, 3] = 1.0

        self.scalar = np.ones(72, dtype=np.float32) * 0.5
        self.market_fractions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)

    def test_d4_augmentation_rotations_and_reflections(self):
        for transform_idx in range(8):
            aug_spatial, aug_crop_target, aug_scalar, aug_market = apply_d4_augmentation(
                spatial=self.spatial,
                crop_target=self.crop_target,
                scalar=self.scalar,
                market_fractions=self.market_fractions,
                transform_idx=transform_idx
            )
            self.assertEqual(aug_spatial.shape, (24, 10, 10))
            self.assertEqual(aug_crop_target.shape, (5, 10, 10))
            np.testing.assert_array_equal(aug_scalar, self.scalar)
            np.testing.assert_array_equal(aug_market, self.market_fractions)
            aug_spatial_carrot_pos = np.argwhere(aug_spatial[1] == 1.0)
            aug_target_carrot_pos = np.argwhere(aug_crop_target[1] == 1.0)
            np.testing.assert_array_equal(aug_spatial_carrot_pos, aug_target_carrot_pos)

    def test_d4_coordinate_transformation_mapping(self):
        r, c = 2, 3
        r_new, c_new = transform_coordinates_d4(r, c, transform_idx=3)
        self.assertEqual((r_new, c_new), (3, 9 - 2))
        self.assertEqual((r_new, c_new), (3, 7))

        for idx in range(8):
            grid = np.zeros((24, 10, 10), dtype=np.float32)
            target = np.zeros((5, 10, 10), dtype=np.float32)
            grid[0, r, c] = 1.0
            target[0, r, c] = 1.0
            g_aug, t_aug, _, _ = apply_d4_augmentation(grid, target, self.scalar, self.market_fractions, transform_idx=idx)
            rn, cn = transform_coordinates_d4(r, c, transform_idx=idx)
            self.assertEqual(g_aug[0, rn, cn], 1.0)
            self.assertEqual(t_aug[0, rn, cn], 1.0)

    def test_phase_bucket_partitioning(self):
        self.assertEqual(get_phase_bucket(step=0), 0)
        self.assertEqual(get_phase_bucket(step=140), 0)
        self.assertEqual(get_phase_bucket(step=200), 1)
        self.assertEqual(get_phase_bucket(step=350), 2)
        self.assertEqual(get_phase_bucket(step=500), 3)
        self.assertEqual(get_phase_bucket(step=700), 4)

    def test_awil_sample_weight_bounds(self):
        steps = np.random.randint(0, 720, size=100)
        returns = np.random.uniform(50000.0, 300000.0, size=100)
        values = np.random.uniform(40000.0, 280000.0, size=100)
        
        weights = compute_awil_sample_weights(steps=steps, returns=returns, baseline_values=values, tau=1.0)
        self.assertEqual(len(weights), 100)
        self.assertTrue(np.all(weights >= 0.2), f'Min weight {weights.min()} < 0.2')
        self.assertTrue(np.all(weights <= 5.0), f'Max weight {weights.max()} > 5.0')

    def test_exponential_land_expansion_ramp(self):
        steps = []
        for s in range(40):
            quads = ['NW', 'NE'] if s >= 20 else ['NW']
            farm0 = {
                'money': 5000.0,
                'tiles': [[None]*10 for _ in range(10)],
                'unlocked_quadrants': quads,
                'hands': [],
            }
            farm1 = {
                'money': 60000.0,
                'tiles': [[None]*10 for _ in range(10)],
                'unlocked_quadrants': ['NW'],
                'hands': [],
            }
            obs0 = {'step': s, 'farms': [farm0, farm1], 'player': 0, 'private': {'shed': {}}}
            obs1 = {'step': s, 'farms': [farm0, farm1], 'player': 1, 'private': {'shed': {}}}
            
            act0 = {'farmer': ['PASS'], 'market': [['BUY_LAND', 'NE']] if s == 20 else []}
            act1 = {'farmer': ['PASS']}
            
            r0 = 70000.0 if s == 39 else 0.0
            r1 = 60000.0 if s == 39 else 0.0
            
            steps.append([
                {'observation': obs0, 'action': act0, 'reward': r0},
                {'observation': obs1, 'action': act1, 'reward': r1},
            ])

        replay_dict = {'steps': steps}
        transitions = parse_replay_transitions(replay_dict, min_cash=50000.0)
        self.assertGreater(len(transitions), 0)

        p0_transitions = [t for t in transitions if t.get('player', 0) == 0]
        for t in p0_transitions:
            step = t['step']
            land_val = t['land_expand']
            if step <= 20:
                dt = 20 - step
                if dt <= 12:
                    expected_ramp = math.exp(-dt / 12.0)
                    self.assertAlmostEqual(land_val, expected_ramp, places=3,
                                           msg=f'Step {step}: dt={dt}, land_val={land_val}, expected={expected_ramp}')
                else:
                    self.assertEqual(land_val, 0.0)
            else:
                self.assertEqual(land_val, 0.0)

    def test_livestock_quotas_extraction(self):
        farm0 = {
            'money': 60000.0,
            'tiles': [[None]*10 for _ in range(10)],
            'unlocked_quadrants': ['NW'],
            'hands': [],
        }
        farm0['tiles'][0][0] = {'kind': 'PASTURE', 'animal': 'COW', 'fed_today': True}
        farm0['tiles'][0][1] = {'kind': 'PASTURE', 'animal': 'COW', 'fed_today': True}
        farm0['tiles'][0][2] = {'kind': 'PASTURE', 'animal': 'SHEEP', 'fed_today': True}
        farm0['tiles'][1][0] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': True}
        farm0['tiles'][1][1] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': True}
        farm0['tiles'][1][2] = {'kind': 'COOP', 'animal': 'GOOSE', 'fed_today': True}

        steps = []
        for s in range(15):
            obs0 = {'step': s, 'farms': [farm0, farm0], 'player': 0, 'private': {'shed': {}}}
            act0 = {'farmer': ['WATER', 0, 0] if s % 2 == 0 else ['PASS']}
            steps.append([
                {'observation': obs0, 'action': act0, 'reward': 80000.0 if s == 14 else 0.0},
                {'observation': obs0, 'action': act0, 'reward': 10000.0 if s == 14 else 0.0},
            ])

        replay_dict = {'steps': steps}
        transitions = parse_replay_transitions(replay_dict, min_cash=50000.0)
        self.assertGreater(len(transitions), 0)
        
        quotas = transitions[0]['livestock_quotas']
        self.assertEqual(quotas.shape, (3,))
        self.assertEqual(quotas[0], 3.0)
        self.assertEqual(quotas[1], 2.0)
        self.assertEqual(quotas[2], 1.0)

    def test_masked_trade_fractions(self):
        farm0 = {
            'money': 60000.0,
            'tiles': [[None]*10 for _ in range(10)],
            'unlocked_quadrants': ['NW'],
            'hands': [],
        }
        shed = {'WHEAT': 50, 'CARROT': 20, 'MILK': 10}
        steps = []
        for s in range(12):
            obs0 = {'step': s, 'farms': [farm0, farm0], 'player': 0, 'private': {'shed': shed}}
            act0 = {'farmer': ['PASS'], 'market': [['SELL', 'WHEAT', 25], ['SELL', 'CARROT', 20]]}
            steps.append([
                {'observation': obs0, 'action': act0, 'reward': 75000.0 if s == 11 else 0.0},
                {'observation': obs0, 'action': act0, 'reward': 20000.0 if s == 11 else 0.0},
            ])

        replay_dict = {'steps': steps}
        transitions = parse_replay_transitions(replay_dict, min_cash=50000.0)
        self.assertGreater(len(transitions), 0)
        
        fractions = transitions[0]['market_fractions']
        masks = transitions[0]['market_mask']
        self.assertEqual(fractions.shape, (9,))
        self.assertEqual(masks.shape, (9,))
        
        self.assertAlmostEqual(fractions[0], 0.5)
        self.assertEqual(masks[0], 1.0)
        
        self.assertAlmostEqual(fractions[1], 1.0)
        self.assertEqual(masks[1], 1.0)
        
        self.assertAlmostEqual(fractions[6], 0.0)
        self.assertEqual(masks[6], 1.0)
        
        self.assertEqual(masks[2], 0.0)

    def test_quality_filter_and_pass_deduplication(self):
        steps = []
        for s in range(25):
            farm0 = {'money': 45000.0, 'tiles': [[None]*10 for _ in range(10)], 'unlocked_quadrants': ['NW'], 'hands': []}
            farm1 = {'money': 80000.0, 'tiles': [[None]*10 for _ in range(10)], 'unlocked_quadrants': ['NW'], 'hands': []}
            obs0 = {'step': s, 'farms': [farm0, farm1], 'player': 0, 'private': {'shed': {}}}
            obs1 = {'step': s, 'farms': [farm0, farm1], 'player': 1, 'private': {'shed': {}}}
            
            act1 = {'farmer': ['PASS'], 'hands': [], 'market': []}
            steps.append([
                {'observation': obs0, 'action': {'farmer': ['WATER', 0, 0]}, 'reward': 45000.0 if s == 24 else 0.0},
                {'observation': obs1, 'action': act1, 'reward': 80000.0 if s == 24 else 0.0},
            ])

        replay_dict = {'steps': steps}
        transitions = parse_replay_transitions(replay_dict, min_cash=50000.0)
        
        p0_transitions = [t for t in transitions if t.get('player') == 0]
        self.assertEqual(len(p0_transitions), 0)
        
        p1_transitions = [t for t in transitions if t.get('player') == 1]
        self.assertEqual(len(p1_transitions), 1)

    def test_stage1_baseline_value_model(self):
        sample_transitions = [
            {
                'spatial': np.random.randn(24, 10, 10).astype(np.float32),
                'scalar': np.random.randn(72).astype(np.float32),
                'crop_heatmaps': np.zeros((5, 10, 10), dtype=np.float32),
                'livestock_quotas': np.zeros(3, dtype=np.float32),
                'workforce': 1,
                'land_expand': 0.0,
                'seed_replenish': np.zeros(5, dtype=np.float32),
                'market_fractions': np.zeros(9, dtype=np.float32),
                'market_mask': np.zeros(9, dtype=np.float32),
                'terminal_return': 100000.0 + i * 5000.0,
                'step': i * 20,
                'baseline_value': 90000.0 + i * 4000.0,
            }
            for i in range(16)
        ]
        dataset = AWILDataset(transitions=sample_transitions, augment=False)
        baseline_net = Stage1ValueBaseline()
        
        item = dataset[0]
        val_pred = baseline_net(item['x_spatial'].unsqueeze(0), item['x_scalar'].unsqueeze(0))
        self.assertEqual(val_pred.shape, (1, 1))
        
        trained_baseline = train_stage1_baseline(baseline_net, dataset, epochs=2, batch_size=4)
        self.assertIsNotNone(trained_baseline)

    def test_dataset_integrity_verification(self):
        sample_transitions = [
            {
                'spatial': np.random.randn(24, 10, 10).astype(np.float32),
                'scalar': np.random.randn(72).astype(np.float32),
                'crop_heatmaps': np.zeros((5, 10, 10), dtype=np.float32),
                'livestock_quotas': np.array([2.0, 1.0, 0.0], dtype=np.float32),
                'workforce': 2,
                'land_expand': 0.5,
                'seed_replenish': np.zeros(5, dtype=np.float32),
                'market_fractions': np.zeros(9, dtype=np.float32),
                'market_mask': np.zeros(9, dtype=np.float32),
                'terminal_return': 120000.0,
                'step': 50,
                'baseline_value': 110000.0
            }
        ]
        dataset = AWILDataset(transitions=sample_transitions, augment=True)
        report = validate_dataset_integrity(dataset, transitions=sample_transitions)
        self.assertEqual(report['status'], 'PASSED')
        self.assertTrue(report['sample_weights_bounded'])
        self.assertTrue(report['d4_transforms_valid'])


if __name__ == '__main__':
    unittest.main()
