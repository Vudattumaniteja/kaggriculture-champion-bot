import sys
import os
from pathlib import Path

grilling_model_root = Path(__file__).resolve().parent.parent
if str(grilling_model_root) not in sys.path:
    sys.path.insert(0, str(grilling_model_root))

import unittest
import torch
import numpy as np
from kaggle_environments import make

from src.network import ChampionFullNetwork
from build_submission import build_submission_file, add_mask_safe_dirichlet_noise, get_cosine_entropy_coeff


class TestSubmissionPackaging(unittest.TestCase):
    def test_dirichlet_mask_safety(self):
        # 13 actions, only first 3 valid
        logits = torch.zeros(1, 13)
        mask = torch.zeros(1, 13)
        mask[0, :3] = 1.0
        
        noisy_probs = add_mask_safe_dirichlet_noise(logits, mask, alpha=0.3, epsilon=0.25)
        self.assertEqual(noisy_probs.shape, (1, 13))
        # Masked actions (>2) must receive strictly 0.0 probability
        for k in range(3, 13):
            self.assertEqual(noisy_probs[0, k].item(), 0.0)
        # Sum of unmasked must equal 1.0
        self.assertAlmostEqual(noisy_probs.sum().item(), 1.0, places=5)

    def test_cosine_entropy_decay_schedule(self):
        c_start = get_cosine_entropy_coeff(step=0, total_steps=100000)
        c_mid = get_cosine_entropy_coeff(step=50000, total_steps=100000)
        c_end = get_cosine_entropy_coeff(step=100000, total_steps=100000)

        self.assertAlmostEqual(c_start, 0.05, places=3)
        self.assertAlmostEqual(c_end, 0.002, places=3)
        self.assertLess(c_end, c_mid)
        self.assertLess(c_mid, c_start)

    def test_build_submission_and_run_full_match(self):
        # 1. Build standalone submission.py
        weights_path = grilling_model_root / "weights" / "champion_weights.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        network = ChampionFullNetwork()
        torch.save(network.state_dict(), weights_path)

        submission_path = grilling_model_root / "submission.py"
        build_submission_file(output_path=str(submission_path), weights_path=str(weights_path))
        
        self.assertTrue(submission_path.exists(), "submission.py was not created")

        # 2. Dynamically import and run agent from submission.py
        import importlib.util
        spec = importlib.util.spec_from_file_location("standalone_submission", str(submission_path))
        submission_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submission_mod)
        
        self.assertTrue(hasattr(submission_mod, "agent"), "submission.py lacks callable agent")

        # 3. Run full match against starter baseline
        env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
        env.run([submission_mod.agent, "starter"])
        
        last_step = env.steps[-1]
        champ_reward = last_step[0].get("reward", 0.0) or 0.0
        starter_reward = last_step[1].get("reward", 0.0) or 0.0
        
        # Verify valid match execution
        self.assertEqual(len(env.steps), 720)
        self.assertGreater(champ_reward, 0.0, "Champion should finish with positive cash balance")


if __name__ == "__main__":
    unittest.main()
