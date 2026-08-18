"""
Prioritized Fictitious Self-Play (PFSP) Single-Champion League Matchmaker.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np


class HeuristicSpecialist:
    """
    Fixed Mega League heuristic specialists (Carrot Monoculture, Melon Rusher, Dairy Syndicate, etc.).
    """
    SPECIALISTS = [
        "DeterministicGrandmaster",
        "CarrotMonoculture",
        "MelonRusher",
        "DairySyndicate",
        "MarketPriceCrasher"
    ]

    def __init__(self, personality: str = "DeterministicGrandmaster"):
        self.personality = personality

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        # Minimal deterministic baseline dispatch
        player = obs.get("player", 0)
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        money = float(my_farm.get("money", 0.0))
        day = int(obs.get("day", 0))

        market_orders = []
        if day < 27 and money >= 80:
            if self.personality == "MelonRusher" and day <= 5:
                market_orders.append(["BUY_SEED", "MELON", 1])
            elif self.personality == "CarrotMonoculture":
                market_orders.append(["BUY_SEED", "CARROT", 2])
            else:
                market_orders.append(["BUY_SEED", "CARROT", 1])

        return {
            "farmer": ["PASS"],
            "hands": [],
            "market": market_orders,
        }


class PFSPLadder:
    """
    Tracks past checkpoints and computes PFSP probability distribution:
    P(i) propto max(0.05, (1 - WinRate_i)^1.5)
    """
    def __init__(self, max_checkpoints: int = 30):
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Dict[str, Any]] = []

    def add_checkpoint(self, ckpt_id: str, win_rate_vs_champion: float = 0.5):
        if len(self.checkpoints) >= self.max_checkpoints:
            self.checkpoints.pop(0)  # FIFO rolling pool
        
        self.checkpoints.append({
            "id": ckpt_id,
            "win_rate_vs_champion": win_rate_vs_champion,
            "champ_win_rate": 1.0 - win_rate_vs_champion,
        })

    def update_result(self, ckpt_id: str, champ_won: bool):
        for ckpt in self.checkpoints:
            if ckpt["id"] == ckpt_id:
                old_wr = ckpt["champ_win_rate"]
                new_wr = 0.9 * old_wr + 0.1 * (1.0 if champ_won else 0.0)
                ckpt["champ_win_rate"] = new_wr
                ckpt["win_rate_vs_champion"] = 1.0 - new_wr
                break

    def get_probabilities(self) -> np.ndarray:
        if not self.checkpoints:
            return np.array([])
        
        weights = []
        for ckpt in self.checkpoints:
            # Checkpoints where champ win rate is low (champ loses often) receive highest sampling probability
            p_i = max(0.05, (1.0 - ckpt["champ_win_rate"]) ** 1.5)
            weights.append(p_i)
        
        weights_arr = np.array(weights, dtype=np.float64)
        return weights_arr / weights_arr.sum()


class PrioritizedFictitiousSelfPlayMatchmaker:
    """
    Matchmaking selector for Single-Champion League:
    - 40% rolling past checkpoints (PFSP weighted)
    - 40% self-play mirror with Dirichlet exploration noise
    - 20% fixed Mega League heuristic specialists
    """
    def __init__(self, max_checkpoints: int = 30):
        self.ladder = PFSPLadder(max_checkpoints=max_checkpoints)

    def add_checkpoint(self, ckpt_id: str, win_rate_vs_champion: float = 0.5):
        self.ladder.add_checkpoint(ckpt_id, win_rate_vs_champion)

    def sample_opponent(self) -> Tuple[str, str]:
        has_ckpts = len(self.ladder.checkpoints) > 0
        
        if has_ckpts:
            mode = np.random.choice(["checkpoint", "self_play", "heuristic"], p=[0.40, 0.40, 0.20])
        else:
            mode = np.random.choice(["self_play", "heuristic"], p=[0.70, 0.30])

        if mode == "checkpoint" and has_ckpts:
            probs = self.ladder.get_probabilities()
            ckpt_idx = np.random.choice(len(self.ladder.checkpoints), p=probs)
            return "checkpoint", self.ladder.checkpoints[ckpt_idx]["id"]
        elif mode == "heuristic":
            specialist = np.random.choice(HeuristicSpecialist.SPECIALISTS)
            return "heuristic", specialist
        else:
            return "self_play", "champion"
