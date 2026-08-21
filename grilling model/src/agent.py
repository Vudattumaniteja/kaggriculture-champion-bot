"""
Champion Bot Agent Callable Interface complying with Kaggle Environment contract.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch

from src.encoder import encode_observation
from src.network import (
    ChampionFullNetwork,
    build_action_masks,
    apply_action_masks,
    get_gumbel_temperature,
    sample_gumbel_action,
)
from src.micro_solver import apply_market_guardrails, solve_micro_actions


class ChampionAgent:
    """
    Hierarchical 2-Tier Champion Agent integrating:
    1. Observation Encoder -> 24-channel spatial tensor & 72-dim scalar economic vector
    2. FiLM SE-ResNet Backbone Trunk & Decoupled Multi-Head Policy
    3. Two-Stage Market Guardrails (cow feed reservation, midnight shed overflow)
    4. Neural-Weighted Hungarian Micro Assignment Solver
    5. Two-Stage Gumbel Evaluation Temperature (tau=1.0 for t<48, tau=0.0 for t>=48)
    """
    def __init__(self, model_weights_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.network = ChampionFullNetwork().to(self.device)
        self.network.eval()
        if model_weights_path is not None:
            state_dict = torch.load(model_weights_path, map_location=self.device)
            if isinstance(state_dict, dict) and "network_state_dict" in state_dict:
                state_dict = state_dict["network_state_dict"]
            self.network.load_state_dict(state_dict)

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        # 1. State Encoding
        x_spatial_np, x_scalar_np = encode_observation(obs)
        x_spatial = torch.from_numpy(x_spatial_np).unsqueeze(0).to(self.device)
        x_scalar = torch.from_numpy(x_scalar_np).unsqueeze(0).to(self.device)

        # 2. Neural Forward Pass
        with torch.no_grad():
            outputs = self.network(x_spatial, x_scalar)

        crop_heatmaps = outputs["crop_heatmaps"].squeeze(0).cpu().numpy()
        livestock_quotas = outputs["livestock_quotas"].squeeze(0).cpu().numpy()
        workforce_logits = outputs["workforce_logits"].squeeze(0)
        land_expand_logit = outputs["land_expand_logit"].item()
        seed_replenish_logits = outputs["seed_replenish_logits"].squeeze(0).cpu().numpy()
        market_fractions = outputs["market_fractions"].squeeze(0).cpu().numpy()

        # 3. Action Masks & Two-Stage Gumbel Sampling
        player = obs.get("player", 0)
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        money = float(my_farm.get("money", 0.0))
        unlocked_quads = my_farm.get("unlocked_quadrants", ["NW"])
        num_workers = len(my_farm.get("hands", []))
        shed = (obs.get("private", {}) or {}).get("shed", {})

        masks = build_action_masks(money, unlocked_quads, num_workers, shed)
        step = int(obs.get("step", 0))
        tau = get_gumbel_temperature(step)
        workforce_idx = sample_gumbel_action(workforce_logits, masks["workforce_mask"], temperature=tau)

        # 4. Market Guardrails
        market_orders = apply_market_guardrails(
            market_fractions=market_fractions,
            obs=obs,
            seed_replenish_logits=seed_replenish_logits,
            land_expand_logit=land_expand_logit if masks["land_expand_mask"] > 0.5 else -1e9,
            workforce_logit_idx=workforce_idx,
            livestock_quotas=livestock_quotas,
        )

        # 5. Hungarian Micro Chore Assignment
        farmer_act, hands_acts = solve_micro_actions(
            obs=obs,
            crop_heatmaps=crop_heatmaps,
            livestock_quotas=livestock_quotas
        )

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders,
        }


# Global singleton instance for Kaggle environment runner
_agent_instance = ChampionAgent()


def agent(obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
    return _agent_instance(obs, config)
