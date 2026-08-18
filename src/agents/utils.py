"""
Helper utilities, pathfinding, and observation parsing for Kaggriculture agents.
"""

from typing import Any, Dict, List, Optional, Tuple

# Direction mappings
DIRS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}

SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]


def get_manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def get_step_towards(curr: Tuple[int, int], target: Tuple[int, int]) -> str:
    """Returns the single-step movement command from curr toward target."""
    cx, cy = curr
    tx, ty = target
    if cx == tx and cy == ty:
        return "PASS"

    dx = tx - cx
    dy = ty - cy

    # Move horizontally or vertically towards target
    if abs(dx) >= abs(dy):
        if dx > 0:
            return "EAST"
        elif dx < 0:
            return "WEST"
    if dy > 0:
        return "SOUTH"
    elif dy < 0:
        return "NORTH"

    return "PASS"


def is_tile_unlocked(tile_data: Any) -> bool:
    return tile_data != "LOCKED"


def parse_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    player = obs["player"]
    my_farm = obs["farms"][player]
    opp_farm = obs["farms"][1 - player]
    private = obs.get("private", {})
    market = obs.get("market", {})
    town = obs.get("town", {})

    return {
        "player": player,
        "step": obs.get("step", 0),
        "day": obs.get("day", 0),
        "hour": obs.get("hour", 0),
        "money": my_farm.get("money", 0.0),
        "farmer_pos": tuple(my_farm.get("farmer", [4, 4])),
        "hands_pos": [tuple(h) for h in my_farm.get("hands", [])],
        "tiles": my_farm.get("tiles", []),
        "unlocked_quads": my_farm.get("unlocked_quadrants", ["NW"]),
        "hires_today": my_farm.get("hires_today", 0),
        "shed": private.get("shed", {}),
        "seeds": private.get("seeds", {}),
        "inventories": private.get("inventories", []),
        "market_prices": market.get("prices", {}),
        "market_inv": market.get("inventory", {}),
        "unlocked_shops": town.get("unlocked_shops", []),
    }
