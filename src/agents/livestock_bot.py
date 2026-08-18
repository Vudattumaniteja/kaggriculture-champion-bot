"""
Livestock Husbandry and Compounding Care Multiplier baseline agent.
Builds coops and pastures, raises geese, cows, and sheep, grows self-feeding wheat,
banks daily CARE bonuses, and harvests premium eggs, milk, wool, and fertilizer.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from src.agents.utils import DIRS, SHED_TILES, get_manhattan_dist, get_step_towards, parse_observation
except ImportError:
    from utils import DIRS, SHED_TILES, get_manhattan_dist, get_step_towards, parse_observation


class LivestockHusbandryAgent:
    def __init__(
        self,
        target_geese: int = 2,
        target_cows: int = 2,
        target_sheep: int = 2,
        wheat_tiles: int = 6,
    ):
        self.target_geese = target_geese
        self.target_cows = target_cows
        self.target_sheep = target_sheep
        self.wheat_tiles = wheat_tiles

    def __call__(self, obs: Dict[str, Any], config: Any = None) -> Dict[str, Any]:
        state = parse_observation(obs)
        day = state["day"]
        hour = state["hour"]
        step = state["step"]
        money = state["money"]
        farmer_pos = state["farmer_pos"]
        hands_pos = state["hands_pos"]
        tiles = state["tiles"]
        shed = state["shed"]
        seeds = state["seeds"]
        inventories = state["inventories"]
        hires_today = state["hires_today"]
        unlocked = state["unlocked_quads"]

        market_orders: List[List[Any]] = []

        # --- 1. COUNT FARM ASSETS & STRUCTURES ---
        living_geese = 0
        living_cows = 0
        living_sheep = 0
        empty_coops = 0
        empty_pastures = 0
        planted_wheat = 0
        unlocked_empty_tiles: List[Tuple[int, int]] = []
        animal_tiles: List[Tuple[int, int, str, Dict[str, Any]]] = []  # (x, y, kind, tile_dict)
        crop_tiles: List[Tuple[int, int, Dict[str, Any]]] = []

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if t is None:
                    unlocked_empty_tiles.append(pos)
                elif isinstance(t, dict):
                    k = t.get("kind")
                    animal = t.get("animal")
                    if k == "COOP":
                        if animal == "GOOSE":
                            living_geese += 1
                            animal_tiles.append((c, r, "GOOSE", t))
                        elif animal is None:
                            empty_coops += 1
                    elif k == "PASTURE":
                        if animal == "COW":
                            living_cows += 1
                            animal_tiles.append((c, r, "COW", t))
                        elif animal == "SHEEP":
                            living_sheep += 1
                            animal_tiles.append((c, r, "SHEEP", t))
                        elif animal is None:
                            empty_pastures += 1
                    elif k == "PLANT":
                        crop_tiles.append((c, r, t))
                        if t.get("crop") == "WHEAT":
                            planted_wheat += 1

        total_living_animals = living_geese + living_cows + living_sheep

        # --- 2. MARKET STRATEGY ---
        # A. Sell harvested high-value animal goods and excess produce from shed
        for item in ["EGG", "MILK", "WOOL", "FERTILIZER", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]:
            count = shed.get(item, 0)
            if count > 0:
                # Keep up to 2 fertilizer for crops if we have crops, otherwise sell all
                if item == "FERTILIZER" and len(crop_tiles) > 0 and count > 2:
                    market_orders.append(["SELL", item, count - 2])
                elif item != "FERTILIZER":
                    market_orders.append(["SELL", item, count])

        # B. Wheat feed reserve management
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_need = total_living_animals
        safe_wheat_reserve = max(6, daily_feed_need * 3)

        if wheat_in_shed > safe_wheat_reserve + 8:
            market_orders.append(["SELL", "WHEAT", wheat_in_shed - safe_wheat_reserve])
        elif wheat_in_shed < daily_feed_need and money >= 50 and day < 28:
            # Emergency wheat buy for animal feeding if crops haven't yielded
            buy_qty = min(6, safe_wheat_reserve - wheat_in_shed)
            if buy_qty > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_qty])
                money -= 30 * buy_qty

        # C. Seed Purchasing for Wheat
        wheat_seeds = seeds.get("WHEAT", 0)
        if day < 26 and (wheat_seeds + planted_wheat) < self.wheat_tiles and money >= 40:
            qty = min(4, self.wheat_tiles - (wheat_seeds + planted_wheat))
            if qty > 0:
                market_orders.append(["BUY_SEED", "WHEAT", qty])
                money -= 10 * qty

        # D. Animal Purchasing: Buy animals if structure is ready or planned
        geese_in_shed = shed.get("GOOSE", 0)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)

        # Goose ($300): fast 4-day first yield, daily eggs ($50)
        if (
            empty_coops > geese_in_shed
            and (living_geese + geese_in_shed) < self.target_geese
            and money >= 400
            and day <= 22
        ):
            market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
            money -= 300
            geese_in_shed += 1
        # Cow ($400): milk ($160) every 2 days
        elif (
            empty_pastures > (cows_in_shed + sheep_in_shed)
            and (living_cows + cows_in_shed) < self.target_cows
            and money >= 550
            and day <= 18
        ):
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_in_shed += 1
        # Sheep ($500): wool ($200) every 3 days + huge care multipliers
        elif (
            empty_pastures > (cows_in_shed + sheep_in_shed)
            and (living_sheep + sheep_in_shed) < self.target_sheep
            and money >= 650
            and day <= 16
        ):
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_in_shed += 1

        # E. Land Expansion
        if day <= 16 and "NE" not in unlocked and money >= 1400:
            market_orders.append(["BUY_LAND"])
            money -= 1000

        # F. Labor Hiring (Hire 2 farmhands daily)
        if day < 29 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])
            money -= 2

        # --- 3. FIELD CHORES SCAN & PLAN ---
        feed_tasks: List[Tuple[int, int]] = []
        care_tasks: List[Tuple[int, int]] = []
        harvest_animal_tasks: List[Tuple[int, int]] = []
        fertilizer_tasks: List[Tuple[int, int]] = []

        water_crop_tasks: List[Tuple[int, int]] = []
        harvest_crop_tasks: List[Tuple[int, int]] = []
        fertilize_crop_tasks: List[Tuple[int, int]] = []

        weed_tasks: List[Tuple[int, int]] = []
        plant_tasks: List[Tuple[int, int]] = []
        build_coop_tasks: List[Tuple[int, int]] = []
        build_pasture_tasks: List[Tuple[int, int]] = []
        place_animal_tasks: List[Tuple[int, int, str]] = []  # (x, y, animal_type)

        for ax, ay, aname, atile in animal_tiles:
            pos = (ax, ay)
            if not atile.get("fed_today", False):
                feed_tasks.append(pos)
            if not atile.get("cared_today", False):
                care_tasks.append(pos)
            if atile.get("yield_units", 0) > 0:
                harvest_animal_tasks.append(pos)
            if atile.get("fertilizer_available", False):
                fertilizer_tasks.append(pos)

        for cx, cy, ctile in crop_tiles:
            pos = (cx, cy)
            crop = ctile.get("crop", "WHEAT")
            age = day - ctile.get("planted_day", day)
            if age >= 4 or (day >= 28 and ctile.get("yield_units", 0) > 0):
                harvest_crop_tasks.append(pos)
            elif not ctile.get("watered_today", False):
                water_crop_tasks.append(pos)
            elif ctile.get("fertilized_until_day", -1) < day:
                fertilize_crop_tasks.append(pos)

        for r in range(len(tiles)):
            for c in range(len(tiles[r])):
                t = tiles[r][c]
                pos = (c, r)
                if isinstance(t, dict):
                    if t.get("kind") == "WEED":
                        weed_tasks.append(pos)
                    elif t.get("kind") == "COOP" and "animal" not in t:
                        place_animal_tasks.append((c, r, "GOOSE"))
                    elif t.get("kind") == "PASTURE" and "animal" not in t:
                        if (living_cows + cows_in_shed) <= (living_sheep + sheep_in_shed):
                            place_animal_tasks.append((c, r, "COW"))
                        else:
                            place_animal_tasks.append((c, r, "SHEEP"))

        # Sort empty tiles by proximity to shed
        unlocked_empty_tiles.sort(key=lambda p: get_manhattan_dist(p, (4, 4)))

        # Build structures planning
        total_coops = living_geese + empty_coops
        total_pastures = living_cows + living_sheep + empty_pastures

        curr_empty_idx = 0
        if total_coops < self.target_geese and curr_empty_idx < len(unlocked_empty_tiles) and day <= 20:
            build_coop_tasks.append(unlocked_empty_tiles[curr_empty_idx])
            curr_empty_idx += 1

        if total_pastures < (self.target_cows + self.target_sheep) and curr_empty_idx < len(unlocked_empty_tiles) and day <= 18:
            build_pasture_tasks.append(unlocked_empty_tiles[curr_empty_idx])
            curr_empty_idx += 1

        for p in unlocked_empty_tiles[curr_empty_idx : curr_empty_idx + self.wheat_tiles]:
            if wheat_seeds > 0 and day < 27:
                plant_tasks.append(p)

        # --- 4. MULTI-WORKER COORDINATION & ACTIONS ---
        units = [farmer_pos] + hands_pos
        unit_actions: List[List[Any]] = []
        claimed_tasks: Set[Tuple[int, int]] = set()

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in SHED_TILES

            # 1. SHED PICKUP / PREPARATION
            # If at shed, check what this unit should pick up:
            # - Animal waiting in shed to be placed?
            # - Wheat for feeding hungry animals?
            # - Fertilizer for crops?
            if is_at_shed:
                # Pickup animal if structure is waiting and unit doesn't have an animal
                placed_target = next((pt for pt in place_animal_tasks if pt[:2] not in claimed_tasks), None)
                if placed_target is not None:
                    target_animal = placed_target[2]
                    if shed.get(target_animal, 0) > 0 and u_inv.get(target_animal, 0) == 0:
                        unit_actions.append(["PICKUP", target_animal, 1])
                        shed[target_animal] -= 1
                        u_inv[target_animal] = u_inv.get(target_animal, 0) + 1
                        continue

                # Pickup Wheat for feeding if animals are hungry and unit has no wheat
                if feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    qty = min(4, shed.get("WHEAT", 0), len(feed_tasks))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "WHEAT", qty])
                        shed["WHEAT"] -= qty
                        u_inv["WHEAT"] = u_inv.get("WHEAT", 0) + qty
                        continue

                # Pickup Fertilizer if crops need fertilizing
                if fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "FERTILIZER", qty])
                        shed["FERTILIZER"] -= qty
                        u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + qty
                        continue

            # 2. IMMEDIATE ON-TILE OPERATIONS
            if isinstance(u_tile, dict):
                k = u_tile.get("kind")
                animal = u_tile.get("animal")

                if k in ["COOP", "PASTURE"]:
                    if animal is not None:
                        # A. Feed animal if hungry and we have wheat
                        if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                            unit_actions.append(["FEED"])
                            u_inv["WHEAT"] -= 1
                            continue
                        # B. Care for animal daily (compounding multiplier!)
                        if not u_tile.get("cared_today", False):
                            unit_actions.append(["CARE"])
                            continue
                        # C. Collect daily fertilizer
                        if u_tile.get("fertilizer_available", False):
                            unit_actions.append(["COLLECT_FERTILIZER"])
                            u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + 1
                            continue
                        # D. Harvest animal produce (Eggs, Milk, Wool)
                        if u_tile.get("yield_units", 0) > 0:
                            unit_actions.append(["HARVEST"])
                            continue
                    else:
                        # Empty coop / pasture -> Place matching animal if in inventory
                        for aname in ["GOOSE", "COW", "SHEEP"]:
                            if u_inv.get(aname, 0) > 0:
                                req_struct = "COOP" if aname == "GOOSE" else "PASTURE"
                                if k == req_struct:
                                    unit_actions.append(["PLACE", aname])
                                    u_inv[aname] -= 1
                                    break
                        else:
                            pass
                        if len(unit_actions) > u_idx:
                            continue

                elif k == "PLANT":
                    age = day - u_tile.get("planted_day", day)
                    # Harvest mature wheat
                    if age >= 4 or (day >= 28 and u_tile.get("yield_units", 0) > 0):
                        unit_actions.append(["HARVEST"])
                        continue
                    # Fertilize if holding fertilizer
                    if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0:
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue
                    # Water plant daily
                    if not u_tile.get("watered_today", False):
                        unit_actions.append(["WATER"])
                        continue

                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue

            elif u_tile is None:
                # Check build actions
                if u_pos in build_coop_tasks:
                    unit_actions.append(["BUILD_COOP"])
                    build_coop_tasks.remove(u_pos)
                    continue
                elif u_pos in build_pasture_tasks:
                    unit_actions.append(["BUILD_PASTURE"])
                    build_pasture_tasks.remove(u_pos)
                    continue
                elif u_pos in plant_tasks and wheat_seeds > 0 and day < 27:
                    unit_actions.append(["PLANT", "WHEAT"])
                    wheat_seeds -= 1
                    continue

            # 3. CHORE DISPATCH & NAVIGATION
            # Determine best target for this unit based on its inventory and needs
            best_target: Optional[Tuple[int, int]] = None
            best_dist = 999

            # If unit has an animal to place, head to empty coop/pasture
            has_animal_in_hand = any(u_inv.get(a, 0) > 0 for a in ["GOOSE", "COW", "SHEEP"])
            if has_animal_in_hand:
                for pt in place_animal_tasks:
                    pos = pt[:2]
                    if pos not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, pos)
                        if d < best_dist:
                            best_dist = d
                            best_target = pos

            # If unit has wheat, prioritize feeding unfed animals
            if best_target is None and u_inv.get("WHEAT", 0) > 0 and feed_tasks:
                for task in feed_tasks:
                    if task not in claimed_tasks:
                        d = get_manhattan_dist(u_pos, task)
                        if d < best_dist:
                            best_dist = d
                            best_target = task

            # If unit needs wheat to feed and shed has wheat, visit shed
            if best_target is None and feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                best_target = nearest_shed
                best_dist = get_manhattan_dist(u_pos, nearest_shed)

            # High priority field chores: CARE, WATER, HARVEST, COLLECT_FERTILIZER, BUILD, PLANT
            if best_target is None:
                priority_queue = (
                    care_tasks
                    + water_crop_tasks
                    + harvest_animal_tasks
                    + harvest_crop_tasks
                    + fertilizer_tasks
                    + build_coop_tasks
                    + build_pasture_tasks
                    + plant_tasks
                    + weed_tasks
                )
                for task in priority_queue:
                    if task in claimed_tasks:
                        continue
                    d = get_manhattan_dist(u_pos, task)
                    if d < best_dist:
                        best_dist = d
                        best_target = task

            if best_target is not None:
                claimed_tasks.add(best_target)
                unit_actions.append([get_step_towards(u_pos, best_target)])
            else:
                # Return near shed if far
                if not is_at_shed:
                    nearest_shed = min(SHED_TILES, key=lambda s: get_manhattan_dist(u_pos, s))
                    unit_actions.append([get_step_towards(u_pos, nearest_shed)])
                else:
                    unit_actions.append(["PASS"])

        farmer_action = unit_actions[0] if unit_actions else ["PASS"]
        hands_actions = unit_actions[1:] if len(unit_actions) > 1 else []

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_orders[:10],
        }


livestock_agent = LivestockHusbandryAgent()


def agent(obs, config=None):
    return livestock_agent(obs, config)
