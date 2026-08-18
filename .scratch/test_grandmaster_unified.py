import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from kaggle_environments import make
from src.agents.arbitrage_bot import TownShopArbitrageAgent
from src.agents.livestock_bot import LivestockHusbandryAgent

# Let's create an integrated Grandmaster Unified Agent
class GrandmasterMasterAgent:
    def __init__(self):
        self.crop_specs = {
            "WHEAT": {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
            "CARROT": {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
            "TOMATO": {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
            "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
            "MELON": {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
        }
        self.base_prices = {
            "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250,
            "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100,
        }
        self.shops_catalog = {
            "BAKERY": ["EGG", "WHEAT"],
            "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
            "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
            "YARN_STORE": ["WOOL"],
            "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
            "PET_CAFE": ["CARROT"],
            "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
            "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
        }
        self.shed_tiles = [(4, 4), (5, 4), (4, 5), (5, 5)]

    def _get_dist(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_step(self, curr, target):
        cx, cy = curr
        tx, ty = target
        if cx == tx and cy == ty:
            return "PASS"
        dx = tx - cx
        dy = ty - cy
        if abs(dx) >= abs(dy):
            if dx > 0: return "EAST"
            elif dx < 0: return "WEST"
        if dy > 0: return "SOUTH"
        elif dy < 0: return "NORTH"
        return "PASS"

    def __call__(self, obs, config=None):
        player = obs.get("player", 0)
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if player < len(farms) else {}
        private = obs.get("private", {}) or {}
        market = obs.get("market", {}) or {}
        town = obs.get("town", {}) or {}

        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", 0))
        money = float(my_farm.get("money", 3000.0))
        farmer_pos = tuple(my_farm.get("farmer", [4, 4]))
        hands_pos = [tuple(h) for h in my_farm.get("hands", [])]
        tiles = my_farm.get("tiles", [])
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        inventories = private.get("inventories", [])
        hires_today = int(my_farm.get("hires_today", 0))
        unlocked = my_farm.get("unlocked_quadrants", ["NW"])
        market_prices = market.get("prices", {})
        unlocked_shops = town.get("unlocked_shops", [])

        market_orders = []

        # 1. Town shop demand multipliers
        shop_demands = {p: 1.0 for p in self.base_prices}
        for shop in unlocked_shops:
            prods = self.shops_catalog.get(shop, [])
            mult = 2.5 if len(prods) == 1 else 1.5
            for p in prods:
                shop_demands[p] = shop_demands.get(p, 1.0) + mult

        # Count farm assets
        living_geese = 0
        living_cows = 0
        living_sheep = 0
        empty_coops = 0
        empty_pastures = 0
        planted_wheat = 0
        planted_crops = {c: 0 for c in self.crop_specs}
        animal_tiles = []
        crop_tiles = []
        unlocked_empty_tiles = []

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
                        crop_name = t.get("crop", "CARROT")
                        crop_tiles.append((c, r, t))
                        planted_crops[crop_name] = planted_crops.get(crop_name, 0) + 1
                        if crop_name == "WHEAT":
                            planted_wheat += 1

        total_animals = living_geese + living_cows + living_sheep

        # Market Sales: sell all non-animal goods from shed
        for item, count in shed.items():
            if count > 0 and item not in ["GOOSE", "COW", "SHEEP"]:
                cur_price = market_prices.get(item, self.base_prices.get(item, 25))
                base_price = self.base_prices.get(item, 25)
                if item == "WHEAT":
                    feed_reserve = max(4, total_animals * 2) if day < 27 else 0
                    sell_wheat = max(0, count - feed_reserve)
                    if sell_wheat > 0 and (cur_price >= base_price * 0.95 or day >= 26):
                        market_orders.append(["SELL", item, sell_wheat])
                elif item == "FERTILIZER":
                    if len(crop_tiles) > 0 and day <= 22 and count > 2:
                        market_orders.append(["SELL", item, count - 2])
                    else:
                        market_orders.append(["SELL", item, count])
                else:
                    if cur_price >= base_price * 0.90 or day >= 26:
                        market_orders.append(["SELL", item, count])

        # Wheat reserve safety buy
        wheat_price = market_prices.get("WHEAT", 25)
        wheat_in_shed = shed.get("WHEAT", 0)
        daily_feed_need = total_animals
        if wheat_in_shed < daily_feed_need and total_animals > 0 and money >= 60 and day < 28:
            buy_qty = min(6, max(4, daily_feed_need * 2) - wheat_in_shed)
            if buy_qty > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_qty])
                money -= wheat_price * buy_qty
                shed["WHEAT"] = wheat_in_shed + buy_qty

        # Land Expansion
        if day <= 16:
            if "NE" not in unlocked and money >= 1250:
                market_orders.append(["BUY_LAND"])
                money -= 1000
            elif "SW" not in unlocked and "NE" in unlocked and money >= 2400 and day <= 12:
                market_orders.append(["BUY_LAND"])
                money -= 2000

        # Labor Hiring (2 farmhands daily for $2 total)
        if day < 29 and hour <= 1 and hires_today < 2 and money >= 30:
            market_orders.append(["HIRE"])
            money -= 2

        # Animal purchasing
        geese_in_shed = shed.get("GOOSE", 0)
        cows_in_shed = shed.get("COW", 0)
        sheep_in_shed = shed.get("SHEEP", 0)

        target_geese = 2
        target_cows = 2
        target_sheep = 2

        if empty_coops > geese_in_shed and (living_geese + geese_in_shed) < target_geese and money >= 400 and day <= 18:
            market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
            money -= 300
            geese_in_shed += 1
        elif empty_pastures > (cows_in_shed + sheep_in_shed) and (living_cows + cows_in_shed) < target_cows and money >= 550 and day <= 16:
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400
            cows_in_shed += 1
        elif empty_pastures > (cows_in_shed + sheep_in_shed) and (living_sheep + sheep_in_shed) < target_sheep and money >= 650 and day <= 14:
            market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
            money -= 500
            sheep_in_shed += 1

        # Wheat seed purchasing for feed
        wheat_seeds = seeds.get("WHEAT", 0)
        if day < 24 and (wheat_seeds + planted_wheat) < 6 and money >= 40:
            qty = min(4, 6 - (wheat_seeds + planted_wheat))
            if qty > 0:
                market_orders.append(["BUY_SEED", "WHEAT", qty])
                money -= 10 * qty
                seeds["WHEAT"] = wheat_seeds + qty

        # High-yield crop seeds purchasing
        crop_scores = [
            ("CARROT", shop_demands.get("CARROT", 1.0) * (35 / 20) if day <= 25 else 0.0),
            ("WHEAT", shop_demands.get("WHEAT", 1.0) * (25 / 10) if day <= 23 else 0.0),
            ("TOMATO", shop_demands.get("TOMATO", 1.0) * (60 / 50) if day <= 18 else 0.0),
            ("STRAWBERRY", shop_demands.get("STRAWBERRY", 1.0) * (120 / 100) if day <= 16 else 0.0),
            ("MELON", (250 / 80) if day <= 15 else 0.0),
        ]
        crop_scores.sort(key=lambda x: x[1], reverse=True)

        if day < 26 and money >= 60:
            for crop_name, score in crop_scores:
                if score <= 0:
                    continue
                cur_planted = planted_crops.get(crop_name, 0)
                cur_seed = seeds.get(crop_name, 0)
                cost = self.crop_specs[crop_name]["seed"]
                if cur_planted + cur_seed < 16 and money >= cost * 2:
                    qty = min(4, 16 - (cur_planted + cur_seed), int(money // cost))
                    if qty > 0:
                        market_orders.append(["BUY_SEED", crop_name, qty])
                        money -= cost * qty
                        seeds[crop_name] = cur_seed + qty
                        break

        # Field chores
        feed_tasks = []
        care_tasks = []
        harvest_animal_tasks = []
        fertilizer_tasks = []
        water_crop_tasks = []
        harvest_crop_tasks = []
        fertilize_crop_tasks = []
        weed_tasks = []
        plant_tasks = []
        build_coop_tasks = []
        build_pasture_tasks = []
        place_animal_tasks = []

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
            crop = ctile.get("crop", "CARROT")
            cspec = self.crop_specs.get(crop, self.crop_specs["CARROT"])
            age = day - ctile.get("planted_day", day)
            is_ongoing = cspec["ongoing"]
            yield_u = ctile.get("yield_units", 0)

            if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
            ):
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

        unlocked_empty_tiles.sort(key=lambda p: self._get_dist(p, (4, 4)))
        curr_e_idx = 0
        if (living_geese + empty_coops) < target_geese and curr_e_idx < len(unlocked_empty_tiles) and day <= 18:
            build_coop_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1

        if (living_cows + living_sheep + empty_pastures) < (target_cows + target_sheep) and curr_e_idx < len(unlocked_empty_tiles) and day <= 16:
            build_pasture_tasks.append(unlocked_empty_tiles[curr_e_idx])
            curr_e_idx += 1

        total_seeds_count = sum(seeds.values())
        if day < 26 and total_seeds_count > 0:
            for p in unlocked_empty_tiles[curr_e_idx:24]:
                plant_tasks.append(p)

        units = [farmer_pos] + hands_pos
        unit_actions = []
        claimed_tasks = set()

        for u_idx, u_pos in enumerate(units):
            ux, uy = u_pos
            u_tile = tiles[uy][ux] if uy < len(tiles) and ux < len(tiles[uy]) else None
            u_inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_at_shed = u_pos in self.shed_tiles

            if is_at_shed:
                placed_target = next((pt for pt in place_animal_tasks if pt[:2] not in claimed_tasks), None)
                if placed_target is not None:
                    target_animal = placed_target[2]
                    if shed.get(target_animal, 0) > 0 and u_inv.get(target_animal, 0) == 0:
                        unit_actions.append(["PICKUP", target_animal, 1])
                        shed[target_animal] -= 1
                        u_inv[target_animal] = u_inv.get(target_animal, 0) + 1
                        continue

                if feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    qty = min(4, shed.get("WHEAT", 0), len(feed_tasks))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "WHEAT", qty])
                        shed["WHEAT"] -= qty
                        u_inv["WHEAT"] = u_inv.get("WHEAT", 0) + qty
                        continue

                if fertilize_crop_tasks and u_inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                    qty = min(2, shed.get("FERTILIZER", 0))
                    if qty > 0:
                        unit_actions.append(["PICKUP", "FERTILIZER", qty])
                        shed["FERTILIZER"] -= qty
                        u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + qty
                        continue

            if isinstance(u_tile, dict):
                k = u_tile.get("kind")
                animal = u_tile.get("animal")
                if k in ["COOP", "PASTURE"]:
                    if animal is not None:
                        if not u_tile.get("fed_today", False) and u_inv.get("WHEAT", 0) > 0:
                            unit_actions.append(["FEED"])
                            u_inv["WHEAT"] -= 1
                            continue
                        if not u_tile.get("cared_today", False):
                            unit_actions.append(["CARE"])
                            continue
                        if u_tile.get("fertilizer_available", False):
                            unit_actions.append(["COLLECT_FERTILIZER"])
                            u_inv["FERTILIZER"] = u_inv.get("FERTILIZER", 0) + 1
                            continue
                        if u_tile.get("yield_units", 0) > 0:
                            unit_actions.append(["HARVEST"])
                            continue
                    else:
                        for aname in ["GOOSE", "COW", "SHEEP"]:
                            if u_inv.get(aname, 0) > 0:
                                req_struct = "COOP" if aname == "GOOSE" else "PASTURE"
                                if k == req_struct:
                                    unit_actions.append(["PLACE", aname])
                                    u_inv[aname] -= 1
                                    break
                        if len(unit_actions) > u_idx:
                            continue
                elif k == "PLANT":
                    crop = u_tile.get("crop", "CARROT")
                    cspec = self.crop_specs.get(crop, self.crop_specs["CARROT"])
                    age = day - u_tile.get("planted_day", day)
                    is_ongoing = cspec["ongoing"]
                    yield_u = u_tile.get("yield_units", 0)

                    if (not is_ongoing and (age >= cspec["max_yield_day"] or (day >= 28 and yield_u > 0))) or (
                        is_ongoing and age >= cspec["first_yield_day"] and yield_u > 0
                    ):
                        unit_actions.append(["HARVEST"])
                        continue
                    if not u_tile.get("watered_today", False):
                        unit_actions.append(["WATER"])
                        continue
                    if u_tile.get("fertilized_until_day", -1) < day and u_inv.get("FERTILIZER", 0) > 0:
                        unit_actions.append(["FERTILIZE"])
                        u_inv["FERTILIZER"] -= 1
                        continue
                elif k == "WEED":
                    unit_actions.append(["DIG"])
                    continue
            elif u_tile is None:
                if u_pos in build_coop_tasks:
                    unit_actions.append(["BUILD_COOP"])
                    build_coop_tasks.remove(u_pos)
                    continue
                elif u_pos in build_pasture_tasks:
                    unit_actions.append(["BUILD_PASTURE"])
                    build_pasture_tasks.remove(u_pos)
                    continue
                elif u_pos in plant_tasks and total_seeds_count > 0 and day < 26:
                    for crop_cand, _ in crop_scores:
                        if seeds.get(crop_cand, 0) > 0:
                            unit_actions.append(["PLANT", crop_cand])
                            seeds[crop_cand] -= 1
                            total_seeds_count -= 1
                            break
                    else:
                        unit_actions.append(["PASS"])
                    continue

            best_target = None
            best_dist = 999
            has_animal_in_hand = any(u_inv.get(a, 0) > 0 for a in ["GOOSE", "COW", "SHEEP"])
            if has_animal_in_hand:
                for pt in place_animal_tasks:
                    pos = pt[:2]
                    if pos not in claimed_tasks:
                        d = self._get_dist(u_pos, pos)
                        if d < best_dist:
                            best_dist = d
                            best_target = pos

            if best_target is None and u_inv.get("WHEAT", 0) > 0 and feed_tasks:
                for task in feed_tasks:
                    if task not in claimed_tasks:
                        d = self._get_dist(u_pos, task)
                        if d < best_dist:
                            best_dist = d
                            best_target = task

            if best_target is None and feed_tasks and u_inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                nearest_shed = min(self.shed_tiles, key=lambda s: self._get_dist(u_pos, s))
                best_target = nearest_shed
                best_dist = self._get_dist(u_pos, nearest_shed)

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
                    d = self._get_dist(u_pos, task)
                    if d < best_dist:
                        best_dist = d
                        best_target = task

            if best_target is not None:
                claimed_tasks.add(best_target)
                unit_actions.append([self._get_step(u_pos, best_target)])
            else:
                if not is_at_shed:
                    nearest_shed = min(self.shed_tiles, key=lambda s: self._get_dist(u_pos, s))
                    unit_actions.append([self._get_step(u_pos, nearest_shed)])
                else:
                    unit_actions.append(["PASS"])

        farmer_action = unit_actions[0] if unit_actions else ["PASS"]
        hands_actions = unit_actions[1:] if len(unit_actions) > 1 else []

        return {
            "farmer": farmer_action,
            "hands": hands_actions,
            "market": market_orders[:10],
        }

gm_master = GrandmasterMasterAgent()

for opp in ["starter", "random", "pass"]:
    t0 = time.time()
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)
    env.run([gm_master, opp])
    dur = time.time() - t0
    r0 = env.steps[-1][0].reward
    r1 = env.steps[-1][1].reward
    print(f"GM Master vs {opp.upper():7s}: R0=${r0:,.2f} vs R1=${r1:,.2f} | Winner: {'GM Master' if r0 > r1 else opp} | Latency: {dur*1000/720:.2f}ms/turn")
