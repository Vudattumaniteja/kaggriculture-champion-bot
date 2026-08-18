import glob
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

COMMODITIES = ['FERTILIZER', 'WOOL', 'MILK', 'MELON', 'CARROT', 'STRAWBERRY', 'TOMATO', 'WHEAT', 'EGG']

def analyze_episode(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ep_id = int(os.path.basename(file_path).split('-')[1])
    steps = data.get('steps', [])
    num_steps = len(steps)
    team_names = data.get('info', {}).get('TeamNames', ['P0', 'P1'])
    
    bot_idx = 0
    if len(team_names) > 1 and 'alfphafarm' in team_names[1].lower():
        bot_idx = 1
    elif len(team_names) > 0 and 'alfphafarm' in team_names[0].lower():
        bot_idx = 0
    else:
        bot_idx = 0
    
    opp_idx = 1 - bot_idx
    
    last_step = steps[-1]
    bot_reward = last_step[bot_idx].get('reward', 0)
    opp_reward = last_step[opp_idx].get('reward', 0) if len(last_step) > 1 else 0
    bot_name = team_names[bot_idx] if bot_idx < len(team_names) else f"P{bot_idx}"
    opp_name = team_names[opp_idx] if opp_idx < len(team_names) else f"P{opp_idx}"
    
    # Track prices history: item -> list of prices per step
    prices_history = defaultdict(list)
    
    # Track player transactions: (step, day, hr, player, action_type, item, qty, price, total_cost_or_rev)
    transactions = []
    
    # Track units, buildings, tiles, shed inventory per step
    bot_money_history = []
    opp_money_history = []
    
    bot_shed_history = []
    opp_shed_history = []
    
    # Sales breakdown: player -> item -> list of (step, qty, unit_price, revenue)
    sales_by_player = {0: defaultdict(list), 1: defaultdict(list)}
    buys_by_player = {0: defaultdict(list), 1: defaultdict(list)}
    
    # Town shop tracking: step -> unlocked_shops, active orders, etc.
    town_history = []
    
    for step_idx, step in enumerate(steps):
        s_obs = step[0].get('observation', {}) # market & town are global/shared
        day = s_obs.get('day', step_idx // 24)
        hour = s_obs.get('hour', step_idx % 24)
        
        m_info = s_obs.get('market', {})
        current_prices = m_info.get('prices', {})
        for comm in COMMODITIES:
            prices_history[comm].append(current_prices.get(comm, 0))
            
        town_info = s_obs.get('town', {})
        town_history.append((step_idx, day, hour, town_info))
        
        # Farm observations
        farms = s_obs.get('farms', [])
        if len(farms) > bot_idx:
            bot_money_history.append((step_idx, day, hour, farms[bot_idx].get('money', 0)))
        if len(farms) > opp_idx:
            opp_money_history.append((step_idx, day, hour, farms[opp_idx].get('money', 0)))
            
        # Parse actions for both players
        for p in [0, 1]:
            if p >= len(step):
                continue
            p_step = step[p]
            act = p_step.get('action', {})
            if not isinstance(act, dict):
                continue
            
            market_act = act.get('market', [])
            for m in market_act:
                if not isinstance(m, list) or len(m) == 0:
                    continue
                op = m[0]
                if op == 'SELL':
                    item = m[1] if len(m) > 1 else 'UNKNOWN'
                    qty = m[2] if len(m) > 2 else 1
                    price = current_prices.get(item, 0)
                    rev = qty * price
                    transactions.append((step_idx, day, hour, p, 'SELL', item, qty, price, rev))
                    sales_by_player[p][item].append((step_idx, day, hour, qty, price, rev))
                elif op in ['BUY', 'BUY_SEED', 'BUY_ANIMAL', 'BUY_PRODUCT']:
                    item = m[1] if len(m) > 1 else 'UNKNOWN'
                    qty = m[2] if len(m) > 2 else 1
                    price = current_prices.get(item, 0)
                    cost = qty * price
                    transactions.append((step_idx, day, hour, p, op, item, qty, price, cost))
                    buys_by_player[p][item].append((step_idx, day, hour, qty, price, cost))
                elif op == 'BUY_LAND':
                    transactions.append((step_idx, day, hour, p, 'BUY_LAND', '', 1, 0, 0))
                elif op == 'HIRE':
                    transactions.append((step_idx, day, hour, p, 'HIRE', '', m[1] if len(m) > 1 else 1, 0, 0))

    # Inspect end-state inventory in shed & tiles
    last_obs = steps[-1][0].get('observation', {})
    last_farms = last_obs.get('farms', [])
    
    bot_deadweight = defaultdict(int)
    opp_deadweight = defaultdict(int)
    bot_tiles_status = defaultdict(int)
    opp_tiles_status = defaultdict(int)
    
    if len(last_farms) > bot_idx:
        bfarm = last_farms[bot_idx]
        tiles = bfarm.get('tiles', [])
        for r in tiles:
            for tile in r:
                st = tile.get('state', '')
                bot_tiles_status[st] += 1
                crop = tile.get('crop')
                if crop:
                    bot_deadweight[f"CROP_{crop}"] += 1
                animal = tile.get('animal')
                if animal:
                    bot_deadweight[f"ANIMAL_{animal}"] += 1
                    
    if len(last_farms) > opp_idx:
        ofarm = last_farms[opp_idx]
        tiles = ofarm.get('tiles', [])
        for r in tiles:
            for tile in r:
                st = tile.get('state', '')
                opp_tiles_status[st] += 1
                crop = tile.get('crop')
                if crop:
                    opp_deadweight[f"CROP_{crop}"] += 1
                animal = tile.get('animal')
                if animal:
                    opp_deadweight[f"ANIMAL_{animal}"] += 1

    return {
        'ep_id': ep_id,
        'bot_idx': bot_idx,
        'opp_idx': opp_idx,
        'bot_name': bot_name,
        'opp_name': opp_name,
        'bot_reward': bot_reward,
        'opp_reward': opp_reward,
        'steps': num_steps,
        'prices_history': prices_history,
        'sales_by_player': sales_by_player,
        'buys_by_player': buys_by_player,
        'transactions': transactions,
        'bot_money_history': bot_money_history,
        'opp_money_history': opp_money_history,
        'town_history': town_history,
        'bot_deadweight': bot_deadweight,
        'opp_deadweight': opp_deadweight,
        'bot_tiles_status': bot_tiles_status,
        'opp_tiles_status': opp_tiles_status
    }

print("Forensic module compiled.")
