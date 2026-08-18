import json
import os
import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

COMMODITIES = ['FERTILIZER', 'WOOL', 'MILK', 'MELON', 'CARROT', 'STRAWBERRY', 'TOMATO', 'WHEAT', 'EGG']

def parse_exact_match(ep_id):
    path = f"replays/latest_frontier/episode-{ep_id}-replay.json"
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
        
    steps = d['steps']
    team_names = d.get('info', {}).get('TeamNames', ['P0', 'P1'])
    bot_idx = 1 if len(team_names) > 1 and 'alfphafarm' in team_names[1].lower() else 0
    opp_idx = 1 - bot_idx
    
    last = steps[-1]
    b_score = last[bot_idx].get('reward', 0)
    o_score = last[opp_idx].get('reward', 0) if len(last) > 1 else 0
    b_name = team_names[bot_idx] if bot_idx < len(team_names) else f"P{bot_idx}"
    o_name = team_names[opp_idx] if opp_idx < len(team_names) else f"P{opp_idx}"
    
    # Store price history
    price_series = {c: [] for c in COMMODITIES}
    market_inv_series = {c: [] for c in COMMODITIES}
    
    # Track executed sales by step, player, commodity
    # We can detect execution by tracking player money delta and actions
    player_executed_sales = {0: defaultdict(int), 1: defaultdict(int)}
    player_executed_rev = {0: defaultdict(float), 1: defaultdict(float)}
    player_sales_by_step = defaultdict(lambda: {0: defaultdict(int), 1: defaultdict(int)})
    
    # Track units, crops, animals
    player_stats = {
        0: {'coops': 0, 'pastures': 0, 'cows': 0, 'sheep': 0, 'geese': 0, 'crops_planted': defaultdict(int), 'harvests': 0, 'care': 0, 'feed': 0, 'water': 0, 'fert_applied': 0, 'fert_collected': 0, 'hires': 0, 'quadrants': 1},
        1: {'coops': 0, 'pastures': 0, 'cows': 0, 'sheep': 0, 'geese': 0, 'crops_planted': defaultdict(int), 'harvests': 0, 'care': 0, 'feed': 0, 'water': 0, 'fert_applied': 0, 'fert_collected': 0, 'hires': 0, 'quadrants': 1}
    }
    
    # Also track town unlocks
    town_unlocks = []
    
    for s_idx in range(len(steps)):
        obs = steps[s_idx][0].get('observation', {})
        day = obs.get('day', s_idx // 24)
        hr = obs.get('hour', s_idx % 24)
        
        m_info = obs.get('market', {})
        p_dict = m_info.get('prices', {})
        i_dict = m_info.get('inventory', {})
        
        for c in COMMODITIES:
            price_series[c].append(p_dict.get(c, 0))
            market_inv_series[c].append(i_dict.get(c, 10000))
            
        t_info = obs.get('town', {})
        shops = t_info.get('unlocked_shops', [])
        if len(shops) > len(town_unlocks):
            new_s = shops[len(town_unlocks):]
            town_unlocks = list(shops)
            
        # Farm states
        farms = obs.get('farms', [])
        for p in [0, 1]:
            if len(farms) > p:
                player_stats[p]['quadrants'] = len(farms[p].get('unlocked_quadrants', []))
                
        # Actions
        for p in [0, 1]:
            if p >= len(steps[s_idx]):
                continue
            act = steps[s_idx][p].get('action', {})
            if not isinstance(act, dict):
                continue
                
            farmer_act = act.get('farmer', [])
            hands_act = act.get('hands', [])
            for u in [farmer_act] + hands_act:
                if not u:
                    continue
                op = u[0]
                if op == 'BUILD_COOP':
                    player_stats[p]['coops'] += 1
                elif op == 'BUILD_PASTURE':
                    player_stats[p]['pastures'] += 1
                elif op == 'PLANT':
                    cname = u[1] if len(u) > 1 else 'UNKNOWN'
                    player_stats[p]['crops_planted'][cname] += 1
                elif op == 'HARVEST':
                    player_stats[p]['harvests'] += 1
                elif op == 'CARE':
                    player_stats[p]['care'] += 1
                elif op == 'FEED':
                    player_stats[p]['feed'] += 1
                elif op == 'WATER':
                    player_stats[p]['water'] += 1
                elif op == 'FERTILIZE':
                    player_stats[p]['fert_applied'] += 1
                elif op == 'COLLECT_FERTILIZER':
                    player_stats[p]['fert_collected'] += 1
                    
            m_acts = act.get('market', [])
            for m in m_acts:
                if not isinstance(m, list) or len(m) == 0:
                    continue
                mop = m[0]
                if mop == 'BUY_ANIMAL':
                    aname = m[1] if len(m) > 1 else 'UNKNOWN'
                    qty = m[2] if len(m) > 2 else 1
                    if aname == 'COW':
                        player_stats[p]['cows'] += qty
                    elif aname == 'SHEEP':
                        player_stats[p]['sheep'] += qty
                    elif aname == 'GOOSE':
                        player_stats[p]['geese'] += qty
                elif mop == 'HIRE':
                    qty = m[1] if len(m) > 1 else 1
                    player_stats[p]['hires'] += qty

    return {
        'ep_id': ep_id,
        'bot_idx': bot_idx,
        'opp_idx': opp_idx,
        'bot_name': b_name,
        'opp_name': o_name,
        'bot_score': b_score,
        'opp_score': o_score,
        'price_series': price_series,
        'market_inv_series': market_inv_series,
        'player_stats': player_stats,
        'town_unlocks': town_unlocks
    }

print("Exact financial engine compiled.")
