import json
import os
import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def detailed_audit(ep_id):
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
    
    # Track stats for both players
    stats = {
        bot_idx: {
            'role': 'BOT (alfphafarm)',
            'name': b_name,
            'score': b_score,
            'coops': 0,
            'pastures': 0,
            'cows_bought': 0,
            'sheep_bought': 0,
            'geese_bought': 0,
            'quadrants_unlocked': 0,
            'crops_planted': defaultdict(int),
            'crops_harvested': defaultdict(int),
            'fertilizer_collected': 0,
            'fertilizer_applied': 0,
            'care_actions': 0,
            'feed_actions': 0,
            'water_actions': 0,
            'sales_qty': defaultdict(int),
            'sales_rev': defaultdict(float),
            'sales_prices': defaultdict(list),
            'buys_qty': defaultdict(int),
            'buys_cost': defaultdict(float),
            'hires_count': 0,
            'hires_cost': 0,
            'cash_trajectory': [],
            'idle_unit_turns': 0,
            'total_unit_turns': 0
        },
        opp_idx: {
            'role': 'OPPONENT',
            'name': o_name,
            'score': o_score,
            'coops': 0,
            'pastures': 0,
            'cows_bought': 0,
            'sheep_bought': 0,
            'geese_bought': 0,
            'quadrants_unlocked': 0,
            'crops_planted': defaultdict(int),
            'crops_harvested': defaultdict(int),
            'fertilizer_collected': 0,
            'fertilizer_applied': 0,
            'care_actions': 0,
            'feed_actions': 0,
            'water_actions': 0,
            'sales_qty': defaultdict(int),
            'sales_rev': defaultdict(float),
            'sales_prices': defaultdict(list),
            'buys_qty': defaultdict(int),
            'buys_cost': defaultdict(float),
            'hires_count': 0,
            'hires_cost': 0,
            'cash_trajectory': [],
            'idle_unit_turns': 0,
            'total_unit_turns': 0
        }
    }
    
    for s_idx, step in enumerate(steps):
        obs = step[0].get('observation', {})
        day = obs.get('day', s_idx // 24)
        hr = obs.get('hour', s_idx % 24)
        m_prices = obs.get('market', {}).get('prices', {})
        farms = obs.get('farms', [])
        
        for p in [0, 1]:
            if p >= len(step) or p not in stats:
                continue
            p_data = stats[p]
            p_step = step[p]
            act = p_step.get('action', {})
            
            if len(farms) > p:
                money = farms[p].get('money', 0)
                p_data['cash_trajectory'].append(money)
                p_data['quadrants_unlocked'] = len(farms[p].get('unlocked_quadrants', []))
            
            if isinstance(act, dict):
                farmer_act = act.get('farmer', [])
                hands_act = act.get('hands', [])
                all_units = [farmer_act] + hands_act
                
                for u in all_units:
                    if not u:
                        continue
                    p_data['total_unit_turns'] += 1
                    op = u[0]
                    if op == 'PASS':
                        p_data['idle_unit_turns'] += 1
                    elif op == 'BUILD_COOP':
                        p_data['coops'] += 1
                    elif op == 'BUILD_PASTURE':
                        p_data['pastures'] += 1
                    elif op == 'PLANT':
                        cname = u[1] if len(u) > 1 else 'UNKNOWN'
                        p_data['crops_planted'][cname] += 1
                    elif op == 'HARVEST':
                        p_data['crops_harvested']['TOTAL'] += 1
                    elif op == 'COLLECT_FERTILIZER':
                        p_data['fertilizer_collected'] += 1
                    elif op == 'FERTILIZE':
                        p_data['fertilizer_applied'] += 1
                    elif op == 'CARE':
                        p_data['care_actions'] += 1
                    elif op == 'FEED':
                        p_data['feed_actions'] += 1
                    elif op == 'WATER':
                        p_data['water_actions'] += 1
                        
                m_acts = act.get('market', [])
                for m in m_acts:
                    if not isinstance(m, list) or len(m) == 0:
                        continue
                    mop = m[0]
                    if mop == 'SELL':
                        item = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        pr = m_prices.get(item, 0)
                        p_data['sales_qty'][item] += qty
                        p_data['sales_rev'][item] += qty * pr
                        p_data['sales_prices'][item].append((s_idx, qty, pr))
                    elif mop == 'BUY_ANIMAL':
                        aname = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        if aname == 'COW':
                            p_data['cows_bought'] += qty
                        elif aname == 'SHEEP':
                            p_data['sheep_bought'] += qty
                        elif aname == 'GOOSE':
                            p_data['geese_bought'] += qty
                    elif mop == 'BUY_SEED':
                        cname = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        p_data['buys_qty'][f"SEED_{cname}"] += qty
                        p_data['buys_cost'][f"SEED_{cname}"] += qty * m_prices.get(cname, 0)
                    elif mop == 'BUY_PRODUCT':
                        item = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        p_data['buys_qty'][item] += qty
                        p_data['buys_cost'][item] += qty * m_prices.get(item, 0)
                    elif mop == 'HIRE':
                        qty = m[1] if len(m) > 1 else 1
                        p_data['hires_count'] += qty

    return stats

def run_comparison():
    episodes = [93981122, 93979393, 93977628, 93978496, 93980262, 93982072]
    for ep in episodes:
        st = detailed_audit(ep)
        print("=" * 110)
        print(f"EPISODE {ep} FORENSIC AUDIT")
        print("=" * 110)
        
        for p_idx in sorted(st.keys()):
            p = st[p_idx]
            print(f"\n[{p['role']}] {p['name']} | Final Score: ${p['score']:,.1f} | Quadrants: {p['quadrants_unlocked']}/4")
            print(f"  Infrastructure: Coops={p['coops']}, Pastures={p['pastures']} | Animals: Cows={p['cows_bought']}, Sheep={p['sheep_bought']}, Geese={p['geese_bought']}")
            print(f"  Care/Fertilizer: Care Actions={p['care_actions']}, Feed={p['feed_actions']}, Fertilizer Collected={p['fertilizer_collected']}, Applied={p['fertilizer_applied']}")
            print(f"  Labor: Total Unit-Turns={p['total_unit_turns']}, Idle Passes={p['idle_unit_turns']} ({p['idle_unit_turns']/max(1, p['total_unit_turns'])*100:.1f}%), Hires={p['hires_count']}")
            print(f"  Crops Planted: {dict(p['crops_planted'])}")
            print("  Sales Breakdown (Qty, Total Rev, Weighted Avg Price):")
            for item in sorted(p['sales_qty'].keys()):
                qty = p['sales_qty'][item]
                rev = p['sales_rev'][item]
                avg_p = rev / max(1, qty)
                print(f"    - {item:<12}: {qty:4d} sold | Rev: ${rev:8,.1f} | Avg Realized Price: ${avg_p:6.2f}")
            print("  Major Purchases:")
            for item in sorted(p['buys_qty'].keys()):
                print(f"    - {item:<12}: {p['buys_qty'][item]:4d} bought | Cost: ${p['buys_cost'][item]:8,.1f}")

if __name__ == '__main__':
    run_comparison()
