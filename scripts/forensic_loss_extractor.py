import json
import os
import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EPISODES = [
    93979393, # $87.5k win
    93977628, # $87.4k win
    93976714, # $79.7k match
    93982072, # $64.9k match
    93981181, # $61.7k win
    93975835, # $39.3k win
    93980262, # $34.8k match
    93978496, # $32.3k match
    93981122, # $12.2k collapse vs Abracadabra $133.4k
    93982276, # $16.2k collapse
    93953896  # $9.5k collapse
]

def full_forensic_extract():
    all_data = {}
    
    for ep_id in EPISODES:
        filepath = f"replays/latest_frontier/episode-{ep_id}-replay.json"
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
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
        
        # Track hourly spot prices
        prices_720 = {c: [] for c in ['FERTILIZER', 'WOOL', 'MILK', 'MELON', 'WHEAT', 'STRAWBERRY', 'TOMATO', 'CARROT', 'EGG']}
        market_inv_720 = {c: [] for c in prices_720.keys()}
        
        # Track bot and opp sales step by step
        bot_sales_by_item = defaultdict(list) # item -> [(step, day, hr, qty, price, rev)]
        opp_sales_by_item = defaultdict(list)
        
        town_shops_at_step = []
        
        for s in range(len(steps)):
            obs = steps[s][0]['observation']
            p_dict = obs['market']['prices']
            i_dict = obs['market']['inventory']
            for c in prices_720.keys():
                prices_720[c].append(p_dict.get(c, 0))
                market_inv_720[c].append(i_dict.get(c, 10000))
                
            town_shops_at_step.append(list(obs.get('town', {}).get('unlocked_shops', [])))
            
            # Check sales in action
            for p in [0, 1]:
                if p >= len(steps[s]):
                    continue
                act = steps[s][p].get('action', {})
                if not isinstance(act, dict):
                    continue
                m_list = act.get('market', [])
                for m in m_list:
                    if isinstance(m, list) and len(m) > 0 and m[0] == 'SELL':
                        item = m[1] if len(m) > 1 else 'UNKNOWN'
                        qty = m[2] if len(m) > 2 else 1
                        pr = p_dict.get(item, 0)
                        rev = qty * pr
                        if p == bot_idx:
                            bot_sales_by_item[item].append((s, s // 24, s % 24, qty, pr, rev))
                        else:
                            opp_sales_by_item[item].append((s, s // 24, s % 24, qty, pr, rev))
                            
        # Compute exact revenue metrics & losses
        loss_analysis = {}
        for c in prices_720.keys():
            b_orders = bot_sales_by_item[c]
            total_qty = sum(x[3] for x in b_orders)
            actual_rev = sum(x[5] for x in b_orders)
            
            p_arr = prices_720[c]
            mean_p = float(np.mean(p_arr))
            base_p = p_arr[0]
            max_p = max(p_arr)
            min_p = min(p_arr)
            
            cf_mean_rev = total_qty * mean_p
            cf_base_rev = total_qty * base_p
            cf_max_rev = total_qty * max_p
            
            loss_vs_mean = max(0.0, cf_mean_rev - actual_rev)
            loss_vs_base = max(0.0, cf_base_rev - actual_rev)
            
            # Collapse turns detection
            # Find turn where price collapsed permanently or for extended duration
            collapse_events = []
            for s in range(1, len(p_arr)):
                if p_arr[s] < p_arr[s-1] * 0.7: # 30% instant drop
                    collapse_events.append((s, s // 24, s % 24, p_arr[s-1], p_arr[s]))
                    
            loss_analysis[c] = {
                'total_qty': total_qty,
                'actual_rev': actual_rev,
                'avg_realized_price': (actual_rev / total_qty) if total_qty > 0 else 0,
                'base_p': base_p,
                'mean_p': mean_p,
                'max_p': max_p,
                'min_p': min_p,
                'loss_vs_mean': loss_vs_mean,
                'loss_vs_base': loss_vs_base,
                'collapse_events': collapse_events
            }
            
        all_data[ep_id] = {
            'ep_id': ep_id,
            'bot_name': b_name,
            'opp_name': o_name,
            'bot_score': b_score,
            'opp_score': o_score,
            'loss_analysis': loss_analysis,
            'town_shops': town_shops_at_step[-1],
            'prices_720': prices_720,
            'bot_sales_by_item': bot_sales_by_item,
            'opp_sales_by_item': opp_sales_by_item
        }
        
    return all_data

if __name__ == '__main__':
    data = full_forensic_extract()
    print("Forensic data extracted successfully for", len(data), "episodes.")
    for ep, d in data.items():
        print(f"\n--- Episode {ep} (Bot: ${d['bot_score']:,.1f} vs Opp: ${d['opp_score']:,.1f}) ---")
        for c in ['WOOL', 'MILK', 'FERTILIZER', 'MELON']:
            la = d['loss_analysis'][c]
            print(f"  {c:<11}: Sold={la['total_qty']:3d} | Realized Avg=${la['avg_realized_price']:6.2f} (Base=${la['base_p']}, Mean=${la['mean_p']:.1f}, Min=${la['min_p']}, Max=${la['max_p']}) | Loss vs Mean=${la['loss_vs_mean']:8,.1f} | Loss vs Base=${la['loss_vs_base']:8,.1f}")
