import json
import os
import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

KEY_COMMODITIES = ['FERTILIZER', 'WOOL', 'MILK', 'MELON']
ALL_COMMODITIES = ['FERTILIZER', 'WOOL', 'MILK', 'MELON', 'CARROT', 'STRAWBERRY', 'TOMATO', 'WHEAT', 'EGG']

def analyze_full_details(ep_id):
    filepath = f"replays/latest_frontier/episode-{ep_id}-replay.json"
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    steps = data.get('steps', [])
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
    bot_score = last_step[bot_idx].get('reward', 0)
    opp_score = last_step[opp_idx].get('reward', 0) if len(last_step) > 1 else 0
    bot_name = team_names[bot_idx] if bot_idx < len(team_names) else f"P{bot_idx}"
    opp_name = team_names[opp_idx] if opp_idx < len(team_names) else f"P{opp_idx}"
    
    prices = {c: [] for c in ALL_COMMODITIES}
    market_inv = {c: [] for c in ALL_COMMODITIES}
    
    bot_money = []
    opp_money = []
    
    bot_sales = defaultdict(lambda: {'qty': 0, 'rev': 0.0, 'orders': []})
    opp_sales = defaultdict(lambda: {'qty': 0, 'rev': 0.0, 'orders': []})
    
    bot_buys = defaultdict(lambda: {'qty': 0, 'cost': 0.0, 'orders': []})
    opp_buys = defaultdict(lambda: {'qty': 0, 'cost': 0.0, 'orders': []})
    
    # Track inventory in shed and field at end
    # Track hourly dumps and price changes
    hourly_dumps = [] # (step, day, hr, player, item, qty, price_before, price_after, rev)
    
    town_shop_timeline = []
    current_unlocked = []
    
    for s_idx in range(len(steps)):
        step = steps[s_idx]
        obs = step[0].get('observation', {})
        day = obs.get('day', s_idx // 24)
        hr = obs.get('hour', s_idx % 24)
        
        m_info = obs.get('market', {})
        p_dict = m_info.get('prices', {})
        i_dict = m_info.get('inventory', {})
        
        for c in ALL_COMMODITIES:
            prices[c].append(p_dict.get(c, 0))
            market_inv[c].append(i_dict.get(c, 10000))
            
        t_info = obs.get('town', {})
        unlocked = t_info.get('unlocked_shops', [])
        if len(unlocked) > len(current_unlocked):
            new_s = unlocked[len(current_unlocked):]
            current_unlocked = list(unlocked)
            town_shop_timeline.append((s_idx, day, hr, new_s, list(unlocked)))
            
        farms = obs.get('farms', [])
        if len(farms) > bot_idx:
            bot_money.append(farms[bot_idx].get('money', 0))
        if len(farms) > opp_idx:
            opp_money.append(farms[opp_idx].get('money', 0))
            
        # Check sales this step
        for p in [0, 1]:
            if p >= len(step):
                continue
            act = step[p].get('action', {})
            if not isinstance(act, dict):
                continue
            m_orders = act.get('market', [])
            for m in m_orders:
                if not isinstance(m, list) or len(m) == 0:
                    continue
                op = m[0]
                if op == 'SELL':
                    item = m[1] if len(m) > 1 else 'UNKNOWN'
                    qty = m[2] if len(m) > 2 else 1
                    pr = p_dict.get(item, 0)
                    rev = qty * pr
                    if p == bot_idx:
                        bot_sales[item]['qty'] += qty
                        bot_sales[item]['rev'] += rev
                        bot_sales[item]['orders'].append((s_idx, day, hr, qty, pr, rev))
                    else:
                        opp_sales[item]['qty'] += qty
                        opp_sales[item]['rev'] += rev
                        opp_sales[item]['orders'].append((s_idx, day, hr, qty, pr, rev))
                    hourly_dumps.append((s_idx, day, hr, p, item, qty, pr, rev))
                elif op in ['BUY', 'BUY_SEED', 'BUY_ANIMAL', 'BUY_PRODUCT']:
                    item = m[1] if len(m) > 1 else 'UNKNOWN'
                    qty = m[2] if len(m) > 2 else 1
                    pr = p_dict.get(item, 0)
                    cost = qty * pr
                    if p == bot_idx:
                        bot_buys[item]['qty'] += qty
                        bot_buys[item]['cost'] += cost
                        bot_buys[item]['orders'].append((s_idx, day, hr, qty, pr, cost))
                    else:
                        opp_buys[item]['qty'] += qty
                        opp_buys[item]['cost'] += cost
                        opp_buys[item]['orders'].append((s_idx, day, hr, qty, pr, cost))

    # Detect largest price drops
    price_crashes = []
    for c in ALL_COMMODITIES:
        pr_arr = prices[c]
        for s in range(1, len(pr_arr)):
            drop = pr_arr[s-1] - pr_arr[s]
            pct_drop = (drop / pr_arr[s-1]) if pr_arr[s-1] > 0 else 0
            if drop >= 20 or pct_drop >= 0.25:
                # Find who sold at step s-1
                sellers_at_s = [d for d in hourly_dumps if d[0] == s-1 and d[4] == c]
                price_crashes.append({
                    'commodity': c,
                    'step': s,
                    'day': s // 24,
                    'hour': s % 24,
                    'price_before': pr_arr[s-1],
                    'price_after': pr_arr[s],
                    'drop': drop,
                    'pct_drop': pct_drop,
                    'sellers': sellers_at_s
                })

    return {
        'ep_id': ep_id,
        'bot_idx': bot_idx,
        'opp_idx': opp_idx,
        'bot_name': bot_name,
        'opp_name': opp_name,
        'bot_score': bot_score,
        'opp_score': opp_score,
        'prices': prices,
        'market_inv': market_inv,
        'bot_money': bot_money,
        'opp_money': opp_money,
        'bot_sales': bot_sales,
        'opp_sales': opp_sales,
        'bot_buys': bot_buys,
        'opp_buys': opp_buys,
        'price_crashes': price_crashes,
        'town_shop_timeline': town_shop_timeline,
        'hourly_dumps': hourly_dumps
    }

def print_match_deep_dive(ep_id):
    res = analyze_full_details(ep_id)
    print("=" * 100)
    print(f"MATCH FORENSIC DEEP DIVE: Episode {res['ep_id']} | Bot: {res['bot_name']} (${res['bot_score']:,.1f}) vs Opp: {res['opp_name']} (${res['opp_score']:,.1f})")
    print("=" * 100)
    
    print("\n1. COMMODITY PRICE SUMMARY (Step 0 -> Step 719):")
    print(f"{'Commodity':<12} | {'Start':<7} | {'Min':<7} | {'Max':<7} | {'Final':<7} | {'Mean':<8} | {'Bot Sold (Qty/Rev)':<25} | {'Opp Sold (Qty/Rev)':<25}")
    print("-" * 100)
    for c in KEY_COMMODITIES + [x for x in ALL_COMMODITIES if x not in KEY_COMMODITIES]:
        pr = res['prices'][c]
        b_s = res['bot_sales'][c]
        o_s = res['opp_sales'][c]
        b_str = f"{b_s['qty']} (${b_s['rev']:,.0f})" if b_s['qty'] > 0 else "-"
        o_str = f"{o_s['qty']} (${o_s['rev']:,.0f})" if o_s['qty'] > 0 else "-"
        print(f"{c:<12} | ${pr[0]:<6} | ${min(pr):<6} | ${max(pr):<6} | ${pr[-1]:<6} | ${np.mean(pr):<7.1f} | {b_str:<25} | {o_str:<25}")
        
    print("\n2. MAJOR PRICE CRASHES DETECTED (>25% or >$20 drop):")
    if not res['price_crashes']:
        print("  None detected.")
    else:
        for cr in sorted(res['price_crashes'], key=lambda x: x['drop'], reverse=True)[:10]:
            seller_summary = ", ".join([f"P{s[3]} sold {s[5]} {s[4]} at ${s[6]}" for s in cr['sellers']])
            print(f"  Step {cr['step']:3d} (D{cr['day']:02d}:H{cr['hour']:02d}) {cr['commodity']:<11}: ${cr['price_before']} -> ${cr['price_after']} (Drop: -${cr['drop']} / {cr['pct_drop']*100:.1f}%) | Trigger: {seller_summary if seller_summary else 'Market decay/other'}")
            
    print("\n3. TOWN SHOP UNLOCKS & PRICE IMPACT:")
    for (s_idx, day, hr, new_s, all_s) in res['town_shop_timeline']:
        print(f"  Step {s_idx:3d} (Day {day:02d}, Hr {hr:02d}): Unlocked {new_s} | Active Total: {len(all_s)}")

if __name__ == '__main__':
    episodes_to_examine = [93979393, 93977628, 93981122, 93978496, 93980262, 93982072]
    for ep in episodes_to_examine:
        print_match_deep_dive(ep)
        print("\n")
