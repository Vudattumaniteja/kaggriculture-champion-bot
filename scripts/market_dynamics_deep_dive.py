import glob
import json
import os
import sys
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

COMMODITIES = ['FERTILIZER', 'WOOL', 'MILK', 'MELON', 'CARROT', 'STRAWBERRY', 'TOMATO', 'WHEAT', 'EGG']

# Shop consumption mappings based on Kaggriculture rules
SHOP_CONSUMPTION = {
    'YARN_STORE': ['WOOL'],
    'ICE_CREAM_SHOP': ['MILK', 'STRAWBERRY'],
    'BAKERY': ['WHEAT', 'EGG'],
    'PIZZA_SHOP': ['TOMATO', 'WHEAT'],
    'SMOOTHIE_SHOP': ['MELON', 'STRAWBERRY', 'CARROT'],
    'PET_CAFE': ['MILK', 'EGG']
}

def analyze_match(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    ep_id = int(os.path.basename(filepath).split('-')[1])
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
    bot_score = last_step[bot_idx].get('reward', 0)
    opp_score = last_step[opp_idx].get('reward', 0) if len(last_step) > 1 else 0
    bot_name = team_names[bot_idx] if bot_idx < len(team_names) else f"P{bot_idx}"
    opp_name = team_names[opp_idx] if opp_idx < len(team_names) else f"P{opp_idx}"

    prices_by_step = {c: [] for c in COMMODITIES}
    market_inv_by_step = {c: [] for c in COMMODITIES}
    
    # Track hourly sales by player: step -> player -> item -> qty
    sales_timeline = defaultdict(lambda: {0: defaultdict(int), 1: defaultdict(int)})
    buys_timeline = defaultdict(lambda: {0: defaultdict(int), 1: defaultdict(int)})
    
    # Cumulative sales and revenue
    bot_sales = defaultdict(int)
    bot_sales_rev = defaultdict(float)
    opp_sales = defaultdict(int)
    opp_sales_rev = defaultdict(float)
    
    # Detailed sell orders: (step, day, hr, player, item, qty, price, rev)
    all_sell_orders = []
    
    town_shops_unlocked = []
    town_shop_events = []
    
    # Financial trajectory
    bot_cash_trajectory = []
    opp_cash_trajectory = []
    
    for step_idx, step in enumerate(steps):
        obs = step[0].get('observation', {})
        day = obs.get('day', step_idx // 24)
        hr = obs.get('hour', step_idx % 24)
        
        m_info = obs.get('market', {})
        p_dict = m_info.get('prices', {})
        i_dict = m_info.get('inventory', {})
        
        for c in COMMODITIES:
            prices_by_step[c].append(p_dict.get(c, 0))
            market_inv_by_step[c].append(i_dict.get(c, 10000))
            
        t_info = obs.get('town', {})
        current_shops = t_info.get('unlocked_shops', [])
        if len(current_shops) > len(town_shops_unlocked):
            new_s = current_shops[len(town_shops_unlocked):]
            town_shops_unlocked = list(current_shops)
            town_shop_events.append((step_idx, day, hr, new_s, list(current_shops)))
            
        farms = obs.get('farms', [])
        if len(farms) > bot_idx:
            bot_cash_trajectory.append(farms[bot_idx].get('money', 0))
        if len(farms) > opp_idx:
            opp_cash_trajectory.append(farms[opp_idx].get('money', 0))
            
        # Check actions
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
                    sales_timeline[step_idx][p][item] += qty
                    all_sell_orders.append((step_idx, day, hr, p, item, qty, pr, rev))
                    if p == bot_idx:
                        bot_sales[item] += qty
                        bot_sales_rev[item] += rev
                    else:
                        opp_sales[item] += qty
                        opp_sales_rev[item] += rev
                elif op in ['BUY', 'BUY_SEED', 'BUY_ANIMAL', 'BUY_PRODUCT']:
                    item = m[1] if len(m) > 1 else 'UNKNOWN'
                    qty = m[2] if len(m) > 2 else 1
                    buys_timeline[step_idx][p][item] += qty

    # Calculate price statistics & collapse points
    price_stats = {}
    for c in COMMODITIES:
        pr_arr = prices_by_step[c]
        if not pr_arr:
            continue
        start_p = pr_arr[0]
        min_p = min(pr_arr)
        max_p = max(pr_arr)
        final_p = pr_arr[-1]
        mean_p = float(np.mean(pr_arr))
        
        # Find first collapse turn: price drops below 50% of start_p or 30% of start_p
        collapse_turn_50 = None
        collapse_turn_min = None
        for s, p in enumerate(pr_arr):
            if collapse_turn_50 is None and p <= start_p * 0.5:
                collapse_turn_50 = s
            if p == min_p and collapse_turn_min is None:
                collapse_turn_min = s
                
        price_stats[c] = {
            'start': start_p,
            'min': min_p,
            'max': max_p,
            'final': final_p,
            'mean': mean_p,
            'collapse_turn_50': collapse_turn_50,
            'collapse_turn_min': collapse_turn_min
        }

    # Quantify loss from selling at troughs vs counterfactual selling at baseline/mean
    bot_loss_vs_mean = 0.0
    bot_loss_vs_start = 0.0
    bot_dump_loss_by_item = defaultdict(lambda: {'actual_rev': 0.0, 'cf_mean_rev': 0.0, 'cf_start_rev': 0.0, 'loss_vs_mean': 0.0, 'loss_vs_start': 0.0, 'qty_sold': 0})
    
    for (step_idx, day, hr, p, item, qty, pr, rev) in all_sell_orders:
        if p == bot_idx:
            mean_pr = price_stats.get(item, {}).get('mean', pr)
            start_pr = price_stats.get(item, {}).get('start', pr)
            cf_mean = qty * mean_pr
            cf_start = qty * start_pr
            
            bot_dump_loss_by_item[item]['actual_rev'] += rev
            bot_dump_loss_by_item[item]['cf_mean_rev'] += cf_mean
            bot_dump_loss_by_item[item]['cf_start_rev'] += cf_start
            bot_dump_loss_by_item[item]['loss_vs_mean'] += max(0.0, cf_mean - rev)
            bot_dump_loss_by_item[item]['loss_vs_start'] += max(0.0, cf_start - rev)
            bot_dump_loss_by_item[item]['qty_sold'] += qty
            
            bot_loss_vs_mean += max(0.0, cf_mean - rev)
            bot_loss_vs_start += max(0.0, cf_start - rev)

    return {
        'ep_id': ep_id,
        'bot_idx': bot_idx,
        'opp_idx': opp_idx,
        'bot_name': bot_name,
        'opp_name': opp_name,
        'bot_score': bot_score,
        'opp_score': opp_score,
        'prices_by_step': prices_by_step,
        'price_stats': price_stats,
        'bot_sales': bot_sales,
        'bot_sales_rev': bot_sales_rev,
        'opp_sales': opp_sales,
        'opp_sales_rev': opp_sales_rev,
        'bot_dump_loss_by_item': bot_dump_loss_by_item,
        'bot_loss_vs_mean': bot_loss_vs_mean,
        'bot_loss_vs_start': bot_loss_vs_start,
        'town_shop_events': town_shop_events,
        'all_sell_orders': all_sell_orders,
        'bot_cash_trajectory': bot_cash_trajectory,
        'opp_cash_trajectory': opp_cash_trajectory
    }

def print_forensic_comparison(ep_list):
    results = [analyze_match(f"replays/latest_frontier/episode-{ep}-replay.json") for ep in ep_list]
    
    print("\n" + "=" * 120)
    print(f"{'EPISODE FORENSIC COMPARISON SUMMARY':^120}")
    print("=" * 120)
    
    header = f"{'Episode':<10} | {'Bot Score':<10} | {'Opp Score':<10} | {'Opponent Name':<20} | {'Bot Loss vs Mean':<18} | {'Bot Loss vs Base':<18} | {'Top Sold Item (Qty/Rev)':<25}"
    print(header)
    print("-" * 120)
    
    for r in results:
        top_item = "None"
        if r['bot_sales']:
            sorted_items = sorted(r['bot_sales'].items(), key=lambda x: r['bot_sales_rev'][x[0]], reverse=True)
            t_name = sorted_items[0][0]
            top_item = f"{t_name} ({r['bot_sales'][t_name]}: ${r['bot_sales_rev'][t_name]:,.0f})"
            
        print(f"{r['ep_id']:<10} | ${r['bot_score']:<9,.0f} | ${r['opp_score']:<9,.0f} | {r['opp_name'][:20]:<20} | ${r['bot_loss_vs_mean']:<17,.1f} | ${r['bot_loss_vs_start']:<17,.1f} | {top_item:<25}")

if __name__ == '__main__':
    # Analyze key episodes across high, mid, low cohorts
    test_eps = [93979393, 93977628, 93976714, 93982072, 93981181, 93975835, 93980262, 93978496, 93981122, 93982276, 93953896]
    print_forensic_comparison(test_eps)
