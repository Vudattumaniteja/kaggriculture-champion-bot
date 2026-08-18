"""
Competitor Replay Parser and Strategy Forensics Tool for Kaggriculture
Extracts turn-by-turn plot moves, crop choices, labor allocations, and cash curves.
"""

import os
import json
import glob
import pandas as pd
from collections import defaultdict, Counter

def analyze_replay(replay_path):
    with open(replay_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    info = data.get('info', {})
    agents = info.get('TeamNames') or [a.get('Name') for a in info.get('Agents', [])] or ['Player 0', 'Player 1']
    episode_id = info.get('EpisodeId') or data.get('id')
    seed = info.get('seed')
    steps = data.get('steps', [])
    num_steps = len(steps)
    
    print(f"==================================================")
    print(f"ANALYSIS: Episode {episode_id} | Seed: {seed} | Steps: {num_steps}")
    print(f"Match: {agents[0]} (P0) vs {agents[1]} (P1)")
    print(f"==================================================")
    
    # Track statistics per player
    stats = {
        0: {
            "name": agents[0],
            "cash_curve": [],
            "crops_planted": Counter(),
            "crops_harvested": Counter(),
            "market_orders": Counter(),
            "quadrant_unlocks": {},
            "structures_built": Counter(),
            "daily_hands_hired": defaultdict(int),
            "total_labor_cost": 0.0,
            "total_revenue": 0.0,
            "final_reward": 0.0,
            "actions_summary": Counter(),
        },
        1: {
            "name": agents[1],
            "cash_curve": [],
            "crops_planted": Counter(),
            "crops_harvested": Counter(),
            "market_orders": Counter(),
            "quadrant_unlocks": {},
            "structures_built": Counter(),
            "daily_hands_hired": defaultdict(int),
            "total_labor_cost": 0.0,
            "total_revenue": 0.0,
            "final_reward": 0.0,
            "actions_summary": Counter(),
        }
    }
    
    prev_unlocked = {0: set(['NW']), 1: set(['NW'])}
    prev_money = {0: 3000.0, 1: 3000.0}
    
    for t, step in enumerate(steps):
        day = t // 24
        hour = t % 24
        
        for p_idx in [0, 1]:
            p_step = step[p_idx]
            obs = p_step.get('observation', {})
            act = p_step.get('action', {})
            reward = p_step.get('reward', 0.0)
            
            farms = obs.get('farms', [])
            if len(farms) > p_idx:
                farm = farms[p_idx]
                money = farm.get('money', 0.0)
                unlocked = set(farm.get('unlocked_quadrants', []))
                hands = farm.get('hands', [])
                hires_today = farm.get('hires_today', 0)
                
                stats[p_idx]["cash_curve"].append((t, day, hour, money))
                stats[p_idx]["daily_hands_hired"][day] = max(stats[p_idx]["daily_hands_hired"][day], hires_today)
                
                # Check quadrant unlocks
                new_unlocks = unlocked - prev_unlocked[p_idx]
                for quad in new_unlocks:
                    stats[p_idx]["quadrant_unlocks"][quad] = {"turn": t, "day": day, "hour": hour}
                prev_unlocked[p_idx] = unlocked
                
                # Check actions
                if isinstance(act, dict):
                    farmer_act = act.get('farmer', [])
                    if farmer_act:
                        op = farmer_act[0]
                        stats[p_idx]["actions_summary"][op] += 1
                        if op == 'PLANT' and len(farmer_act) > 1:
                            stats[p_idx]["crops_planted"][farmer_act[1]] += 1
                        elif op == 'HARVEST':
                            stats[p_idx]["crops_harvested"]["farmer_harvest"] += 1
                        elif op.startswith('BUILD_'):
                            stats[p_idx]["structures_built"][op] += 1
                    
                    hands_act = act.get('hands', [])
                    for h_act in hands_act:
                        if h_act:
                            op = h_act[0]
                            stats[p_idx]["actions_summary"][f"hand_{op}"] += 1
                            if op == 'PLANT' and len(h_act) > 1:
                                stats[p_idx]["crops_planted"][h_act[1]] += 1
                            elif op == 'HARVEST':
                                stats[p_idx]["crops_harvested"]["hand_harvest"] += 1
                    
                    market_act = act.get('market', [])
                    for m_order in market_act:
                        if m_order:
                            m_op = m_order[0]
                            stats[p_idx]["market_orders"][m_op] += 1
                            if m_op == 'BUY_SEED' and len(m_order) > 1:
                                stats[p_idx]["market_orders"][f"BUY_SEED_{m_order[1]}"] += m_order[2] if len(m_order) > 2 else 1
            
            if t == num_steps - 1:
                stats[p_idx]["final_reward"] = reward
    
    # Print comparison
    for p_idx in [0, 1]:
        p = stats[p_idx]
        print(f"\n--- Player {p_idx}: {p['name']} ---")
        print(f"Final Reward: ${p['final_reward']:,.2f}")
        print(f"Quadrant Unlocks: {p['quadrant_unlocks']}")
        print(f"Crops Planted: {dict(p['crops_planted'])}")
        print(f"Structures Built: {dict(p['structures_built'])}")
        print(f"Top Actions: {p['actions_summary'].most_common(8)}")
        print(f"Market Summary: {dict(p['market_orders'])}")
        
        # Calculate daily hands schedule
        hands_sched = [p['daily_hands_hired'][d] for d in range(30)]
        print(f"Daily Hands Hired (Days 0-29): {hands_sched}")
        print(f"Total Farmhand Hires: {sum(hands_sched)}")
        
        # Peak Cash
        peak_cash = max([c[3] for c in p['cash_curve']]) if p['cash_curve'] else 0
        print(f"Peak Cash: ${peak_cash:,.2f}")
    
    return stats

if __name__ == '__main__':
    replay_files = glob.glob('replays/top_competitors/*.json')
    for rf in sorted(replay_files):
        if not rf.endswith('-0.json') and not rf.endswith('-1.json'):
            analyze_replay(rf)
