import glob
import json
import os
import sys

# Ensure stdout handles utf-8 safely
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    files = glob.glob('replays/latest_frontier/episode-*-replay.json')
    print(f"Found {len(files)} replays in replays/latest_frontier/")

    rows = []
    for f in files:
        try:
            ep_id = int(os.path.basename(f).split('-')[1])
            with open(f, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
            steps = d['steps']
            last = steps[-1]
            r0 = last[0].get('reward', 0)
            r1 = last[1].get('reward', 0)
            names = d.get('info', {}).get('TeamNames', ['P0', 'P1'])
            
            # Determine which player is alfphafarm
            p0_name = names[0] if len(names) > 0 else 'P0'
            p1_name = names[1] if len(names) > 1 else 'P1'
            
            bot_idx = None
            if 'alfphafarm' in p0_name.lower():
                bot_idx = 0
            elif 'alfphafarm' in p1_name.lower():
                bot_idx = 1
            else:
                # validation or test match
                bot_idx = 0
                
            bot_reward = r0 if bot_idx == 0 else r1
            opp_reward = r1 if bot_idx == 0 else r0
            opp_name = p1_name if bot_idx == 0 else p0_name
            
            rows.append({
                'id': ep_id,
                'bot_idx': bot_idx,
                'bot_name': names[bot_idx] if bot_idx is not None and bot_idx < len(names) else 'alfphafarm',
                'opp_name': opp_name,
                'bot_reward': bot_reward,
                'opp_reward': opp_reward,
                'steps': len(steps),
                'file': f
            })
        except Exception as e:
            print(f"Err {f}: {e}")

    rows.sort(key=lambda x: (x['bot_reward'] if x['bot_reward'] is not None else -999999), reverse=True)
    print(f"\n{'Episode':<10} | {'Bot Idx':<7} | {'Bot Score':<12} | {'Opponent Score':<14} | {'Winner':<8} | {'Opponent Name':<30}")
    print("-" * 95)
    for r in rows:
        b_s = f"${r['bot_reward']:,.1f}" if r['bot_reward'] is not None else 'None'
        o_s = f"${r['opp_reward']:,.1f}" if r['opp_reward'] is not None else 'None'
        win = "BOT" if (r['bot_reward'] or 0) > (r['opp_reward'] or 0) else ("OPP" if (r['bot_reward'] or 0) < (r['opp_reward'] or 0) else "TIE")
        print(f"{r['id']:<10} | {r['bot_idx'] if r['bot_idx'] is not None else '?':<7} | {b_s:<12} | {o_s:<14} | {win:<8} | {r['opp_name'][:30]:<30}")

if __name__ == '__main__':
    main()
