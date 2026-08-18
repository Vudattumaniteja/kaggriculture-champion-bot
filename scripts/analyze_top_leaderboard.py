import glob
import json
import os
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    files = glob.glob('replays/**/*.json', recursive=True)
    team_replays = defaultdict(list)
    all_replays = []

    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
            if 'steps' not in d or len(d['steps']) < 2:
                continue
            names = d.get('info', {}).get('TeamNames', ['P0', 'P1'])
            last = d['steps'][-1]
            r0 = float(last[0].get('reward') or 0)
            r1 = float(last[1].get('reward') or 0)
            ep_id = d.get('id') or os.path.basename(f)
            
            team_replays[names[0]].append((ep_id, 0, r0, r1, f))
            team_replays[names[1]].append((ep_id, 1, r1, r0, f))
            all_replays.append({
                'ep_id': ep_id,
                'team0': names[0],
                'team1': names[1],
                'r0': r0,
                'r1': r1,
                'path': f,
                'steps': len(d['steps'])
            })
        except Exception as e:
            pass

    print(f"Total valid replays analyzed: {len(all_replays)}")
    print("\nTop 20 Teams Ranked by Peak Score in Replays:")
    print("-" * 85)
    print(f"{'Team Name':<30} | {'Matches':<8} | {'Max Score':<15} | {'Mean Score':<15}")
    print("-" * 85)
    
    sorted_teams = sorted(team_replays.items(), key=lambda x: max([e[2] for e in x[1]]), reverse=True)
    for name, eps in sorted_teams[:25]:
        scores = [e[2] for e in eps]
        print(f"{name:<30} | {len(eps):<8} | ${max(scores):>13,.1f} | ${sum(scores)/len(scores):>13,.1f}")

    print("\nTop 20 Highest-Scoring Individual Replays:")
    print("-" * 110)
    print(f"{'Episode ID':<20} | {'Team 0 (Score)':<30} | {'Team 1 (Score)':<30} | {'Max Score':<12}")
    print("-" * 110)
    all_replays.sort(key=lambda x: max(x['r0'], x['r1']), reverse=True)
    for r in all_replays[:20]:
        t0_str = f"{r['team0'][:18]} (${r['r0']:,.0f})"
        t1_str = f"{r['team1'][:18]} (${r['r1']:,.0f})"
        max_s = f"${max(r['r0'], r['r1']):,.0f}"
        print(f"{str(r['ep_id']):<20} | {t0_str:<30} | {t1_str:<30} | {max_s:<12}")

if __name__ == '__main__':
    main()
