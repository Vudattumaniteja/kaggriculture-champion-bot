import glob
import json
import os
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("."))

from src.models.encoder import encode_observation, MACRO_ACTIONS, NUM_MACRO_ACTIONS
from src.training.dataset import infer_macro_action

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def analyze_top_replays():
    files = glob.glob('replays/**/*.json', recursive=True)
    
    top_teams = ["peikopon", "kowalskii", "Efe Can Celiksoy", "Galaxantic", "One-For-All", 
                 "Abracadabra", "カワシギ", "Thomas Tschinkel", "Kostiantyn Isaienkov", 
                 "SCLim2022080004", "Matteo123383iend", "ReCurSiON"]
    
    action_by_phase = defaultdict(Counter)
    action_by_team = defaultdict(Counter)
    total_parsed_steps = 0

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
            
            for p_idx in [0, 1]:
                team_name = names[p_idx]
                score = r0 if p_idx == 0 else r1
                # Filter for top teams or high scoring matches (> 80k)
                is_top = (score >= 75000) or any(t.lower() in team_name.lower() for t in top_teams)
                if not is_top:
                    continue
                
                for t in range(len(d['steps']) - 1):
                    obs = d['steps'][t][p_idx].get('observation', {})
                    act = d['steps'][t + 1][p_idx].get('action', {})
                    day = obs.get('day', t // 24)
                    
                    macro = infer_macro_action(act, day)
                    macro_name = MACRO_ACTIONS[macro]
                    
                    phase = "Opening (D0-D2)" if day <= 2 else ("Midgame (D3-D24)" if day <= 24 else "Late/Liquidation (D25-D29)")
                    action_by_phase[phase][macro_name] += 1
                    action_by_team[team_name][macro_name] += 1
                    total_parsed_steps += 1
        except Exception as e:
            pass

    print(f"Total Grandmaster Transitions Extracted: {total_parsed_steps:,}")
    print("\nAction Distribution by Game Phase:")
    print("-" * 65)
    for phase, counts in action_by_phase.items():
        total_p = sum(counts.values())
        print(f"\nPhase: {phase} ({total_p:,} steps)")
        for act_name, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {act_name:<30}: {cnt:>6,} ({cnt/total_p*100:>5.1f}%)")

if __name__ == '__main__':
    analyze_top_replays()
