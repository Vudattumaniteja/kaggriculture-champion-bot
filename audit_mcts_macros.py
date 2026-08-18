import json
import submission
from collections import defaultdict

replay_path = r'C:\Users\Manit\Downloads\93966516.json'
with open(replay_path, 'r') as f:
    replay = json.load(f)

steps = replay['steps']

bot = submission.FrontierGrandmasterUnifiedBot()

macro_names = submission.MACRO_ACTIONS

print("=" * 80)
print("AUDITING MCTS MACRO-ACTION SELECTIONS ACROSS 720 TURNS")
print("=" * 80)

for p in range(2):
    macro_counts = defaultdict(int)
    macro_by_day = defaultdict(lambda: defaultdict(int))
    
    print(f"\nEvaluating Player {p}...")
    for s_idx in range(len(steps)):
        obs = steps[s_idx][p].get('observation')
        if not obs or not obs.get('farms'):
            continue
        day = obs.get('day', 0)
        
        # Run search
        best_macro, pi_target, root_val = bot.searcher.search(obs, temperature=0.0)
        macro_counts[best_macro] += 1
        macro_by_day[day][best_macro] += 1

    print(f"Player {p} Macro Action Distribution:")
    for m_idx in range(len(macro_names)):
        cnt = macro_counts[m_idx]
        print(f"  [{m_idx}] {macro_names[m_idx]:<30}: {cnt:>4} times ({cnt/len(steps)*100:.1f}%)")

    print("\nMacro Selections by Day Group:")
    for d_range, name in [((0, 5), "Days 0-4"), ((5, 10), "Days 5-9"), ((10, 15), "Days 10-14"), ((15, 20), "Days 15-19"), ((20, 25), "Days 20-24"), ((25, 30), "Days 25-29")]:
        agg = defaultdict(int)
        for d in range(d_range[0], d_range[1]):
            for m_idx, c in macro_by_day[d].items():
                agg[m_idx] += c
        print(f"  {name}: {dict(sorted(agg.items()))}")
