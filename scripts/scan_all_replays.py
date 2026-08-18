import glob
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    files = glob.glob('replays/**/*.json', recursive=True)
    print(f'Total JSON files found: {len(files)}')
    parsed = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
            if 'steps' not in d:
                continue
            names = d.get('info', {}).get('TeamNames', ['P0', 'P1'])
            last = d['steps'][-1]
            r0 = last[0].get('reward', 0)
            r1 = last[1].get('reward', 0)
            ep_id = d.get('id') or os.path.basename(f)
            parsed.append({
                'ep_id': ep_id,
                'names': names,
                'r0': r0,
                'r1': r1,
                'path': f,
                'steps': len(d['steps'])
            })
        except Exception as e:
            pass

    print(f'Valid replays parsed: {len(parsed)}')
    parsed.sort(key=lambda x: max(x['r0'] or 0, x['r1'] or 0), reverse=True)
    for p in parsed[:30]:
        print(f"ID: {p['ep_id']} | Teams: {p['names']} | Rewards: {p['r0']} vs {p['r1']} | File: {p['path']}")

if __name__ == '__main__':
    main()
