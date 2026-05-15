"""Deep dive into LocalFileManagementTask2 — extract data needed for failure analysis."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path('/home/xmdjy/workspace/MobileWorld')
seed = json.load(open(ROOT / 'downloads/mobileworld_seed2pro/seed-2.0-pro.gui-only.json'))
claude = json.load(open(ROOT / 'downloads/mobileworld_claude_opus_4_7/claude-opus-4.7.gui-only.json'))

TASK = 'LocalFileManagementTask2'

for name, data in [('SEED', seed), ('CLAUDE', claude)]:
    if TASK not in data: continue
    td = data[TASK]
    traj = td['traj']
    print('=' * 90)
    print(f'{name}  n_steps={len(traj)}')
    print(f'Goal: {traj[0]["task_goal"]}')
    print(f'Result: {td["result"]}')
    print()

    # Action type histogram
    ats = Counter(s['action'].get('action_type', '?') for s in traj if s.get('action'))
    print(f'Action types: {dict(ats)}')
    print()

    # Print every 5th step's thought + action (so we get a high-level trace)
    print('Trajectory trace (sampled):')
    for i in [0, 1, 2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 45, 49]:
        if i >= len(traj): continue
        t = (traj[i].get('prediction') or '')[:250].replace('\n', ' ')
        a = traj[i].get('action') or {}
        aa = json.dumps(a, ensure_ascii=False)[:100]
        print(f'  [s{i+1:2d}] {t}')
        print(f'       → {aa}')
    print()
