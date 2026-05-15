"""
Sample real trajectories from universal failures to verify root cause.
For each task: print last few thoughts + action history pattern.
"""
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path('/home/xmdjy/workspace/MobileWorld')

def load(path, is_local=False):
    if is_local:
        with open(path) as f:
            d = json.load(f)
        return d['0']
    with open(path) as f:
        return json.load(f)

seed = load(ROOT / 'downloads/mobileworld_seed2pro/seed-2.0-pro.gui-only.json')
claude = load(ROOT / 'downloads/mobileworld_claude_opus_4_7/claude-opus-4.7.gui-only.json')

# Pick 6 representative universal tasks across categories
SAMPLES = [
    ('MastodonAddFeaturedHashtagsTask', 'mastodon-feature-may-not-exist'),
    ('MastodonChangeLanguageTask', 'simple-setting-toggle'),
    ('CheckGithubInfoTask', 'cross-app-info-lookup'),
    ('LocalFileManagementTask2', 'file-ops'),
    ('PhotoManagementTask', 'folder-not-found'),
    ('MastodonGetServerInfoTask', 'wrong-info-extraction'),
]

for task_name, why in SAMPLES:
    print('=' * 90)
    print(f'TASK: {task_name}    [{why}]')
    print('=' * 90)
    for model_name, data in [('SEED', seed), ('CLAUDE', claude)]:
        if task_name not in data: continue
        td = data[task_name]
        traj = td['traj']
        print(f'\n--- {model_name} (n_steps={len(traj)}) result: {td["result"][:200]}')
        if traj:
            goal = traj[0].get('task_goal', '')
            print(f'Goal: {goal[:200]}')

            # Action type histogram
            ats = Counter()
            for s in traj:
                a = s.get('action') or {}
                ats[a.get('action_type', '?')] += 1
            print(f'Action types: {dict(ats)}')

            # Print first 2 thoughts and last 3 thoughts
            print('\n  First 2 thoughts:')
            for i in range(min(2, len(traj))):
                t = (traj[i].get('prediction') or '')[:300].replace('\n', ' ')
                a = traj[i].get('action') or {}
                print(f'    [step {i+1}] {t}')
                print(f'      → action: {a.get("action_type")} {{x:{a.get("x")},y:{a.get("y")}}}')
            print('\n  Last 3 thoughts:')
            for i in range(max(0, len(traj)-3), len(traj)):
                t = (traj[i].get('prediction') or '')[:400].replace('\n', ' ')
                a = traj[i].get('action') or {}
                aa = json.dumps(a, ensure_ascii=False)[:150]
                print(f'    [step {i+1}] {t}')
                print(f'      → action: {aa}')
        print()
    print()
