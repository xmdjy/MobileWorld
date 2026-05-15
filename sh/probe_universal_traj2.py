"""Sample second batch — focus on tasks that classify_failure flagged as memory-heavy."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path('/home/xmdjy/workspace/MobileWorld')
seed = json.load(open(ROOT / 'downloads/mobileworld_seed2pro/seed-2.0-pro.gui-only.json'))
claude = json.load(open(ROOT / 'downloads/mobileworld_claude_opus_4_7/claude-opus-4.7.gui-only.json'))

SAMPLES = [
    ('MastodonNewFilterTask', 'sounds-like-PMH-missing-obj'),
    ('MastodonReportTask', 'PMH-attr-or-content-match'),
    ('MattermostProjectStatusReportTask', 'multi-step-cross-app'),
    ('MastodonMultiInviteTask', 'SCD-multi-constraint'),
    ('MastodonInviteTask', 'SMS-content-mismatch'),
    ('MastodonRevisePhotoAltTask', 'content-extraction'),
    ('MastodonOpenAutomatedDeletionTask', 'setting-not-found'),
    ('MastodonShareLocationTask', 'URL-share'),
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
            print(f'Goal: {traj[0].get("task_goal", "")[:200]}')
            ats = Counter()
            for s in traj:
                a = s.get('action') or {}
                ats[a.get('action_type', '?')] += 1
            print(f'Action types: {dict(ats)}')
            # Print last 3 thoughts only (compact)
            print('  Last 3 thoughts:')
            for i in range(max(0, len(traj)-3), len(traj)):
                t = (traj[i].get('prediction') or '')[:300].replace('\n', ' ')
                a = traj[i].get('action') or {}
                aa = json.dumps(a, ensure_ascii=False)[:120]
                print(f'    [s{i+1}] {t}')
                print(f'      → {aa}')
        print()
