"""
Apply classify_failure(...) to 26 universal residual failures × 3 models.
Compare label distribution between:
  - 26 universal residual (cross-model failure)
  - 44 only_qwen failures (capability gap)

Outputs:
  - exp/universal_labels.tsv (per-task per-model labels)
  - exp/label_distribution_compare.tsv (memory% per subset per model)
"""
import json, re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter, defaultdict

ROOT = Path('/home/xmdjy/workspace/MobileWorld')
QWEN_LOCAL = ROOT / 'output/test'
SEED_JSON = ROOT / 'downloads/mobileworld_seed2pro/seed-2.0-pro.gui-only.json'
CLAUDE_JSON = ROOT / 'downloads/mobileworld_claude_opus_4_7/claude-opus-4.7.gui-only.json'

RESULT_RE = re.compile(r'score:\s*([\d.]+)\s*reason:\s*(.+)', re.S)

MEMORY_LABELS = {'ProcMH-loop', 'ProcMH-drift', 'OMH-memory',
                 'PMH-set', 'PMH-attr', 'PMH-multi-attr',
                 'CCL', 'SCD'}
NON_MEMORY_LABELS = {'OMH-pure', 'OMH-no-action', 'AGE', 'TGM',
                     'ANSWER-WRONG', 'PERCEPTION', 'ENV-FAIL'}
BORDERLINE = {'PMH-missing-obj'}


def classify_failure(traj: list, reason: str) -> list:
    """Multi-label classify a single failure trajectory. Adapted from analysis.md §4."""
    labels = []
    r = reason.lower()

    # ----- L1: result.txt reason -----
    if 'missing:' in r or 'unexpected:' in r:
        labels.append('PMH-set')
    elif 'days mismatch' in r and 'weekend' in r:
        labels.append('PMH-set')
    elif r.count('current ') >= 2:
        labels.append('PMH-multi-attr')
    elif 'is not at' in r and 'current' in r:
        labels.append('PMH-attr')
    elif 'mismatch' in r and ('expected' in r or 'actual' in r or '!=' in r):
        labels.append('PMH-attr')

    if 'not found' in r and any(k in r for k in
        ['contact', 'folder', 'channel', 'list', 'filter', 'tags', 'invite',
         'setting', 'event', 'image', 'file', 'export', 'languages', 'report']):
        labels.append('PMH-missing-obj')

    if 'wrong recipient' in r or 'to wrong' in r:
        labels.append('CCL')

    if 'is not at maximum' in r or 'is not at minimum' in r:
        labels.append('AGE')

    if 'incorrect answer' in r or 'invalid answer' in r or 'could not parse' in r or 'parse answer' in r:
        labels.append('ANSWER-WRONG')

    # ----- L2: trajectory patterns -----
    if not traj:
        return list(set(labels)) or ['NO-TRAJ']

    actions = []
    thoughts = []
    for s in traj:
        a = s.get('action') or {}
        actions.append(a.get('action_type', 'unknown'))
        thoughts.append((s.get('prediction') or '')[:300])
    n = len(actions)

    # ProcMH-loop: action_type 重复 ≥ 4
    for i in range(n - 3):
        if len(set(actions[i:i+4])) == 1 and actions[i] in ('drag', 'click', 'scroll', 'swipe'):
            labels.append('ProcMH-loop')
            break

    # ProcMH-loop (alt): thought 高相似 ≥ 3 连续
    if 'ProcMH-loop' not in labels:
        for i in range(n - 2):
            s1 = SequenceMatcher(None, thoughts[i], thoughts[i+1]).ratio()
            s2 = SequenceMatcher(None, thoughts[i+1], thoughts[i+2]).ratio()
            if s1 > 0.75 and s2 > 0.75:
                labels.append('ProcMH-loop')
                break

    # ProcMH-drift: ≥ 50 步且最后没 terminate
    last_at = actions[-1] if actions else ''
    if n >= 50 and last_at not in ('finished', 'answer', 'terminate'):
        labels.append('ProcMH-drift')

    # OMH-pure / OMH-no-action: 自封成功但 score=0
    if last_at in ('finished', 'terminate') or 'finish' in last_at.lower():
        if n <= 5 and 'not found' in r and any(k in r for k in ['sms', 'email', 'callback', 'sent']):
            labels.append('OMH-no-action')
        else:
            labels.append('OMH-pure')

    # ANSWER-WRONG (also from action type)
    if last_at == 'answer':
        if 'ANSWER-WRONG' not in labels:
            labels.append('ANSWER-WRONG')

    # PROCESS-INCOMPLETE
    if n >= 50 and last_at not in ('finished', 'terminate', 'answer') and 'ProcMH-drift' in labels:
        labels.append('PROCESS-INCOMPLETE')

    return list(set(labels)) or ['UNCLASSIFIED']


def is_memory(labels: list) -> tuple:
    has_mem = any(l in MEMORY_LABELS for l in labels)
    has_nonmem = any(l in NON_MEMORY_LABELS for l in labels)
    has_border = any(l in BORDERLINE for l in labels)
    return has_mem, has_nonmem, has_border


def parse_local(root: Path) -> dict:
    """Returns {task_name: {result, traj}}"""
    out = {}
    for d in root.iterdir():
        if not d.is_dir() or 'backup' in d.name:
            continue
        rt = d / 'result.txt'
        tj = d / 'traj.json'
        if not rt.exists() or not tj.exists():
            continue
        m = RESULT_RE.search(rt.read_text())
        if not m: continue
        try:
            with open(tj) as f:
                tdata = json.load(f)
            traj = tdata.get('0', {}).get('traj', [])
        except Exception:
            traj = []
        out[d.name] = {
            'score': float(m.group(1)),
            'reason': m.group(2).strip(),
            'traj': traj,
        }
    return out


def parse_lb(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    out = {}
    for tname, tdata in d.items():
        if tname.startswith('_'): continue
        m = RESULT_RE.search(tdata['result'])
        if not m: continue
        out[tname] = {
            'score': float(m.group(1)),
            'reason': m.group(2).strip(),
            'traj': tdata.get('traj', []),
        }
    return out


def main():
    qwen = parse_local(QWEN_LOCAL)
    seed = parse_lb(SEED_JSON)
    claude = parse_lb(CLAUDE_JSON)

    common = set(qwen) & set(seed) & set(claude)
    qf = {t for t in common if qwen[t]['score'] < 1.0}
    sf = {t for t in common if seed[t]['score'] < 1.0}
    cf = {t for t in common if claude[t]['score'] < 1.0}

    universal = qf & sf & cf
    only_qwen = qf - sf - cf
    print(f'universal residual: {len(universal)}')
    print(f'only_qwen: {len(only_qwen)}')

    # === Run classify_failure on universal × 3 models ===
    rows = []
    rows.append(['task', 'qwen_n_steps', 'qwen_labels', 'claude_n_steps', 'claude_labels',
                 'seed_n_steps', 'seed_labels', 'reason_qwen', 'reason_claude', 'reason_seed'])
    label_count_per_model = {'qwen': Counter(), 'claude': Counter(), 'seed': Counter()}
    mem_per_model = defaultdict(lambda: {'mem': 0, 'nonmem': 0, 'border': 0, 'mem_only': 0, 'nonmem_only': 0, 'both': 0})

    for t in sorted(universal):
        ql = classify_failure(qwen[t]['traj'], qwen[t]['reason'])
        cl = classify_failure(claude[t]['traj'], claude[t]['reason'])
        sl = classify_failure(seed[t]['traj'], seed[t]['reason'])
        rows.append([t,
                     str(len(qwen[t]['traj'])), ','.join(ql),
                     str(len(claude[t]['traj'])), ','.join(cl),
                     str(len(seed[t]['traj'])), ','.join(sl),
                     qwen[t]['reason'][:80], claude[t]['reason'][:80], seed[t]['reason'][:80]])
        for m, labels in [('qwen', ql), ('claude', cl), ('seed', sl)]:
            for l in labels:
                label_count_per_model[m][l] += 1
            hm, hnm, hb = is_memory(labels)
            if hm: mem_per_model[m]['mem'] += 1
            if hnm: mem_per_model[m]['nonmem'] += 1
            if hb: mem_per_model[m]['border'] += 1
            if hm and not hnm: mem_per_model[m]['mem_only'] += 1
            if hnm and not hm: mem_per_model[m]['nonmem_only'] += 1
            if hm and hnm: mem_per_model[m]['both'] += 1

    out = ROOT / 'exp/universal_labels.tsv'
    with open(out, 'w') as f:
        for r in rows:
            f.write('\t'.join(r) + '\n')
    print(f'\nSaved: {out}')

    # === Same on only_qwen for comparison ===
    only_q_label = Counter()
    only_q_mem = {'mem': 0, 'nonmem': 0, 'border': 0, 'mem_only': 0, 'nonmem_only': 0, 'both': 0}
    for t in only_qwen:
        ql = classify_failure(qwen[t]['traj'], qwen[t]['reason'])
        for l in ql: only_q_label[l] += 1
        hm, hnm, hb = is_memory(ql)
        if hm: only_q_mem['mem'] += 1
        if hnm: only_q_mem['nonmem'] += 1
        if hb: only_q_mem['border'] += 1
        if hm and not hnm: only_q_mem['mem_only'] += 1
        if hnm and not hm: only_q_mem['nonmem_only'] += 1
        if hm and hnm: only_q_mem['both'] += 1

    # === Print comparison report ===
    print('\n' + '=' * 70)
    print('UNIVERSAL RESIDUAL (n=26) label distribution per model:')
    print('=' * 70)
    all_labels = set()
    for m in label_count_per_model.values(): all_labels.update(m)
    print(f"{'Label':22} {'Qwen':>8} {'Claude':>8} {'Seed':>8}  | category")
    cat = {**{l: 'memory' for l in MEMORY_LABELS},
           **{l: 'non-mem' for l in NON_MEMORY_LABELS},
           **{l: 'border' for l in BORDERLINE}}
    for l in sorted(all_labels, key=lambda x: -max(label_count_per_model['qwen'][x],
                                                    label_count_per_model['claude'][x],
                                                    label_count_per_model['seed'][x])):
        c = cat.get(l, '?')
        q = label_count_per_model['qwen'][l]
        cl = label_count_per_model['claude'][l]
        s = label_count_per_model['seed'][l]
        print(f'{l:22} {q:>8} {cl:>8} {s:>8}  | {c}')

    print('\n' + '=' * 70)
    print('UNIVERSAL — Memory vs Non-memory coverage per model (n=26 each):')
    print('=' * 70)
    print(f"{'Model':10} {'has_mem':>10} {'has_nonmem':>12} {'has_border':>12}  {'mem_only':>10} {'nonmem_only':>12} {'both':>6}")
    for m in ['qwen', 'claude', 'seed']:
        d = mem_per_model[m]
        print(f"{m:10} {d['mem']:>10} {d['nonmem']:>12} {d['border']:>12}  {d['mem_only']:>10} {d['nonmem_only']:>12} {d['both']:>6}")

    print('\n' + '=' * 70)
    print('COMPARISON: 44 only_qwen vs 26 universal — Qwen labels')
    print('=' * 70)
    print(f"{'Label':22} {'only_qwen(n=44)':>18} {'universal(n=26)':>18}  {'shift':>10}")
    all_labels2 = set(only_q_label) | set(label_count_per_model['qwen'])
    for l in sorted(all_labels2, key=lambda x: -(only_q_label[x] + label_count_per_model['qwen'][x])):
        oq_pct = only_q_label[l] / 44 * 100
        u_pct = label_count_per_model['qwen'][l] / 26 * 100
        delta = u_pct - oq_pct
        sign = '+' if delta >= 0 else ''
        print(f'{l:22} {only_q_label[l]:>5} ({oq_pct:>5.1f}%)  {label_count_per_model["qwen"][l]:>5} ({u_pct:>5.1f}%)  {sign}{delta:>5.1f}pp')

    print('\n' + '=' * 70)
    print('Memory class share comparison:')
    print('=' * 70)
    for cat_name, agg_func in [('mem%', lambda d, n: d['mem']/n*100),
                                ('nonmem%', lambda d, n: d['nonmem']/n*100),
                                ('mem_only%', lambda d, n: d['mem_only']/n*100),
                                ('nonmem_only%', lambda d, n: d['nonmem_only']/n*100),
                                ('both%', lambda d, n: d['both']/n*100)]:
        oq_v = agg_func(only_q_mem, 44)
        uq_v = agg_func(mem_per_model['qwen'], 26)
        uc_v = agg_func(mem_per_model['claude'], 26)
        us_v = agg_func(mem_per_model['seed'], 26)
        print(f'{cat_name:14}  only_qwen={oq_v:5.1f}%  universal_qwen={uq_v:5.1f}%  universal_claude={uc_v:5.1f}%  universal_seed={us_v:5.1f}%')


if __name__ == '__main__':
    main()
