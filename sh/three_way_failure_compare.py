"""
3-way failure comparison: Qwen3-8B (local) vs Claude Opus 4.7 vs Seed-2.0-Pro.

Outputs:
  - SR per model on the common task set
  - 7-region Venn of failure sets
  - Universal residual failure list (all 3 fail) -> gold set for paper analysis
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path('/home/xmdjy/workspace/MobileWorld')
QWEN_LOCAL = ROOT / 'output/test'
SEED_JSON = ROOT / 'downloads/mobileworld_seed2pro/seed-2.0-pro.gui-only.json'
CLAUDE_JSON = ROOT / 'downloads/mobileworld_claude_opus_4_7/claude-opus-4.7.gui-only.json'

RESULT_RE = re.compile(r'score:\s*([\d.]+)\s*reason:\s*(.+)', re.S)


def parse_local(root: Path) -> dict:
    out = {}
    for d in root.iterdir():
        if not d.is_dir() or 'backup' in d.name:
            continue
        rt = d / 'result.txt'
        if not rt.exists():
            continue
        m = RESULT_RE.search(rt.read_text())
        if m:
            out[d.name] = {'score': float(m.group(1)), 'reason': m.group(2).strip()}
    return out


def parse_leaderboard_json(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    out = {}
    for tname, tdata in d.items():
        if tname.startswith('_'):
            continue
        m = RESULT_RE.search(tdata['result'])
        if m:
            out[tname] = {
                'score': float(m.group(1)),
                'reason': m.group(2).strip(),
                'n_steps': len(tdata.get('traj', [])),
            }
    return out


def venn3(qf: set, cf: set, sf: set) -> dict:
    return {
        'only_qwen': qf - cf - sf,
        'only_claude': cf - qf - sf,
        'only_seed': sf - qf - cf,
        'qc_not_s': (qf & cf) - sf,
        'qs_not_c': (qf & sf) - cf,
        'cs_not_q': (cf & sf) - qf,
        'all3': qf & cf & sf,
    }


def main() -> None:
    qwen = parse_local(QWEN_LOCAL)
    seed = parse_leaderboard_json(SEED_JSON)
    claude = parse_leaderboard_json(CLAUDE_JSON)

    common = set(qwen) & set(seed) & set(claude)
    print(f'Common tasks across 3 models: {len(common)}')
    print(f'  Qwen local total: {len(qwen)}, Seed: {len(seed)}, Claude: {len(claude)}')

    def fails(d: dict) -> set:
        return {t for t in common if d[t]['score'] < 1.0}

    qf, cf, sf = fails(qwen), fails(claude), fails(seed)

    print('\n=== SR on common subset ===')
    for name, fset in [('Qwen3-8B', qf), ('Claude4.7', cf), ('Seed2.0Pro', sf)]:
        succ = len(common) - len(fset)
        print(f'  {name:12s}: {succ}/{len(common)} = {succ / len(common) * 100:.1f}%')

    regions = venn3(qf, cf, sf)
    print('\n=== Failure Venn ===')
    for k, v in regions.items():
        print(f'  {k:14s}: {len(v):3d}')

    out_path = ROOT / 'exp/universal_residual_failures.tsv'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('task\tqwen_reason\tclaude_reason\tseed_reason\n')
        for t in sorted(regions['all3']):
            f.write(f'{t}\t{qwen[t]["reason"][:120]}\t{claude[t]["reason"][:120]}\t{seed[t]["reason"][:120]}\n')
    print(f'\nUniversal residual failures saved -> {out_path}')


if __name__ == '__main__':
    main()
