"""Tabular-JEPA v3 ladder judge (docs/prior_v2_plan.md section 5, R1-R3): per-dataset paired sign tests, overall and by
class count, from the suite JSONs only (E9 protocol, K_FEAT 64 / CTX 1024). Pairs whose pass is missing are skipped.
Run: uv run python eval/prior_ladder_2026-09-10/judge_v3.py   -> results_judge_v3.json + judge_v3.md (this directory)"""
import json, statistics as st
from math import comb
from pathlib import Path

R = Path(__file__).resolve().parents[2]; D = Path(__file__).parent
PASSES = ['results_suite_TabularJEPA_v3_tabicl2_s0.json', 'results_suite_PFN_tabicl2_ds_s0.json', 'results_suite_TabularJEPA_v3_tabicl1_s0.json',
          'results_suite_TabularJEPA_v3_scm_w64_s0.json', 'results_suite_real_dualsplit.json', 'results_suite_real_ds.json',
          # architecture ladder 2 (docs/prior_v2_plan.md 5.2, 2026-09-11): value head on the encoder; EMA / DAE / SIGReg targets
          'results_suite_TabularJEPA_v3_tabicl2_headenc_s0.json', 'results_suite_TabularJEPA_v3_tabicl2_dae_headenc_s0.json',
          'results_suite_TabularJEPA_v3_tabicl2_sigreg_headenc_s0.json']
DUAL, RDS = 'v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0', 'real_ds_lr5e-4_s0'
PAIRS = [('R1', 'TabularJEPA_v3_tabicl2_s0_snap9250', DUAL), ('R1', 'TabularJEPA_v3_tabicl2_s0', DUAL),
         ('R2', 'TabularJEPA_v3_tabicl2_s0_snap9250', 'PFN_tabicl2_ds_s0'), ('R2', 'TabularJEPA_v3_tabicl2_s0', 'PFN_tabicl2_ds_s0'),
         ('R3', 'TabularJEPA_v3_tabicl2_s0_snap9250', 'TabularJEPA_v3_tabicl1_s0'), ('R3', 'TabularJEPA_v3_tabicl1_s0', 'TabularJEPA_v3_scm_w64_s0'),
         ('prior|ds', 'PFN_tabicl2_ds_s0', RDS), ('prior|ds', 'PFN_tabicl2_ds_s0', DUAL),
         ('ref', 'TabularJEPA_v3_tabicl2_s0_snap9250', 'histgb'), ('ref', 'PFN_tabicl2_ds_s0', 'histgb'), ('ref', DUAL, 'histgb')]
# L2 = ladder-2 pre-registered value judgment (plan 5.2): each arm vs the ds baseline on the same prior and vs v3's best snapshot;
# L2x = the three arms against each other (target family: EMA diff vs DAE vs SIGReg, all with the encoder head).
L2 = ['TabularJEPA_v3_tabicl2_headenc_s0', 'TabularJEPA_v3_tabicl2_dae_headenc_s0', 'TabularJEPA_v3_tabicl2_sigreg_headenc_s0']
PAIRS += [('L2', a, b) for a in L2 for b in ('PFN_tabicl2_ds_s0', 'TabularJEPA_v3_tabicl2_s0_snap9250', DUAL)]
PAIRS += [('L2x', L2[i], L2[j]) for i in range(3) for j in range(i + 1, 3)]
# L2t = step-matched read (2026-09-13 analysis): the ds baseline at the snapshot whose val_mse first matches headenc's 20k value
# (.5513: ds reaches it at 7750, nearest 1k snapshot = 8000). Slowdown vs ceiling: a ~50:50 here means the latent term only
# delays the value path at this horizon. ds 20k vs its own 8k snapshot is the sanity row.
DS8K = 'PFN_tabicl2_ds_s0_snap8000'
PASSES += ['results_suite_PFN_tabicl2_ds_s0_snap8000.json']
PAIRS += [('L2t', a, DS8K) for a in (L2[0], L2[1], 'PFN_tabicl2_ds_s0')]
# L3 = open-horizon pair (plan 5.3, 2026-09-14): every scored *_conv_s0 pass (20k snapshots via e9_gpu.sh, final = no snap
# suffix) vs the 20k ds baseline, and each headenc pass vs the ds pass with the same suffix. Files land as the runs progress.
import glob, os
CONV = sorted(os.path.basename(p) for p in glob.glob(str(R / 'eval' / 'results_suite_*_conv_s0*.json')))
PASSES += CONV
L3 = [f[len('results_suite_'):-len('.json')] for f in CONV]
PAIRS += [('L3', s, 'PFN_tabicl2_ds_s0') for s in L3]
PAIRS += [('L3', s, s.replace('TabularJEPA_v3_tabicl2_headenc', 'PFN_tabicl2_ds')) for s in L3 if s.startswith('TabularJEPA')]
PAIRS += [('L3', s, 'PFN_tabicl2_ds_conv_s0') for s in L3 if s.startswith('TabularJEPA') and '_snap' in s]   # headenc snapshots vs the stopped ds final
STRATA = [('all', lambda k, c: True), ('ncls=2', lambda k, c: c == 2), ('ncls=3-5', lambda k, c: c is not None and 3 <= c <= 5),
          ('ncls=6-10', lambda k, c: c is not None and 6 <= c <= 10),
          ('src=CC18', lambda k, c: k.startswith('CC18|')), ('src=Grinsztajn', lambda k, c: k.startswith('Grinsztajn|')),   # by benchmark, both tasks
          ('src=TabArena', lambda k, c: k.startswith('TabArena|'))]

def load(name):
    p = R / 'eval' / name
    return json.load(open(p)) if p.exists() else {}

passes = [load(f) for f in PASSES]
ncls = {r['key']: r['n_classes'] for r in load('results_class_count_census.json')['datasets']}
keys = sorted(set().union(*[p.keys() for p in passes]))

def score(key, model):
    if model == 'histgb' and key.endswith('|reg'):
        model = 'histgbr'
    if model == 'linear':                                   # the suite's linear baselines: logreg (clf) / ridge (reg)
        model = 'ridge' if key.endswith('|reg') else 'logreg'
    for p in passes:
        e = p.get(key, {}).get(model)
        if e is not None:
            return e

def sign_p(w, l):
    n, k = w + l, min(w, l)
    return None if n == 0 else round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 5)

def compare(a, b, task, keep):
    m = 'acc' if task == 'clf' else 'r2'
    xa, xb = [], []
    for k in keys:
        if not k.endswith('|' + task) or not keep(k, ncls.get(k)):   # class strata keep nothing on reg -> that row is skipped
            continue
        sa, sb = score(k, a), score(k, b)
        if sa is not None and sb is not None:
            xa.append(sa[m]); xb.append(sb[m])
    if not xa:
        return None
    w = sum(x > y for x, y in zip(xa, xb)); l = sum(x < y for x, y in zip(xa, xb))
    return dict(n=len(xa), win=w, loss=l, tie=len(xa) - w - l, p=sign_p(w, l), mean_a=round(st.mean(xa), 4), mean_b=round(st.mean(xb), 4))

out, md = {}, ['<!-- generated by eval/prior_ladder_2026-09-10/judge_v3.py from the suite JSONs -->',
               '| rule | A | B | task | stratum | n | A>B | A<B | p (sign) | mean A | mean B |', '|---|---|---|---|---|---|---|---|---|---|---|']
for rule, a, b in PAIRS:
    for task in ('clf', 'reg'):
        for lab, keep in STRATA:
            r = compare(a, b, task, keep)
            if r is None:
                continue
            out[f'{rule}|{a}|{b}|{task}|{lab}'] = r
            md.append(f"| {rule} | {a} | {b} | {task} | {lab} | {r['n']} | {r['win']} | {r['loss']} | {r['p']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} |")
            print(f"{rule:8s} {a[:28]:28s} vs {b[:28]:28s} {task} {lab:9s} n={r['n']:3d} {r['win']:3d}:{r['loss']:<3d} p={r['p']}  {r['mean_a']:.3f} vs {r['mean_b']:.3f}")
# baseline table: every run vs HistGB and vs the linear baseline, clf (acc) and reg (r2), all datasets
RUNS = ['PFN_tabicl2_ds_s0', 'TabularJEPA_v3_tabicl2_s0_snap9250', 'TabularJEPA_v3_tabicl2_s0', 'TabularJEPA_v3_tabicl1_s0',
        'TabularJEPA_v3_scm_w64_s0', RDS, DUAL] + L2 + [DS8K] + L3
md += ['', '<!-- baselines: same passes; W:L = per-dataset wins:losses of the run against the baseline -->',
       '| run | clf n | mean acc | vs HistGB W:L | vs logreg W:L | reg n | mean r2 | vs HistGB W:L | vs ridge W:L |', '|---|---|---|---|---|---|---|---|---|']
base = {}
for run in RUNS:
    c_h, c_l = compare(run, 'histgb', 'clf', lambda k, c: True), compare(run, 'linear', 'clf', lambda k, c: True)
    r_h, r_l = compare(run, 'histgb', 'reg', lambda k, c: True), compare(run, 'linear', 'reg', lambda k, c: True)
    if c_h is None:
        continue
    base[run] = dict(clf=dict(n=c_h['n'], mean=c_h['mean_a'], vs_histgb=[c_h['win'], c_h['loss']], vs_logreg=[c_l['win'], c_l['loss']], histgb_mean=c_h['mean_b'], logreg_mean=c_l['mean_b']),
                     reg=dict(n=r_h['n'], mean=r_h['mean_a'], vs_histgb=[r_h['win'], r_h['loss']], vs_ridge=[r_l['win'], r_l['loss']], histgb_mean=r_h['mean_b'], ridge_mean=r_l['mean_b']))
    md.append(f"| {run} | {c_h['n']} | {c_h['mean_a']:.3f} | {c_h['win']}:{c_h['loss']} | {c_l['win']}:{c_l['loss']} | {r_h['n']} | {r_h['mean_a']:.3f} | {r_h['win']}:{r_h['loss']} | {r_l['win']}:{r_l['loss']} |")
    print(f"BASE {run[:36]:36s} clf n={c_h['n']:3d} acc {c_h['mean_a']:.3f}  vsHGB {c_h['win']:3d}:{c_h['loss']:<3d} vsLR {c_l['win']:3d}:{c_l['loss']:<3d} | reg n={r_h['n']:2d} r2 {r_h['mean_a']:.3f} vsHGB {r_h['win']:2d}:{r_h['loss']:<2d} vsRidge {r_l['win']:2d}:{r_l['loss']:<2d}")
md.append(f"| (baseline means) | | HistGB {base[RUNS[0]]['clf']['histgb_mean']:.3f} / logreg {base[RUNS[0]]['clf']['logreg_mean']:.3f} | | | | HistGB {base[RUNS[0]]['reg']['histgb_mean']:.3f} / ridge {base[RUNS[0]]['reg']['ridge_mean']:.3f} | | |")
json.dump(dict(note='E9 protocol passes (K_FEAT 64 / CTX 1024); exact two-sided sign test on per-dataset wins; clf = acc, reg = r2; '
                    'class strata from results_class_count_census.json; missing passes skipped', passes=[p for p in PASSES if (R / 'eval' / p).exists()],
               results=out, baselines=base), open(D / 'results_judge_v3.json', 'w'), indent=1)
(D / 'judge_v3.md').write_text('\n'.join(md) + '\n'); print('wrote', D / 'results_judge_v3.json', D / 'judge_v3.md')
