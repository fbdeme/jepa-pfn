"""Aggregate the 5-seed C5 (matched nonlinear rerun) into mean +/- SD.

Reads eval/results_probe_matched_L4_s{0..4}.json (each = ds/dual/lat/random_init edge_auc + mse_f
on the L4 nonlinear prior, lat at its per-seed pre-collapse steelman) and writes
eval/results_probe_matched_L4_seeds.json: per-arm 5-seed mean/SD of edge_auc and mse_f@1.0, the
ds-lat gap, and the random-init floor. Compares to the paper's base 5-seed
(results_probe_seeds.json). Source-backed, robust to missing seeds.

Reproduce (per seed s):
  train.train      configs/matched_L4_ds.yaml   seed=s run_name=matched_L4_ds_s{s}
  train.train_jepa configs/matched_L4_dual.yaml seed=s run_name=matched_L4_dual_s{s}
  train.train_jepa configs/matched_L4_lat.yaml  seed=s run_name=matched_L4_lat_s{s}
  STEELMAN_GLOB=matched_L4_lat_s{s} STEELMAN_TAG=matched_L4_s{s} eval.pick_steelman
  PROBE_RUNS=matched_L4_{ds,dual,lat}_s{s} PROBE_TAG=matched_L4_s{s} PROBE_MLP_DEPTH=3 PROBE_MLP_GAIN=3.0 eval.probe_base
Run: uv run python -m eval.matched_L4_seeds_agg
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
ARMS = {"ds": "matched_L4_ds", "dual": "matched_L4_dual", "lat": "matched_L4_lat",
        "floor": "random_init"}
DS_LEARNED_MAX = 0.5   # a seed is VALID only if its data-space arm learned (ds mse_f@1 < this).
# At L4 (+.128, peak difficulty = the learnability edge) some seeds fail to train: ds val_mse /
# probe mse_f stays ~var(f)~0.96 instead of ~0.08. Those are training failures, not data points,
# and are excluded from the aggregate (reported separately).


def valid_seeds():
    ok, bad = [], []
    for s in range(5):
        p = ROOT / f"eval/results_probe_matched_L4_s{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        k = f"matched_L4_ds_s{s}"
        if k in d and d[k]["mse_f"]["1.0"] < DS_LEARNED_MAX:
            ok.append(s)
        else:
            bad.append(s)
    return ok, bad


VALID, EXCLUDED = valid_seeds()


def _series(arm_key):
    edge, msef = [], []
    for s in VALID:
        p = ROOT / f"eval/results_probe_matched_L4_s{s}.json"
        d = json.loads(p.read_text())
        # arm_key is a template like "matched_L4_ds"; per-seed key adds _s{s}, floor is shared
        key = arm_key if arm_key == "random_init" else f"{arm_key}_s{s}"
        if key not in d:
            continue
        edge.append(d[key]["edge_auc"])
        msef.append(d[key]["mse_f"]["1.0"])
    return np.array(edge, float), np.array(msef, float)


def _stat(a):
    return (dict(mean=round(float(a.mean()), 4), sd=round(float(a.std()), 4), n=int(len(a)))
            if len(a) else None)


def main():
    per = {}
    for name, key in ARMS.items():
        e, m = _series(key)
        per[name] = dict(edge_auc=_stat(e), mse_f=_stat(m))
    # per-seed ds-lat gaps (paired). VALID = conditional on ds training; ITT = all seeds present,
    # including the excluded one (where NO arm trained, so its gap ~= 0). Also record, for each
    # excluded seed, every arm's f-MSE, so "all arms failed" (not a data-space-specific exclusion)
    # is source-visible and the selection-bias objection is defused deterministically.
    gaps_edge, gaps_msef, gaps_itt = [], [], []
    for s in VALID:
        d = json.loads((ROOT / f"eval/results_probe_matched_L4_s{s}.json").read_text())
        ds, lat = f"matched_L4_ds_s{s}", f"matched_L4_lat_s{s}"
        if ds in d and lat in d:
            gaps_edge.append(d[ds]["edge_auc"] - d[lat]["edge_auc"])
            gaps_msef.append(d[ds]["mse_f"]["1.0"] - d[lat]["mse_f"]["1.0"])
    ratios = []
    excl_detail = {}
    for s in range(5):
        p = ROOT / f"eval/results_probe_matched_L4_s{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ds, lat = f"matched_L4_ds_s{s}", f"matched_L4_lat_s{s}"
        if ds in d and lat in d:
            gaps_itt.append(d[ds]["mse_f"]["1.0"] - d[lat]["mse_f"]["1.0"])
        if s in EXCLUDED:
            excl_detail[str(s)] = {a: round(d[f"matched_L4_{a}_s{s}"]["mse_f"]["1.0"], 3)
                                   for a in ("ds", "dual", "lat") if f"matched_L4_{a}_s{s}" in d}
        elif s in VALID and ds in d and lat in d and d[ds]["mse_f"]["1.0"] > 1e-6:
            ratios.append(d[lat]["mse_f"]["1.0"] / d[ds]["mse_f"]["1.0"])   # lat/ds per valid seed
    out = dict(
        note="C5, L4 (+.128) prior. gap = ds - lat (mse_f lower=better); lat at steelman. VALID = "
             "conditional on ds training; ITT = all seeds incl. the untrained one (gap~0).",
        valid_seeds=VALID, excluded_seeds_ds_failed=EXCLUDED, n_valid=len(VALID),
        excluded_all_arms_mse_f=excl_detail,   # every arm ~var(f) at the excluded seed => no bias direction
        per_arm=per,
        ds_minus_lat_edge=_stat(np.array(gaps_edge, float)),
        ds_minus_lat_msef=_stat(np.array(gaps_msef, float)),
        ds_minus_lat_msef_itt=_stat(np.array(gaps_itt, float)),   # intent-to-treat, all seeds
        lat_over_ds_ratio_range=[round(min(ratios), 1), round(max(ratios), 1)] if ratios else None,
    )
    # base 5-seed reference, if present
    base = ROOT / "eval/results_probe_seeds.json"
    if base.exists():
        try:
            bd = json.loads(base.read_text())
            out["base_reference_file"] = "results_probe_seeds.json"
        except Exception:
            pass
    (ROOT / "eval/results_probe_matched_L4_seeds.json").write_text(json.dumps(out, indent=2))
    print(f"valid seeds {VALID}  (excluded, ds-failed: {EXCLUDED})")
    for name in ARMS:
        e, m = per[name]["edge_auc"], per[name]["mse_f"]
        if e:
            print(f"  {name:6s} edge {e['mean']:.3f}+/-{e['sd']:.3f}  mse_f {m['mean']:.3f}+/-{m['sd']:.3f}")
    if out["ds_minus_lat_msef"]:
        g = out["ds_minus_lat_msef"]
        print(f"ds-lat mse_f gap (VALID): {g['mean']:+.3f} +/- {g['sd']:.3f}  (n={g['n']})")
    if out["ds_minus_lat_msef_itt"]:
        gi = out["ds_minus_lat_msef_itt"]
        print(f"ds-lat mse_f gap (ITT):   {gi['mean']:+.3f} +/- {gi['sd']:.3f}  (n={gi['n']})")
    print(f"excluded-seed per-arm mse_f: {out['excluded_all_arms_mse_f']}  | lat/ds ratio range {out['lat_over_ds_ratio_range']}")
    print("wrote eval/results_probe_matched_L4_seeds.json")


if __name__ == "__main__":
    main()
