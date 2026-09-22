"""Aggregate the C4 positive-control multi-seed probe into mean +/- SD.

Reads eval/results_probe_c4_s{0..3}.json (each = ds/dual/lat/random_init mse_f on the factor
substrate, lat at its per-seed pre-collapse steelman) and writes eval/results_probe_c4_seeds.json:
per-arm mean/SD of mse_f, and the arm-minus-ds gaps (negative = beats data-space). This is a narrow
DIRECTIONAL check on the instrument, not a magnitude result: on a substrate with a shared low-rank
latent cause + noise, the latent-AUGMENTED arm (dual) recovers mechanism best and is the only arm
below the random-projection floor, while pure latent still collapses. We take NO cross-substrate
ratio from it -- the factor and SCM f-MSE scales are not commensurate, and a seeded SCM dual-ds does
not reproduce the single-run base-probe sign (see eval/results_probe_seeds.json). Source-backed.

Reproduce (per seed s): train ds/dual/lat with run_name=c4_factor_<arm>_s<s>, then
  STEELMAN_GLOB=c4_factor_lat_s<s> STEELMAN_TAG=c4_s<s> eval.pick_steelman
  PROBE_RUNS=c4_factor_ds_s<s>,c4_factor_dual_s<s>,c4_factor_lat_s<s> PROBE_TAG=c4_s<s> PROBE_PRIOR=factor eval.probe_base
Run: uv run python -m eval.c4_seeds_agg
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
SEEDS = range(4)
ARMS = {"ds": "c4_factor_ds", "dual": "c4_factor_dual", "lat": "c4_factor_lat",
        "arbaux": "c4_factor_arbaux", "floor": "random_init"}


def _load(s):
    p = ROOT / f"eval/results_probe_c4_s{s}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _lat_full():
    """Per-seed lat from the completed 20k reruns (results_c4_lat_full.json):
    steelman (pre-collapse peak) for collapsed seeds, final ckpt for un-collapsed.
    Replaces the truncated ~200-step companion runs (review C9)."""
    d = json.loads((ROOT / "eval/results_c4_lat_full.json").read_text())
    vals = []
    for s in SEEDS:
        e = d.get(f"c4_factor_lat_s{s}_full")
        if e:
            vals.append(e.get("mse_f_steelman", e["mse_f_final"]))
    return np.array(vals, float)


def _floor_band():
    """30-draw seeded random-init floor (results_c4_floor_seeds.json);
    replaces the single seed-999 draw replicated across per-seed files."""
    d = json.loads((ROOT / "eval/results_c4_floor_seeds.json").read_text())
    return np.array(d["draws"], float)


def _series(arm_key):
    if arm_key == "c4_factor_lat":
        return _lat_full()
    if arm_key == "random_init":
        return _floor_band()
    vals = []
    for s in SEEDS:
        d = _load(s)
        if d is None:
            continue
        key = f"{arm_key}_s{s}"
        if key in d:
            vals.append(d[key]["mse_f"]["1.0"])
    return np.array(vals, float)


def _stat(a):
    return (dict(mean=round(float(a.mean()), 4), sd=round(float(a.std()), 4), n=int(len(a)))
            if len(a) else None)


def _paired_gap(arm_key):
    """arm - ds, paired per seed (negative => arm beats data-space)."""
    gaps = []
    for s in SEEDS:
        d = _load(s)
        if d is None:
            continue
        ds = f"c4_factor_ds_s{s}"
        if arm_key == "c4_factor_lat":
            full = json.loads((ROOT / "eval/results_c4_lat_full.json").read_text())
            e = full.get(f"c4_factor_lat_s{s}_full")
            if e and ds in d:
                gaps.append(e.get("mse_f_steelman", e["mse_f_final"]) - d[ds]["mse_f"]["1.0"])
            continue
        a = f"{arm_key}_s{s}"
        if a in d and ds in d:
            gaps.append(d[a]["mse_f"]["1.0"] - d[ds]["mse_f"]["1.0"])
    return np.array(gaps, float)


def main():
    present = [s for s in SEEDS if _load(s) is not None]
    per = {name: dict(mse_f=_stat(_series(key))) for name, key in ARMS.items()}
    out = dict(
        note="C4 instrument directional check, factor substrate (sigma 1.0), re-baselined to a fresh "
             "self-consistent run (models on HF). mse_f = rep->f recovery (lower=better); lat = completed 20k reruns (steelman for collapsed seeds, final for un-collapsed; results_c4_lat_full.json); floor = 30-draw seeded band (results_c4_floor_seeds.json); "
             "gap = arm - ds. dual is the only arm below the random floor. ARBAUX = dual recipe with the "
             "latent target SHUFFLED across cells: it equals ds (arbaux-ds ~= 0), so dual's gain is "
             "specific to the aligned latent target, not generic auxiliary regularization.",
        seeds_present=present, n=len(present),
        per_arm=per,
        dual_minus_ds=_stat(_paired_gap("c4_factor_dual")),       # NEGATIVE: aligned latent target beats ds
        lat_minus_ds=_stat(_paired_gap("c4_factor_lat")),         # ~0/mixed: pure latent ties/loses
        arbaux_minus_ds=_stat(_paired_gap("c4_factor_arbaux")),   # ~0: SHUFFLED target = no benefit over ds
    )
    (ROOT / "eval/results_probe_c4_seeds.json").write_text(json.dumps(out, indent=2))
    print(f"seeds present: {present}")
    for name in ARMS:
        m = per[name]["mse_f"]
        if m:
            print(f"  {name:6s} mse_f {m['mean']:.3f}+/-{m['sd']:.3f}  (n={m['n']})")
    for lbl, k in (("dual-ds", "dual_minus_ds"), ("lat-ds", "lat_minus_ds"), ("arbaux-ds", "arbaux_minus_ds")):
        g = out[k]
        if g:
            tag = "<- dual BEATS ds (aligned target)" if k == "dual_minus_ds" and g["mean"] < 0 else (
                  "<- SHUFFLED = no benefit" if k == "arbaux_minus_ds" else "")
            print(f"{lbl:9s} gap: {g['mean']:+.3f} +/- {g['sd']:.3f}  (n={g['n']})  {tag}")
    print("wrote eval/results_probe_c4_seeds.json")


if __name__ == "__main__":
    main()
