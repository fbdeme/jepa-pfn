"""Extract the data-space arm's held-out value MSE across the K sweep into a committed artifact.

Round 8 asked whether the data-space failure on the shared-factor substrate is a fact about
compressibility. It is not: the arm never learns at ANY rank, including the ranks where nothing
is compressible. That is what `eval/factor_oracle.py:38-42` already predicted for sigma=1.0
("the transformer ds stuck at val_mse 1.0, SNR 1, value prediction noise-limited"), and the runs
confirm it at every K.

`runs/` is gitignored, so the claim has to live in a committed file the way the collapse curve
does (eval/extract_collapse_curve.py). Targets are unit-variance, so val_mse ~ 1.0 is exactly the
score of predicting the mean.

Run: uv run python -m eval.extract_ds_training_curve
Writes eval/results_ds_training_curve.json
"""
import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_ds_training_curve.json"
LEARNED_TOL = 0.1     # val_mse must fall this far below its start to count as having learned
DIVERGED_TOL = 0.1    # ...and rise this far back above its own best to count as diverged


def _runs():
    for d in sorted((ROOT / "runs").glob("c4_factor_ds*")):
        if "_lr" in d.name:   # C1 learning-rate controls live in extract_factor_lr_control.py
            continue
        m = re.search(r"_K(\d+)_", d.name)
        yield (int(m.group(1)) if m else 4), d


def main():
    by_k, missing = {}, []
    for k, d in _runs():
        p = d / "metrics.jsonl"
        if not p.exists():
            missing.append(d.name)
            continue
        vals = [(r["step"], r["val_mse"]) for r in
                (json.loads(l) for l in p.read_text().splitlines() if l.strip())
                if "val_mse" in r]
        if len(vals) < 2:
            missing.append(d.name)
            continue
        best_step, best_val = min(vals, key=lambda t: t[1])
        # Round-9 devil's advocate: classifying on first-vs-last calls a run that learned and then
        # lost it "never moved". Record the peak and whether the run ended above it, because the
        # probe reads the FINAL checkpoint and the latent arms are read at a pre-collapse peak.
        by_k.setdefault(k, []).append({"run": d.name, "n_val": len(vals),
                                       "first_step": vals[0][0], "last_step": vals[-1][0],
                                       "val_mse_first": round(vals[0][1], 4),
                                       "val_mse_last": round(vals[-1][1], 4),
                                       "val_mse_best": round(best_val, 4),
                                       "best_step": best_step,
                                       "learned": bool(vals[0][1] - best_val > LEARNED_TOL),
                                       "diverged": bool(vals[-1][1] - best_val > DIVERGED_TOL)})
    out = {"metric": "val_mse -- held-out value MSE; targets are unit variance, so 1.0 is the "
                     "score of predicting the mean",
           "arm": "c4_factor_ds (data-space) on the shared-factor substrate at sigma=1.0",
           "note": "reported per intrinsic rank K; if the failure were about compressibility it "
                   "would ease at high K, where the target is least compressible",
           "missing_runs": missing, "by_k": {}}
    for k in sorted(by_k):
        rs = by_k[k]
        out["by_k"][str(k)] = {
            "n_runs": len(rs),
            "val_mse_first_mean": round(st.mean(r["val_mse_first"] for r in rs), 4),
            "val_mse_last_mean": round(st.mean(r["val_mse_last"] for r in rs), 4),
            "val_mse_best_mean": round(st.mean(r["val_mse_best"] for r in rs), 4),
            "improvement_mean": round(st.mean(r["val_mse_first"] - r["val_mse_last"] for r in rs), 4),
            "runs": rs}
    ks = sorted(out["by_k"], key=int)
    imps = [out["by_k"][k]["improvement_mean"] for k in ks]
    lasts = [out["by_k"][k]["val_mse_last_mean"] for k in ks]
    # A run "learned" only if held-out MSE moved by more than rounding noise. Six runs move by
    # exactly 1e-4, which is the printing grid, not learning; the tolerance separates them from
    # the two seeds that actually solve.
    TOL = 1e-3
    allruns = [r for k in ks for r in out["by_k"][k]["runs"]]
    learned = [r for r in allruns if abs(r["val_mse_first"] - r["val_mse_last"]) > TOL]
    ever = [r for r in allruns if r["learned"]]
    div = [r for r in ever if r["diverged"]]
    out["summary"] = {
        "k_values": [int(k) for k in ks],
        "improvement_range": [min(imps), max(imps)],
        "val_mse_last_range": [min(lasts), max(lasts)],
        "max_abs_improvement": round(max(abs(i) for i in imps), 4),
        "eases_at_high_k": imps[-1] > imps[0] + 0.05,
        "learning_tolerance": TOL,
        "n_runs": len(allruns),
        "n_no_learning": len(allruns) - len(learned),
        "learned_runs": [r["run"] for r in learned],
        "learned_k_values": sorted({int(re.search(r"_K(\d+)_", r["run"]).group(1))
                                    if "_K" in r["run"] else 4 for r in learned}),
        # the reading that survives the round-9 objection: how many runs ever learned, and how
        # many gave it back before the checkpoint the probe reads
        "n_ever_learned": len(ever),
        "n_learned_then_diverged": len(div),
        "ever_learned_k_values": sorted({int(re.search(r"_K(\d+)_", r["run"]).group(1))
                                         if "_K" in r["run"] else 4 for r in ever}),
        "diverged_runs": [r["run"] for r in div],
        "best_step_range": ([min(r["best_step"] for r in ever), max(r["best_step"] for r in ever)]
                            if ever else None),
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), f"({len(missing)} runs missing)")
    for k in ks:
        e = out["by_k"][k]
        print(f"  K={k:>2}  n={e['n_runs']}  val_mse {e['val_mse_first_mean']:.4f} -> "
              f"{e['val_mse_last_mean']:.4f}  (best {e['val_mse_best_mean']:.4f}, "
              f"improvement {e['improvement_mean']:+.4f})")
    print(f"  eases at high K: {out['summary']['eases_at_high_k']}  "
          f"max |improvement| = {out['summary']['max_abs_improvement']}")


if __name__ == "__main__":
    main()
