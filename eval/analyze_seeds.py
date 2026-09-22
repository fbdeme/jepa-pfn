"""Aggregate results_probe_seeds.json into a seed-CI summary (deterministic).

Turns the Table-2 "n=1" numbers into mean +/- spread over 5 trained seeds and a
30-draw random-init noise-floor band, then reports each arm's edge vs the floor with
a seed-paired-style combined spread. Read-only: prints a summary + writes
eval/results_probe_seeds_summary.json (per-metric aggregates, source-backed).

Run: uv run python -m eval.analyze_seeds
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
D = json.load(open(ROOT / "eval/results_probe_seeds.json"))

METRICS = [("edge_auc", "edge AUC", 0.50),
           ("fam_acc", "family acc", None),
           ("mse_f1", "f-MSE@1", None)]


def val(rec, key):
    return rec["mse_f"]["1.0"] if key == "mse_f1" else rec[key]


def agg(records, key):
    xs = [val(r, key) for r in records]
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else 0.0
    return dict(mean=m, sd=sd, sem=sd / (n ** 0.5) if n else 0.0, n=n,
                lo=min(xs), hi=max(xs))


def fmt(a):
    return f"{a['mean']:.4f} +/- {a['sd']:.4f} (n={a['n']}, [{a['lo']:.4f},{a['hi']:.4f}])"


def main():
    arms = ["ds", "dual", "lat_s"]
    floor = D["floor"]
    summary = {"arms": {}, "floor": {}, "vs_floor": {}}
    print(f"== seed-CI probe summary ==  seeds={D['protocol']['seeds']} "
          f"floor_draws={D['protocol']['n_floor']}\n")
    for key, label, chance in METRICS:
        print(f"-- {label}" + (f" (chance {chance})" if chance else "") + " --")
        fa = agg(floor, key)
        summary["floor"][key] = fa
        for arm in arms:
            a = agg(D["trained"][arm], key)
            summary["arms"].setdefault(arm, {})[key] = a
            # arm vs floor: difference of means, spread combined in quadrature
            diff = a["mean"] - fa["mean"]
            comb_sd = (a["sd"] ** 2 + fa["sd"] ** 2) ** 0.5
            summary["vs_floor"].setdefault(arm, {})[key] = dict(
                diff=diff, comb_sd=comb_sd, z=diff / comb_sd if comb_sd else 0.0)
            print(f"  {arm:6} {fmt(a)}   vs floor: {diff:+.4f} (comb sd {comb_sd:.4f})")
        print(f"  {'floor':6} {fmt(fa)}\n")
    out = ROOT / "eval/results_probe_seeds_summary.json"
    out.write_text(json.dumps(summary, indent=1, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
