"""Real-data suite: per-dataset paired comparison of one run against reference runs (E9 protocol: CC18 + Grinsztajn +
TabArena, K_FEAT 64, CTX 1024). Exact two-sided sign test on per-dataset wins, plus mean metric. Reads the E9 pass and
the reference ds pass for the references, and a new pass for the run under test.
Usage: uv run python suite_compare.py RUN NEW_PASS.json [REF_RUN ...]   -> prints + writes results_suite_compare.json"""
import json, sys
from math import comb
from pathlib import Path
from statistics import mean
D = Path(__file__).parent; ROOT = D.parents[1]
run, newp = sys.argv[1], Path(sys.argv[2])
refs = sys.argv[3:] or ["real_ds_lr5e-4_s0", "v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0", "histgb"]
passes = [json.loads(p.read_text()) for p in (newp, ROOT / "eval/results_suite_real_e9.json", ROOT / "eval/results_suite_real_ds.json",
                                               ROOT / "eval/results_suite_real_dualsplit.json") if p.exists()]   # dualsplit: the 20k pass (2026-09-09, long-run snapshots compare against it)
def score(model, key):
    for p in passes:
        e = p.get(key, {}).get(model)
        if e is not None: return e
    return None
def sign_p(w, l):
    n = w + l
    if n == 0: return None
    k = min(w, l); return round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 5)
keys = sorted(k for k in passes[0] if run in passes[0][k])
out = {"run": run, "n_datasets": len(keys), "vs": {}}
print(f"{run}: {len(keys)} datasets scored")
for ref in refs:
    rec = {}
    for task, metric in (("clf", "acc"), ("clf", "auc"), ("reg", "r2")):
        ks = [k for k in keys if k.endswith("|" + task)]
        pairs = [(score(run, k)[metric], score(ref, k)[metric]) for k in ks if score(run, k) and score(ref, k) and metric in score(run, k) and metric in score(ref, k)]
        if not pairs: continue
        w = sum(a > b for a, b in pairs); l = sum(a < b for a, b in pairs)
        rec[f"{task}_{metric}"] = {"n": len(pairs), "wins": w, "losses": l, "p_sign": sign_p(w, l),
                                   "mean_run": round(mean(a for a, _ in pairs), 4), "mean_ref": round(mean(b for _, b in pairs), 4),
                                   "mean_delta": round(mean(a - b for a, b in pairs), 4)}
        r = rec[f"{task}_{metric}"]
        print(f"  vs {ref:48} {task}/{metric:3} n={r['n']:3}  wins {r['wins']:3} / losses {r['losses']:3}  p={r['p_sign']}  mean {r['mean_run']:.4f} vs {r['mean_ref']:.4f} (delta {r['mean_delta']:+.4f})")
    out["vs"][ref] = rec
import os
OUT = Path(os.environ.get("COMPARE_OUT", D / "results_suite_compare.json"))   # override so other runs do not overwrite the (24d) artifact
OUT.write_text(json.dumps(out, indent=1))
print("wrote", OUT)
