"""CPU-vs-GPU E9 device check (2026-09-13): the same checkpoint scored by eval/suite_bench.py on the local CPU and on the
server GPU. Per-dataset deltas of the run's metrics; the sklearn baselines (logreg/histgb/ridge/histgbr) are scored on CPU
in both passes and must agree exactly, which also proves both passes saw the same datasets/splits.
Usage: python device_check.py RUN PASS_CPU.json PASS_GPU.json   -> prints + writes device_check.json next to this file"""
import json, sys
from pathlib import Path

run, a, b = sys.argv[1], json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
keys = sorted(k for k in a if "|" in k and k in b and run in a[k] and run in b[k])
rows, base_mismatch = [], 0
for k in keys:
    m = "r2" if k.endswith("|reg") else "acc"
    rows.append(dict(key=k, metric=m, cpu=a[k][run][m], gpu=b[k][run][m], delta=round(b[k][run][m] - a[k][run][m], 4)))
    for bl in ("logreg", "histgb", "ridge", "histgbr"):
        if bl in a[k] and bl in b[k] and a[k][bl] != b[k][bl]:
            base_mismatch += 1
d = [abs(r["delta"]) for r in rows]
out = dict(run=run, n=len(rows), baseline_cells_mismatch=base_mismatch, max_abs_delta=max(d), mean_abs_delta=round(sum(d) / len(d), 5),
           n_gt_005=sum(x > .005 for x in d), n_gt_01=sum(x > .01 for x in d), n_gt_02=sum(x > .02 for x in d),
           gpu_minus_cpu_mean=round(sum(r["delta"] for r in rows) / len(rows), 5), rows=rows)
json.dump(out, open(Path(__file__).with_name("device_check.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "rows"})
print("largest:", sorted(rows, key=lambda r: -abs(r["delta"]))[:5])
