"""Re-score specific OpenML dataset ids into an existing suite pass (a transient download error skips a dataset in
eval/suite_bench.py; this fills the hole with the same models / env: PFN_RUNS, K_FEAT, CTX, DEVICE, SUITE_TAG).
Run (on the box, after the main pass): PFN_RUNS=... SUITE_TAG=real_e9 uv run --with openml python -m eval.suite_redo BENCH:DID [...]
  e.g.  python -m eval.suite_redo TabArena:46908"""
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.suite_bench import (PFN_RUNS, ROOT, HistGradientBoostingClassifier, HistGradientBoostingRegressor,   # noqa: E402
                              LogisticRegression, PFNClassifier, PFNRegressor, Ridge, eval_dataset, load_pfn)


def main():
    warnings.filterwarnings("ignore")
    pfns = {r: load_pfn(r) for r in PFN_RUNS if (ROOT / "runs" / r / "ckpt.pt").exists()}
    models_clf = {r: (lambda m=m: PFNClassifier(m)) for r, m in pfns.items()}
    models_clf["logreg"] = lambda: LogisticRegression(max_iter=2000)
    models_clf["histgb"] = lambda: HistGradientBoostingClassifier(random_state=0)
    models_reg = {r: (lambda m=m: PFNRegressor(m)) for r, m in pfns.items()}
    models_reg["ridge"] = lambda: Ridge()
    models_reg["histgbr"] = lambda: HistGradientBoostingRegressor(random_state=0)
    outpath = ROOT / f"eval/results_suite_{os.environ.get('SUITE_TAG', 'full')}.json"
    results = json.loads(outpath.read_text()) if outpath.exists() else {}
    for arg in sys.argv[1:]:
        bench, did = arg.split(":")
        name, kind, out = eval_dataset(int(did), models_clf, models_reg)
        results[f"{bench}|{name}|{kind}"] = out
        print(f"  [{kind}] {name} did={did}  " + " ".join(f"{m}={out[m]['r2' if kind == 'reg' else 'acc']}" for m in out), flush=True)
        outpath.write_text(json.dumps(results, indent=1))
    print(f"wrote {outpath}  ({len(results)} datasets)")


if __name__ == "__main__":
    main()
