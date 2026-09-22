"""Re-run train.train_jepa.validate on a saved JEPA checkpoint with its own config (CPU): the C1 within-table
instrument and the C12 val_ppd / val_total on runs trained before those were logged.

Run: uv run --with pyarrow python -m eval.revalidate RUN [RUN ...] [val_batches=N]  -> eval/results_revalidate.json (merged by run);
     each record also carries the run's last logged val_jepa and whether the re-run reproduces it exactly
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.factor_value_mse import _load_dual      # noqa: E402
from train.train_jepa import DEFAULTS, validate   # noqa: E402


# REVAL_OUT=eval/results_revalidate_vb16.json for a re-validation at a different val_batches (the 8-batch file keeps
# the "reproduces the logged val_jepa" check; the 16-batch file makes pre-C12 runs comparable with v4 runs' logs)
OUT = ROOT / os.environ.get("REVAL_OUT", "eval/results_revalidate.json")


def main():
    runs = [a for a in sys.argv[1:] if "=" not in a]
    over = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for run in runs:
        model, cfg = _load_dual(run)
        cfg = {**DEFAULTS, **cfg, **{k: int(v) for k, v in over.items()}}
        torch.manual_seed(cfg["seed"])
        rec = validate(model, cfg, "cpu")
        logged = [json.loads(l) for l in open(ROOT / "runs" / run / "metrics.jsonl") if '"val_jepa"' in l]
        rec.update(val_batches=cfg["val_batches"], logged_val_jepa_last=logged[-1]["val_jepa"] if logged else None)
        rec["val_jepa_matches_log"] = rec["logged_val_jepa_last"] == rec["val_jepa"]
        res[run] = rec
        print(json.dumps({"run": run, **rec}), flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
