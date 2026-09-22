"""Trivial baselines on the class-head arms' own validation streams (docs/real_prior_plan.md 11 / 11.2).

For each seed the trainer validates on make_batch(y_target) from _make_prior(cfg, 10000+seed, "val") with the same rng, so the
arms' val_err (cell-weighted error at the label cells of query rows) is directly comparable with:
  majority : predict the context rows' majority class
  logreg   : per-table logistic regression on the standardised context rows (an in-context linear reference)
Output eval/results_cls_val_baselines.json. Run: uv run --with scikit-learn python -m eval.cls_val_baselines
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression   # noqa: E402
from train.data import make_batch                     # noqa: E402
from train.train import _make_prior                   # noqa: E402

OUT = ROOT / "eval/results_cls_val_baselines.json"


def baselines(cfg_path):
    cfg = yaml.safe_load(cfg_path.read_text())
    val_seed = 10_000 + cfg["seed"]
    prior = _make_prior(cfg, val_seed, "val"); rng = np.random.default_rng(val_seed)
    maj = lr = n = tables = binary = 0
    for _ in range(cfg.get("val_batches", 8)):
        bt = make_batch(prior, cfg["batch_size"], "y_target", rng, "cpu")
        z = bt["z"].numpy(); tm = bt["target_mask"].numpy(); y = bt["y_ids"].numpy(); S = bt["split"]
        for i in range(z.shape[0]):
            if not tm[i].any():
                continue
            t = int(np.flatnonzero(tm[i].any(0))[0]); yq, yc = y[i, S:, t], y[i, :S, t]
            ok, okc = yq >= 0, yc >= 0
            if ok.sum() == 0 or okc.sum() == 0:
                continue
            vals, cnt = np.unique(yc[okc], return_counts=True); m = vals[cnt.argmax()]
            maj += int((yq[ok] != m).sum()); n += int(ok.sum()); tables += 1; binary += int(len(vals) == 2)
            if len(vals) >= 2:
                clf = LogisticRegression(max_iter=300).fit(np.delete(z[i, :S], t, 1)[okc], yc[okc])
                lr += int((clf.predict(np.delete(z[i, S:], t, 1)[ok]) != yq[ok]).sum())
            else:
                lr += int((yq[ok] != m).sum())
    return {"val_seed": val_seed, "tables": tables, "binary_tables": binary, "target_cells": n,
            "majority_err": round(maj / n, 4), "logreg_err": round(lr / n, 4)}


def main():
    out = {"note": "cell-weighted error at label cells of query rows, same stream as the arms' val_err (train.train / train_jepa cls_err)",
           "seeds": {}}
    for seed in (0, 1, 2):
        out["seeds"][str(seed)] = baselines(ROOT / f"configs/cls_ds_lr5e-4_s{seed}.yaml")
        print(seed, out["seeds"][str(seed)])
    OUT.write_text(json.dumps(out, indent=1) + "\n"); print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
