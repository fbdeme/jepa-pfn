"""Deterministic difficulty ladder for the nonlinearity-ceiling study (design doc:
docs/scm_nonlinearity_ceiling.md). Isolates mechanism nonlinearity: every rung holds the graph,
n_rows, n_cols, and sigma at the base 'toy' prior and varies ONLY (mlp_depth, mlp_gain) -- so a
change in GBM-over-linear headroom is attributable to nonlinearity alone, not to large-N/width
(unlike real_ds_s1, which also widened n_cols/n_rows). CPU-only, seed 0, so the ladder is
bit-reproducible. The GPU learnability smoke (later) reads this JSON and appends per-rung val_mse.

Run:  uv run python -m eval.nonlinearity_ceiling_sweep
Out:  eval/results_nonlinearity_ceiling.json  (source-backed; no hand-typed headrooms)
"""
import json
from pathlib import Path

import numpy as np

from eval.prior_difficulty import score_prior

ROOT = Path(__file__).parents[1]

# Base 'toy' prior held fixed across rungs = base_ds.yaml's ACTUAL training regime (wide-D,
# long-context), NOT pure PriorConfig defaults -- so L0 reproduces the trained model's prior
# (headroom ~.023) and the ladder isolates nonlinearity within the regime the base model really
# sees. sigma held at base too, so a headroom change is (depth, gain) alone.
BASE = dict(n_rows=[128, 384], n_cols=[4, 32], sigma=[0.01, 0.3])

# The ladder: mechanism nonlinearity only. (depth, gain) monotone-ish in expected difficulty.
LADDER = [
    ("L0_toy", 1, 1.0),   # == base_ds paper prior (anchor; headroom ~.023 measured)
    ("L1", 2, 1.5),
    ("L2", 2, 2.0),       # ~ real_ds_s1's depth/gain, but WITHOUT its n_cols/n_rows widening
    ("L3", 3, 2.0),
    ("L4", 3, 3.0),
    ("L5", 4, 3.0),
]


def main():
    rungs = []
    for name, depth, gain in LADDER:
        cfg = dict(BASE, mlp_depth=[depth, depth], mlp_gain=gain)
        s = score_prior(cfg, n_tables=200, seed=0)
        g, r = np.clip(s["gbm"], -1, 1), np.clip(s["ridge"], -1, 1)
        rec = dict(
            rung=name, mlp_depth=depth, mlp_gain=gain,
            n_tables=int(s["n"]),
            headroom_median=round(float(np.median(g - r)), 4),   # nonlinearity worth learning
            gbm_r2_median=round(float(np.median(g)), 4),
            ridge_r2_median=round(float(np.median(r)), 4),
            frac_solved=round(float((g > 0.95).mean()), 4),      # want << 1.0
        )
        rungs.append(rec)
        print(f"{name:8s} depth={depth} gain={gain}  headroom {rec['headroom_median']:+.3f}  "
              f"gbm {rec['gbm_r2_median']:+.3f}  solved {rec['frac_solved']:.2f}")
    out = ROOT / "eval" / "results_nonlinearity_ceiling.json"
    payload = dict(base_prior=BASE, seed=0, n_tables=200, rungs=rungs)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
