"""The masked read-out scored with a NONLINEAR probe, on the headline recipe grid.

Round 9 raised an alignment objection that the paper's own citation creates. Every masked number
in this paper is fit with a ridge-linear probe, while \\citet{reizinger2025cross} -- which the
introduction cites in support of the data-space objective -- proves that cross-entropy training
recovers latents *up to a linear transformation*. A linear probe is therefore the read-out class
in which the data-space arm has a theoretical guarantee and the latent arms have none, so
"data-space encodes more recoverable mechanism" may be partly a statement about the probe.

`eval/probe_seeds_mlp.py` already answers the sibling objection ("a linear probe underestimates")
but only on the UNMASKED read-out, which §4.2.1 supersedes. This script runs the same shallow MLP
on the corrected masked read-out, so the two probe classes are compared on the read-out the paper
actually reports.

Protocol is `eval.probe_masked_all.masked_read` verbatim -- same prior, EVAL_SEED, policies,
non-ROOT target selection, clamp, half/half split -- with `mlp_mse` from eval.probe_seeds_mlp in
place of `ridge_mse`. The ridge value is recomputed alongside as a determinism check: it must
reproduce results_probe_masked_all.json exactly, since only the regressor changes.

Run: uv run python -m eval.probe_masked_mlp     (CPU)
Writes eval/results_probe_masked_mlp.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import (EVAL_SEED, MAX_N, PROTOCOL,          # noqa: E402
                                   load_auto, random_init)
from eval.probe_seeds_mlp import mlp_mse                                  # noqa: E402
from eval.truth_eval import ridge_mse                                     # noqa: E402
from prior.scm import FuncFamily                                          # noqa: E402
from train.data import make_batch                                         # noqa: E402

OUT = ROOT / "eval/results_probe_masked_mlp.json"
POLICY = "any_cell"                 # the data-space arms' own policy: the harder test for them
FAMILY = "recipe"                   # the headline grid
ARMS = {"ds": [f"base_ds_s{i}" for i in range(5)],
        "dual": [f"base_dual_s{i}" for i in range(5)],
        "lat": [f"base_lat_s_s{i}" for i in range(5)],
        "ema_nohead": [f"base_ema_nohead_s{i}" for i in range(5)]}
N_FLOOR = 3


@torch.no_grad()
def _features(enc, prior_fn, ctx, n_batches, batch):
    """Exactly eval.probe_masked_all.masked_read's feature extraction, cell read-out only."""
    torch.manual_seed(EVAL_SEED)
    prior = prior_fn()
    rng = np.random.default_rng(EVAL_SEED)
    K = getattr(enc, "n_cls", 0) or 0
    cell, f_ = [], []
    for _ in range(n_batches):
        bt = make_batch(prior, batch, POLICY, rng, split=ctx, return_truth=True)
        tm = bt["target_mask"]
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        hc = h[:, :, K:] if K else h
        nonroot = (bt["cell_family"] != int(FuncFamily.ROOT)).unsqueeze(1).expand_as(tm)
        sel = tm & nonroot
        if not sel.any():
            continue
        b, r, d = sel.nonzero(as_tuple=True)
        cell.append(hc[b, r, d])
        f_.append(bt["z_f"][b, r, d].clamp(-3.05, 3.05))
    return torch.cat(cell).numpy()[:MAX_N], torch.cat(f_).numpy()[:MAX_N]


def main():
    prior_fn, ctx, n_batches, batch = PROTOCOL[FAMILY]
    out = {"probe": {"kind": "mlp", "hidden": [256], "read_out": "masked cell",
                     "policy": POLICY, "family": FAMILY},
           "note": "nonlinear counterpart of the masked ridge probe. `mse_f_ridge_check` must "
                   "reproduce results_probe_masked_all.json exactly: only the regressor differs.",
           "arms": {}, "floor": {}}
    t0 = time.time()
    for group, runs in ARMS.items():
        for run in runs:
            try:
                enc = load_auto(run)
            except FileNotFoundError:
                print(f"  {run:22} MISSING", flush=True)
                continue
            X, y = _features(enc, prior_fn, ctx, n_batches, batch)
            rec = {"group": group, "mse_f_mlp": round(mlp_mse(X, y), 4),
                   "mse_f_ridge_check": round(ridge_mse(X, y), 4), "n": int(len(X))}
            out["arms"][run] = rec
            print(f"  {run:22} mlp={rec['mse_f_mlp']:.4f}  ridge={rec['mse_f_ridge_check']:.4f}"
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    for s in range(N_FLOOR):
        X, y = _features(random_init(ARMS["ds"][0], 1234 + s), prior_fn, ctx, n_batches, batch)
        out["floor"][f"draw{s}"] = {"mse_f_mlp": round(mlp_mse(X, y), 4),
                                    "mse_f_ridge_check": round(ridge_mse(X, y), 4)}
        print(f"  floor draw {s}          mlp={out['floor'][f'draw{s}']['mse_f_mlp']:.4f}",
              flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
