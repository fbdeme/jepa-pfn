"""Run this paper's read-out protocol on public TabPFN v2 weights, inference only.

The paper claims the instrument and its read-out discipline are reusable for comparing
tabular in-context learners. Every measurement backing that claim so far comes from
encoders we trained ourselves, so the claim and its evidence share a failure mode: the
value-retention pathology could be a property of our training setup rather than of the
read-out. This runs the same protocol on a frozen public model we did not train.

TabPFN v2 admits both of our read-outs, and one of them is exact:

  HIDDEN   (the masked read-out) the target column is the regression target only. The
           model never sees its value for the rows we probe.
  VISIBLE  (the unmasked read-out) the SAME rows, the SAME model, with the observed
           target value appended as an extra feature column. The two conditions differ
           in exactly one thing -- whether the target cell was visible at encoding --
           which is the contrast Section 3.3 defines.

  ROLE     a second, architecture-native instance that costs nothing: context-row
           embeddings come from rows whose target value was given to the model as
           context, query-row embeddings from rows where it was not. Different rows,
           so this is an analogue rather than a replication, and it is reported as one.

Probe, targets, clamp and ridge are this paper's: predict the noiseless f of the target
cell from the frozen 192-d row embedding, ridge at RIDGE_ALPHA, fit on the first half of
the pooled stream and test on the second half. Pooling is table-major, so the fit and
test halves are disjoint tables and the probe must generalize across them.

Weights: the locally cached `tabpfn-v2-regressor.ckpt`, loaded via `model_path` so no
download is attempted. n_estimators=1 -- a single deterministic forward pass, not the
ensemble the package predicts with, because we read a representation and not a prediction.

Run: uv run --with tabpfn==2.1.0 python -m eval.tabpfn_readout
Writes eval/results_tabpfn_readout.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import EVAL_SEED, scm_prior            # noqa: E402
from eval.truth_eval import RIDGE_ALPHA, ridge_mse                # noqa: E402
from prior.scm import FuncFamily                                  # noqa: E402
from train.data import make_batch                                 # noqa: E402

OUT = ROOT / "eval/results_tabpfn_readout.json"
CKPT = Path(os.path.expanduser("~/.cache/tabpfn/tabpfn-v2-regressor.ckpt"))
CTX, NQ = 128, 50
CLAMP = 3.05
N_TABLES = int(os.environ.get("N_TABLES", "12"))
N_TARGETS = int(os.environ.get("N_TARGETS", "3"))   # target columns probed per table


def _regressor():
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor(n_estimators=1, device="cpu", random_state=0,
                           model_path=str(CKPT))


def _embed(X_fit, y_fit, X_probe):
    """Frozen row embeddings for the probe rows, plus the context rows, from one fit."""
    r = _regressor()
    r.fit(X_fit, y_fit)
    q = np.asarray(r.get_embeddings(X_probe, data_source="test"))[0]
    c = np.asarray(r.get_embeddings(X_probe, data_source="train"))[0]
    return q, c


def collect():
    """One pass over the instrumented prior, returning pooled (embedding, f) pairs."""
    torch.manual_seed(EVAL_SEED)
    prior = scm_prior(1.0, CTX, NQ, (4, 32))
    rng = np.random.default_rng(EVAL_SEED)
    acc = {k: ([], []) for k in ("hidden", "visible", "role_query", "role_context")}
    n_pairs = 0
    t0 = time.time()
    for tb in range(N_TABLES):
        bt = make_batch(prior, 1, "any_cell", rng, split=CTX, return_truth=True)
        z, zf = bt["z"][0].numpy(), bt["z_f"][0].numpy()
        fam = bt["cell_family"][0].numpy()
        cols = [d for d in range(z.shape[1]) if fam[d] != int(FuncFamily.ROOT)]
        for d in cols[:N_TARGETS]:
            others = [j for j in range(z.shape[1]) if j != d]
            Xo, y = z[:, others], z[:, d]
            f_q = np.clip(zf[CTX:, d], -CLAMP, CLAMP)
            f_c = np.clip(zf[:CTX, d], -CLAMP, CLAMP)

            # HIDDEN: the target column is the regression target and nothing else.
            q, c = _embed(Xo[:CTX], y[:CTX], Xo[CTX:])
            acc["hidden"][0].append(q);        acc["hidden"][1].append(f_q)
            acc["role_query"][0].append(q);    acc["role_query"][1].append(f_q)
            acc["role_context"][0].append(c);  acc["role_context"][1].append(f_c)

            # VISIBLE: same rows, same model, target cell appended as a feature.
            Xv = np.hstack([Xo, y[:, None]])
            qv, _ = _embed(Xv[:CTX], y[:CTX], Xv[CTX:])
            acc["visible"][0].append(qv);      acc["visible"][1].append(f_q)

            n_pairs += 1
            print(f"  table {tb:>3} col {d:>3}  ({n_pairs} fits x2, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in acc.items()}, n_pairs


def const_map(y):
    """The encoder-free floor on the same targets and the same half/half split."""
    n = len(y) // 2
    return float(np.mean((y[n:] - y[:n].mean()) ** 2))


def main():
    if not CKPT.exists():
        sys.exit(f"missing {CKPT}: this script does not download weights")
    acc, n_pairs = collect()
    out = {"model": "TabPFN v2 regressor (public weights, frozen, inference only)",
           "checkpoint": CKPT.name, "n_estimators": 1, "embedding_dim": 192,
           "protocol": {"source": "this paper: ridge on the frozen representation to the "
                                  "noiseless f of the target cell, alpha "
                                  f"{RIDGE_ALPHA}, clamp {CLAMP}, fit on the first half "
                                  "of a table-major pooled stream and test on the second",
                        "hidden": "masked read-out: target column is the regression "
                                  "target only, never seen for the probed rows",
                        "visible": "unmasked read-out: SAME rows and model, observed "
                                   "target value appended as an extra feature column",
                        "role_query/role_context": "architecture-native analogue: query "
                                                   "rows vs context rows, whose target "
                                                   "value the model was given. Different "
                                                   "rows, so an analogue not a replication",
                        "prior": "instrumented SCM prior, sigma 1.0, cols 4-32, non-ROOT "
                                 "targets", "ctx": CTX, "n_query": NQ,
                        "n_tables": N_TABLES, "n_target_columns": n_pairs},
           "readout": {}}
    for k, (X, y) in acc.items():
        out["readout"][k] = {"f_mse": round(ridge_mse(X, y), 4),
                             "const_map": round(const_map(y), 4), "n_cells": int(len(y))}
    h, v = out["readout"]["hidden"], out["readout"]["visible"]
    out["same_row_contrast"] = {
        "visible_minus_hidden": round(v["f_mse"] - h["f_mse"], 4),
        "reading": "negative means the unmasked read-out scores better on the same rows "
                   "because the answer is in the input, not because more mechanism was "
                   "recovered -- the pathology Section 3.3 names, on a model we did not train"}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for k, r in out["readout"].items():
        print(f"  {k:<14} f-MSE {r['f_mse']:.4f}   const map {r['const_map']:.4f}   "
              f"n {r['n_cells']}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
