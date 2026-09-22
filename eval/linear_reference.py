"""Two encoder-free linear references on the shared-factor substrate, at every K.

Round 8 asked what a linear model gets on this substrate before any pretraining. There are two
different answers and conflating them is the whole trap:

  * ACROSS-TABLE ("competitor"): one ridge fit over cells pooled from many tables, tested on
    held-out tables -- exactly the probe's own protocol. It is a fair competitor to a trained
    arm because it must generalize across tables, and it cannot succeed here: every table draws
    its own factor loadings, so no single linear map inverts them all. It lands at the constant
    map.
  * PER-TABLE ("upper reference"): a fresh ridge per (table, target column), fit on that table's
    own rows. It is NOT a competitor -- it reads ground-truth `f` for the rows it fits, which no
    trained arm ever sees (SSOT rule 2). It bounds what a linear model could extract from this
    substrate at all, which is what makes it useful as a normalizer.

The second is what `eval/factor_oracle.py` already computes as `r2_f`; this script reports it in
the probe's MSE units so the two are directly comparable.

Protocol is copied from eval/probe_masked_all.py: same prior constructor, EVAL_SEED, policies,
non-ROOT target selection, clamp, RIDGE_ALPHA, and the same half/half fit-test split.

Run: uv run python -m eval.linear_reference
Writes eval/results_linear_reference.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import (EVAL_SEED, MAX_N, POLICIES,   # noqa: E402
                                   factor_prior, scm_prior)
from eval.truth_eval import RIDGE_ALPHA, ridge_mse                          # noqa: E402
from prior.scm import FuncFamily                                            # noqa: E402
from train.data import make_batch                                           # noqa: E402

OUT = ROOT / "eval/results_linear_reference.json"
# Round-10 perspective: the headline NEGATIVE rests on a substrate with three lower anchors and
# no ceiling, while this script -- which supplies exactly that ceiling -- was run only on the
# other substrate. Run it on both.
SCM_KEY = "scm"
CTX, NQ, N_BATCHES, BATCH = 128, 50, 8, 8
KS = [1, 2, 4, 6, 8, 12]
CLAMP = 3.05


def _batches(k, policy):
    """k is the factor rank, or SCM_KEY for the single-cell prior the headline negative uses."""
    torch.manual_seed(EVAL_SEED)
    prior = (scm_prior(1.0, CTX, NQ, (4, 32)) if k == SCM_KEY else factor_prior(CTX, NQ, k))
    rng = np.random.default_rng(EVAL_SEED)
    for _ in range(N_BATCHES):
        yield make_batch(prior, BATCH, policy, rng, split=CTX, return_truth=True)


@torch.no_grad()
def across_table(k, policy):
    """One ridge over pooled cells, fit on the first half of the stream and tested on the rest --
    the probe's protocol with the row's observed values in place of an encoder embedding."""
    feats, tgt = [], []
    for bt in _batches(k, policy):
        sel = bt["target_mask"] & (bt["cell_family"] != int(FuncFamily.ROOT)).unsqueeze(1)
        if not sel.any():
            continue
        b, r, d = sel.nonzero(as_tuple=True)
        z = bt["z"].clone()
        z[bt["input_mask"]] = 0.0            # the encoder sees hidden cells as hidden; so does this
        onehot = torch.zeros(len(b), z.shape[2])
        onehot[torch.arange(len(b)), d] = 1.0
        feats.append(torch.cat([z[b, r], onehot], 1))
        tgt.append(bt["z_f"][b, r, d].clamp(-CLAMP, CLAMP))
    X = torch.cat(feats).numpy()[:MAX_N]
    return round(ridge_mse(X, torch.cat(tgt).numpy()[:MAX_N]), 4)


@torch.no_grad()
def per_table(k, policy):
    """A fresh ridge per (table, target column), fit on that table's own visible rows. Sees the
    ground-truth f of the rows it fits, so it is an upper reference and not a competitor."""
    errs = []
    for bt in _batches(k, policy):
        # Round 9 found, and round 10 found again, that this read RAW z while the trained arms
        # see [MASK] at the same cells -- so the "upper reference" was solving an easier problem
        # than the thing it bounds, and the trained arm could exceed it. Mask what the encoder
        # masks. (Round 9 verified the defect and did not repair it; this is that repair.)
        zt = bt["z"].clone()
        zt[bt["input_mask"]] = 0.0
        z, zf, im = zt.numpy(), bt["z_f"].numpy(), bt["input_mask"].numpy()
        # Round 11 (perspective, reproduced): this scored EVERY cell of the test half, where
        # the probe it normalizes scores only target_mask & non-ROOT cells -- a reference over
        # a different population than the thing it references. Filter to the probe's own cells.
        tm = bt["target_mask"].numpy()
        nonroot = (bt["cell_family"] != int(FuncFamily.ROOT)).numpy()
        B, R, D = z.shape
        half = R // 2
        for b in range(B):
            for d in range(D):
                if not nonroot[b, d]:
                    continue
                others = [j for j in range(D) if j != d]
                tr = ~im[b, :half, d]
                te = tm[b, half:, d]
                if tr.sum() < 5 or not te.any():
                    continue
                Xtr = z[b, :half][tr][:, others]
                ytr = np.clip(zf[b, :half][tr, d], -CLAMP, CLAMP)
                Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
                w = np.linalg.solve(Xb.T @ Xb + RIDGE_ALPHA * np.eye(Xb.shape[1]), Xb.T @ ytr)
                Xte = np.hstack([z[b, half:][te][:, others], np.ones((int(te.sum()), 1))])
                errs.append((Xte @ w - np.clip(zf[b, half:][te, d], -CLAMP, CLAMP)) ** 2)
    return round(float(np.mean(np.concatenate(errs))), 4)


def main():
    out = {"protocol": {"source": "eval/probe_masked_all.py (prior, seed, policies, selection, "
                                  "clamp, ridge alpha, half/half split)",
                        "across_table": "fair competitor: one ridge, must generalize across tables",
                        "per_table": "UPPER REFERENCE, not a competitor: refits per table and reads "
                                     "ground-truth f for the rows it fits, which no trained arm sees",
                        "sigma": 1.0, "n_batches": N_BATCHES, "batch": BATCH, "ctx": CTX},
           "by_k": {}}
    for k in KS + [SCM_KEY]:
        # The SCM prior draws a different column count per table (4-32), so a single pooled
        # ridge over row-value features has no fixed width and the across-table competitor is
        # undefined there. The per-table reference -- the ceiling the headline negative lacked --
        # is well defined on both substrates, and that is what this run adds.
        ent = {p: {"per_table": per_table(k, p)} for p in POLICIES}
        if k != SCM_KEY:
            for p in POLICIES:
                ent[p]["across_table"] = across_table(k, p)
        else:
            for p in POLICIES:
                ent[p]["across_table"] = None
            ent["across_table_note"] = ("undefined: the SCM prior varies column count per table, "
                                        "so pooled row-value features have no fixed width")
        out["by_k"][str(k)] = ent
        print(f"  K={str(k):>4}  " + "  ".join(
            f"{p}: across "
            + ("---   " if ent[p]["across_table"] is None else f"{ent[p]['across_table']:.4f}")
            + f" / per-table {ent[p]['per_table']:.4f}" for p in POLICIES), flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
