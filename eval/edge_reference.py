"""Encoder-free reference for the structural read-out (edge AUC / family accuracy).

Round 11, four reviewers: §4.2.1 breaks the probe-alignment tie with structural targets,
but that read-out was never given the control every other read-out in this paper has --
an encoder-free reference that bounds what the read-out extracts from the data alone.
The mechanism probe has the constant map and the across-table ridge; the structural
read-out had nothing.

This supplies it, under c5_scores's own protocol (same prior, seed, batch count, pair
construction, half/half split, LogisticRegression for parity where a fit is involved):

  edge:   |Pearson r| between two columns' context values, per table. Zero parameters.
  family: a logistic regression on four moments of the column's context values
          (mean, std, skew-proxy, kurtosis-proxy) -- no encoder, same fit class.

The result decides how much the structural tie-break can carry: if a statistic with no
encoder outscores every trained arm on edge AUC, the read-out is not measuring what the
encoder added, and the tie-break must be re-scoped to the targets where the encoder-free
reference does NOT win.

Run: uv run python -m eval.edge_reference           (CPU)
Writes eval/results_edge_reference.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
# Round 12 (methodology + artifact audit, independently): the first version of this
# reference imported truth_eval's constants (ctx 100, 24x16 tables, 3-8 columns) while the
# trained-arm numbers it disqualifies come from probe_base (ctx 128, 8x8, 4-32 columns,
# capped subsample). The tell was in the shipped artifacts: two different majority-class
# baselines nine lines apart in the manuscript. The reference now imports the ARMS' OWN
# protocol module and mirrors its pairing, subsampling and split exactly.
from eval.probe_base import (BATCH, CTX, EVAL_SEED, MAX_COLS, MAX_PAIRS,   # noqa: E402
                             MLP_DEPTH, MLP_GAIN, N_BATCHES, N_QUERY, NCOLS)
from eval.truth_eval import CLEAN                                           # noqa: E402
from prior.scm import PriorConfig, SCMPrior                                 # noqa: E402
from train.data import make_batch                                           # noqa: E402

OUT = ROOT / "eval/results_edge_reference.json"


def main():
    torch.manual_seed(EVAL_SEED)
    prior = SCMPrior(PriorConfig(n_rows=(CTX + N_QUERY, CTX + N_QUERY), n_cols=NCOLS,
                                 noise_scale=1.0, mlp_depth=(MLP_DEPTH, MLP_DEPTH),
                                 mlp_gain=MLP_GAIN, **CLEAN), seed=EVAL_SEED)
    rng = np.random.default_rng(EVAL_SEED)
    corr, edges, moments, fams = [], [], [], []
    for _ in range(N_BATCHES):
        bt = make_batch(prior, BATCH, "y_only", rng, split=CTX, return_truth=True)
        z = bt["z"][:, :CTX]                       # context rows only, as c5_scores
        adj = bt["adjacency"].float()
        for i in range(z.shape[0]):
            obs = bt["observed_idx"][i]
            D = len(obs)
            X = z[i].numpy()
            C = np.nan_to_num(np.corrcoef(X.T))
            a = (adj[i][obs][:, obs] + adj[i][obs][:, obs].T).bool()
            ii, jj = torch.triu_indices(D, D, offset=1)
            corr.append(np.abs(C[ii.numpy(), jj.numpy()]))
            edges.append(a[ii, jj].numpy())
            m = np.stack([X.mean(0), X.std(0),
                          ((X - X.mean(0)) ** 3).mean(0),
                          ((X - X.mean(0)) ** 4).mean(0)], 1)
            moments.append(m)
            fams.append(bt["cell_family"][i].numpy())
    corr = np.concatenate(corr); edges = np.concatenate(edges).astype(int)
    moments = np.concatenate(moments); fams = np.concatenate(fams).astype(int)
    sub = np.random.default_rng(0)                 # probe_base's own subsample RNG
    if len(corr) > MAX_PAIRS:
        s_ = sub.choice(len(corr), MAX_PAIRS, replace=False)
        corr, edges = corr[s_], edges[s_]
    if len(moments) > MAX_COLS:
        s_ = sub.choice(len(moments), MAX_COLS, replace=False)
        moments, fams = moments[s_], fams[s_]

    n = len(corr) // 2
    edge_auc_raw = roc_auc_score(edges[n:], corr[n:])
    lr = LogisticRegression(max_iter=1000).fit(corr[:n, None], edges[:n])
    edge_auc_fit = roc_auc_score(edges[n:], lr.predict_proba(corr[n:, None])[:, 1])

    nc = len(moments) // 2
    fam_lr = LogisticRegression(max_iter=1000).fit(moments[:nc], fams[:nc])
    fam_acc = float((fam_lr.predict(moments[nc:]) == fams[nc:]).mean())
    fam_maj = float(np.bincount(fams[nc:]).max() / len(fams[nc:]))

    out = {"protocol": {"source": "eval/probe_base.py collect/c5_scores: the trained arms' own "
                                  "prior, seed, batch shape, pairing, subsample and split; "
                                  "encoder replaced by data statistics (round-12 protocol fix)",
                        "edge_feature": "|Pearson r| of two columns' context values",
                        "family_feature": "four moments of the column's context values"},
           "edge_auc_raw": round(float(edge_auc_raw), 4),
           "edge_auc_fit": round(float(edge_auc_fit), 4),
           "fam_acc": round(fam_acc, 4), "fam_majority": round(fam_maj, 4),
           "n_pairs": int(len(corr)), "n_cols": int(len(moments))}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for k in ("edge_auc_raw", "edge_auc_fit", "fam_acc", "fam_majority"):
        print(f"  {k:14} {out[k]}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
