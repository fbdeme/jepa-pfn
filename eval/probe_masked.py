"""(round-4 repair) Mechanism recovery read where the objective actually writes.

`eval/probe_base.py` has two defects that round 4 converged on, and this script is the
control for both. It does not replace that probe; it answers the question that one cannot.

Defect 1 -- the cell is never hidden. `probe_base.collect` encodes with `no_mask` and then
probes CONTEXT cells (`h[:, :ctx]`), i.e. cells the encoder was shown. Recovering f from a
cell you can see is value preservation, not mechanism inference; probe_base.py:34 says as
much in its own comment. Here the target cells are masked before encoding, so an arm has to
infer them from the rest of the table.

Defect 2 -- the probe reads columns the latent loss never trains. `keep_cls` is never passed
anywhere in eval/, so `CellPFN.encode` strips the CLS columns (model/pfn.py:202). But
`cls_target: true` routes the whole latent loss onto exactly those CLS columns
(model/jepa.py:165). Every latent verdict so far was therefore read off the part of the
representation the latent objective does not directly optimise. Here we read both.

Three read-outs per arm, all on the SAME masked encoding:
  cell       -- the masked cell's own embedding            (E dims)
  cls_mean   -- mean of its row's K CLS embeddings         (E dims, dimension-matched)
  cls_concat -- its row's K CLS embeddings concatenated    (K*E dims)
`cls_concat` has more features than `cell`, so `cls_mean` is reported alongside to foreclose
"the CLS read-out only won because it had 4x the dimensions".

Two targets per read-out:
  mse_f -- the noiseless mechanism value  (what we claim to measure)
  mse_x -- the observed noisy value       (control: value preservation)
An arm whose mse_f tracks its mse_x is preserving values, not isolating mechanism.

Run: uv run python -m eval.probe_masked            (CPU, forward-only, frozen ckpts)
     PROBE_ARMS=base_ds_s0,base_lat_s_s0 uv run python -m eval.probe_masked
Writes eval/results_probe_masked.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.latent_probe import load_jepa           # noqa: E402
from eval.truth_eval import ridge_mse             # noqa: E402
from model.pfn import CellPFN                     # noqa: E402
from prior.scm import FuncFamily, PriorConfig, SCMPrior   # noqa: E402
from train.data import make_batch                 # noqa: E402

ROOT = Path(__file__).parents[1]
CTX, N_QUERY = 128, 50
N_BATCHES, BATCH = 8, 8
EVAL_SEED = 30_000
SIGMAS = (1.0, 2.0, 4.0, 8.0)
NCOLS = (4, 32)
MAX_N = 24000
CLEAN = dict(p_categorize=0.0, p_missing_table=0.0)

DEFAULT_ARMS = ("base_ds_s0,base_ds_s1,base_ds_s2,base_ds_s3,base_ds_s4,"
                "base_dual_s0,base_dual_s1,base_dual_s2,base_dual_s3,base_dual_s4,"
                "base_lat_s_s0,base_lat_s_s1,base_lat_s_s2,base_lat_s_s3,base_lat_s_s4")
ARMS = os.environ.get("PROBE_ARMS", DEFAULT_ARMS).split(",")
REF = os.environ.get("PROBE_REF", "base_ds_s0")   # ckpt whose shape the random floor copies


def load_enc(run_name):
    """Frozen encoder. A ds run is a bare CellPFN; a latent run is a JEPA wrapper whose
    .encoder() is what every other probe in this repo reads, so we match that convention."""
    ck = torch.load(ROOT / "runs" / run_name / "ckpt.pt", map_location="cpu",
                    weights_only=False)
    if any(k.startswith("online.") for k in ck["model"]):
        return load_jepa(run_name).encoder().eval()
    c = ck["cfg"]
    m = CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                n_cls=c.get("n_cls") or 0, r_scheme=c.get("r_scheme") or "resample")
    m.load_state_dict(ck["model"])
    return m.eval()


def random_floor(seed=999):
    c = torch.load(ROOT / "runs" / REF / "ckpt.pt", map_location="cpu",
                   weights_only=False)["cfg"]
    torch.manual_seed(seed)
    return CellPFN(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
                   n_cls=c.get("n_cls") or 0,
                   r_scheme=c.get("r_scheme") or "resample").eval()


@torch.no_grad()
def collect(enc, noise):
    """Encode with the target cells HIDDEN, keeping the CLS columns.

    Returns, per masked non-root query cell: its own embedding, its row's CLS mean and
    concatenation, the noiseless mechanism value f, and the observed value x."""
    torch.manual_seed(EVAL_SEED)          # paired tables across arms
    prior = SCMPrior(PriorConfig(n_rows=(CTX + N_QUERY, CTX + N_QUERY), n_cols=NCOLS,
                                 noise_scale=noise, **CLEAN), seed=EVAL_SEED)
    rng = np.random.default_rng(EVAL_SEED)
    K = enc.n_cls
    cell, cmean, ccat, f_, x_ = [], [], [], [], []
    for _ in range(N_BATCHES):
        bt = make_batch(prior, BATCH, "any_cell", rng, split=CTX, return_truth=True)
        tm = bt["target_mask"]                     # query-row cells to predict
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        hc = h[:, :, K:] if K else h               # cell columns
        # non-root only, matching probe_base's c4_mse_f
        nonroot = (bt["cell_family"] != int(FuncFamily.ROOT)).unsqueeze(1).expand_as(tm)
        sel = tm & nonroot
        if not sel.any():
            continue
        b, r, d = sel.nonzero(as_tuple=True)
        cell.append(hc[b, r, d])
        if K:
            rowcls = h[:, :, :K][b, r]             # (n, K, E) the CLS of each cell's row
            cmean.append(rowcls.mean(1))
            ccat.append(rowcls.reshape(rowcls.shape[0], -1))
        f_.append(bt["z_f"][b, r, d].clamp(-3.05, 3.05))
        x_.append(bt["z"][b, r, d].clamp(-3.05, 3.05))
    out = {"cell": torch.cat(cell).numpy()[:MAX_N]}
    if K:
        out["cls_mean"] = torch.cat(cmean).numpy()[:MAX_N]
        out["cls_concat"] = torch.cat(ccat).numpy()[:MAX_N]
    return out, (torch.cat(f_).numpy()[:MAX_N], torch.cat(x_).numpy()[:MAX_N])


def main():
    res, t0 = {}, time.time()
    arms = list(ARMS) + ["random_init"]
    for name in arms:
        enc = random_floor() if name == "random_init" else load_enc(name)
        rec, t = {}, time.time()
        for s in SIGMAS:
            feats, (f_, x_) = collect(enc, s)
            rec[str(s)] = {rd: {"mse_f": round(ridge_mse(X, f_), 4),
                                "mse_x": round(ridge_mse(X, x_), 4),
                                "dim": int(X.shape[1]), "n": int(X.shape[0])}
                           for rd, X in feats.items()}
        res[name] = rec
        one = rec["1.0"]
        print(f"{name:18} " + "  ".join(
            f"{rd}: f={v['mse_f']:.4f}/x={v['mse_x']:.4f}" for rd, v in one.items())
            + f"   [{time.time()-t:.0f}s]", flush=True)
    res["_protocol"] = {
        "masking": "target cells hidden before encoding (input_mask=target_mask|missing), "
                   "policy any_cell, query rows only",
        "readouts": "cell = masked cell's own embedding; cls_mean / cls_concat = its row's "
                    "CLS columns, read with keep_cls=True",
        "targets": "mse_f = noiseless mechanism value; mse_x = observed value (control)",
        "sigmas": list(SIGMAS), "ctx": CTX, "n_query": N_QUERY,
        "batches": N_BATCHES, "batch": BATCH, "eval_seed": EVAL_SEED,
        "note": "companion to eval/probe_base.py, which encodes UNMASKED context cells and "
                "strips the CLS columns; see this file's docstring",
    }
    (ROOT / "eval/results_probe_masked.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote eval/results_probe_masked.json  [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
