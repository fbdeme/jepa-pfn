"""The isolating control for the probe-reversal claim: visibility, with cell role fixed.

Round 11 (methodology, F1): the paper's headline contrast compares the unmasked probe at
CONTEXT cells (eval/probe_base.py) against the masked probe at QUERY-ROW TARGET cells
(eval/probe_masked.py). Two things change at once -- whether the cell was visible at
encoding, and which cells are read -- so "the unmasked probe scores value retention" is
confounded with cell role. Everything else was verified matched.

This completes the 2x2. Same arms, same tables, same policy draw, same query-row target
cells, same ridge; the ONLY difference between the two conditions here is `input_mask`:

  hidden    the batch's real input_mask -- the paper's masked read-out, reproduced
  visible   input_mask zeroed, so the encoder sees every cell including the targets

If the ordering follows visibility (latent arm ahead when visible, data-space ahead when
hidden) at the SAME cells, the reversal is a visibility effect and the cell-role confound
is closed. If it follows cell role, the paper's §3.3 explanation is wrong and must be
withdrawn. Either way the number decides, not the argument.

Run: uv run python -m eval.probe_visibility_control          (CPU, frozen ckpts)
Writes eval/results_probe_visibility_control.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_masked_all import (EVAL_SEED, MAX_N, POLICIES, PROTOCOL,   # noqa: E402
                                   load_auto, random_init, scm_prior)
from eval.truth_eval import ridge_mse                                       # noqa: E402
from prior.scm import FuncFamily                                            # noqa: E402
from train.data import make_batch                                           # noqa: E402

OUT = ROOT / "eval/results_probe_visibility_control.json"
ARMS = ([(f"base_ds_s{i}", "ds") for i in range(5)]
        + [(f"base_dual_s{i}", "dual") for i in range(5)]
        + [(f"base_ema_nohead_s{i}", "ema_nohead") for i in range(5)])
N_FLOOR = 3
CLAMP = 3.05


@torch.no_grad()
def both_reads(enc, prior_fn, policy, ctx, n_batches, batch):
    """mse_f at the same query-row target cells under both visibility conditions."""
    torch.manual_seed(EVAL_SEED)               # paired tables and column identities
    prior = prior_fn()
    rng = np.random.default_rng(EVAL_SEED)
    K = getattr(enc, "n_cls", 0) or 0
    H = {"hidden": [], "visible": []}
    f_ = []
    for _ in range(n_batches):
        bt = make_batch(prior, batch, policy, rng, split=ctx, return_truth=True)
        tm = bt["target_mask"]
        nonroot = (bt["cell_family"] != int(FuncFamily.ROOT)).unsqueeze(1).expand_as(tm)
        sel = tm & nonroot
        if not sel.any():
            continue
        b, r, d = sel.nonzero(as_tuple=True)
        for cond, im in (("hidden", bt["input_mask"]),
                         ("visible", torch.zeros_like(bt["input_mask"]))):
            h = enc.encode(bt["z"], im, bt["split"], keep_cls=True)
            hc = h[:, :, K:] if K else h
            H[cond].append(hc[b, r, d])
        f_.append(bt["z_f"][b, r, d].clamp(-CLAMP, CLAMP))
    f_ = torch.cat(f_).numpy()[:MAX_N]
    return {cond: round(ridge_mse(torch.cat(v).numpy()[:MAX_N], f_), 4)
            for cond, v in H.items()}, int(len(f_))


def main():
    prior_fn, ctx, n_batches, batch = PROTOCOL["recipe"]
    out = {"protocol": {"source": "eval/probe_masked_all.py PROTOCOL['recipe']; identical "
                                  "query-row target cells under both conditions, only "
                                  "input_mask differs", "ctx": ctx,
                        "n_batches": n_batches, "batch": batch, "eval_seed": EVAL_SEED},
           "arms": {}, "floor": {}}
    for arm, grp in ARMS:
        if not (ROOT / "runs" / arm / "ckpt.pt").exists():
            continue
        t = time.time()
        enc = load_auto(arm)
        rec = {"group": grp}
        for p in POLICIES:
            rec[p], rec["n"] = both_reads(enc, prior_fn, p, ctx, n_batches, batch)
        out["arms"][arm] = rec
        print(f"  {arm:22} " + "  ".join(
            f"{p}: hid={rec[p]['hidden']:.4f}/vis={rec[p]['visible']:.4f}"
            for p in POLICIES) + f"  [{time.time()-t:.0f}s]", flush=True)
    for d in range(N_FLOOR):
        enc = random_init("base_ds_s0", 10_000 + d)
        rec = {}
        for p in POLICIES:
            rec[p], _ = both_reads(enc, prior_fn, p, ctx, n_batches, batch)
        out["floor"][f"draw{d}"] = rec
        print(f"  floor/draw{d}          " + "  ".join(
            f"{p}: hid={rec[p]['hidden']:.4f}/vis={rec[p]['visible']:.4f}"
            for p in POLICIES), flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
