"""Prior -> training batches: context/query split, standardization, masking.

Masking policies (SSOT section 3):
- y_only:   one random column per table is the target, masked in all query rows
            (the shape of supervised learning, TabPFN-style)
- any_cell: each query-row cell masked with a per-batch random rate
            (the generalized task; data-space control for Phase 3 JEPA)
- row_block_partial: every query row loses 50-80% of its cells, never all (C5 / v4 E2)

Cells are standardized per column with context-row statistics. Missing (NaN)
cells are input-masked but never scored; target cells are always observed
values that were hidden from the model.
"""

import numpy as np
import torch


def make_batch(prior, batch_size, policy, rng, device="cpu", split=None,
               return_truth=False):
    b = prior.sample_batch(batch_size)
    x = b.x
    B, R, D = x.shape
    if split is None:  # eval passes a fixed split for context-size curves
        split = int(np.clip(int(R * rng.uniform(0.3, 0.9)), 8, R - 2))

    missing = np.isnan(x)
    tmask = np.zeros((B, R, D), bool)
    if policy == "y_only":
        cols = rng.integers(D, size=B)
        for i in range(B):
            tmask[i, split:, cols[i]] = True
    elif policy == "y_target":  # classification prior: the table's designated target column (hyper) in query rows
        hyp = getattr(b, "hyper", None) or [None] * B
        for i in range(B):
            h = hyp[i] or {}
            if h.get("target_col") is None or not h.get("n_classes"):
                continue                                              # no K-class label column: the table carries no loss
            tmask[i, split:, h["target_col"]] = True
    elif policy == "y_last":  # dump diagnostic: target the LAST column (the label) in query
        tmask[:, split:, -1] = True   # rows, TabPFN's native supervised task on its own prior
    elif policy == "row_block_partial":
        # C5 / E2: every query row loses a U(0.5, 0.8) fraction of its cells (an I-JEPA-sized block per row),
        # never all of them - a fully hidden row has only the row-iid optimum (the average row) to predict.
        for i in range(B):
            k = int(np.clip(round(rng.uniform(0.5, 0.8) * D), 1, max(D - 1, 1)))
            cols = rng.random((R - split, D)).argsort(1)[:, :k]          # k distinct columns per query row
            tmask[i, split + np.arange(R - split)[:, None], cols] = True
    elif policy in ("any_cell", "mixed"):
        # "mixed" (SSOT section 3): iid cells / column blocks / rectangular
        # blocks - chunky masks resist trivial interpolation (I-JEPA/V-JEPA)
        mode = rng.choice(["iid", "col_block", "block"]) if policy == "mixed" else "iid"
        if mode == "iid":
            p = rng.uniform(0.15, 0.5)
            tmask[:, split:, :] = rng.random((B, R - split, D)) < p
            none = ~tmask[:, split:, :].any(-1)  # ensure >=1 target per query row
            rows_b, rows_r = np.nonzero(none)
            tmask[rows_b, split + rows_r, rng.integers(D, size=len(rows_b))] = True
        else:
            for i in range(B):
                k = int(rng.integers(1, D)) if D > 1 else 1  # keep >=1 anchor col
                cols = rng.choice(D, k, replace=False)
                if mode == "col_block":
                    tmask[i, split:, cols] = True
                else:  # block: column subset x random half of the query rows
                    rows = split + rng.choice(R - split, max(1, (R - split) // 2),
                                              replace=False)
                    tmask[i, rows[:, None], cols] = True
    else:
        raise ValueError(policy)
    tmask &= ~missing

    ctx = x[:, :split]
    mean = np.nanmean(ctx, 1, keepdims=True)
    std = np.nanstd(ctx, 1, keepdims=True)
    mean, std = np.nan_to_num(mean), np.nan_to_num(std) + 1e-8
    z = np.nan_to_num(np.clip((x - mean) / std, -100, 100))

    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    out = dict(z=t(z, torch.float32), target_mask=t(tmask, torch.bool),
               input_mask=t(tmask | missing, torch.bool), split=split)
    if policy == "y_target":  # class ids for the class head (raw column values are 0..K-1 after discretisation)
        hyp = getattr(b, "hyper", None) or [None] * B
        out["y_ids"] = t(np.nan_to_num(x, nan=-1).astype(np.int64), torch.long)
        out["n_classes"] = t(np.array([(h or {}).get("n_classes", 0) for h in hyp], np.int64), torch.long)
    if return_truth:  # evaluation only - never reaches a training loss (SSOT)
        z_f = np.clip((b.f - mean) / std, -100, 100)  # same stats as x
        out.update(
            z_f=t(z_f, torch.float32),
            cell_sigma=t(np.take_along_axis(b.noise_sigma, b.observed_idx, 1), torch.float32),
            cell_family=t(np.take_along_axis(b.func_family, b.observed_idx, 1), torch.int8),
            adjacency=t(b.adjacency, torch.bool),
            observed_idx=t(b.observed_idx.astype(np.int64), torch.long),
            categorical=t(b.categorical, torch.int8))      # 0 = continuous column, else category count
        if getattr(b, "hyper", None):
            out["hyper"] = b.hyper
    return out
