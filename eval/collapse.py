"""The ONE collapse criterion every reader uses (jepa_collapse_census, rows_control curves, extract_real_curves,
v4_readout). Before this file the project carried three inconsistent rules (absolute dim_std, peak-relative 0.5/0.05,
batch cosine > .95) — paper_v3/review/round13/experiment_ledger_2026-09-05/verdict.md.

collapsed(stats): the LayerNorm-ed target's per-dim std < THR_DIM_STD, OR (when logged, C1) the within-table variance
fraction < THR_WITHIN_TABLE — a per-table-constant target that batch-level std cannot see (eval/jepa_target_audit.py).
trajectory(vals): onset (first collapsed validation), recovery (first alive validation after onset), state at the last one.
"""
THR_DIM_STD = 1e-3
THR_WITHIN_TABLE = 0.1


def is_collapsed(stats):
    """stats = one val_tgt dict {dim_std, ..., [within_table_var_frac]}."""
    if stats["dim_std"] < THR_DIM_STD:
        return True
    w = stats.get("within_table_var_frac")
    return w is not None and w < THR_WITHIN_TABLE


def trajectory(vals):
    """vals = metrics records carrying val_tgt, in file order (a resumed run appends, so steps need not be monotone).
    onset = first collapsed record's step, recovered = first alive record after it (None = never), collapsed = the last record."""
    onset = recovered = None
    for r in vals:
        c = is_collapsed(r["val_tgt"])
        if c and onset is None:
            onset = r["step"]
        elif not c and onset is not None and recovered is None:
            recovered = r["step"]
    last = vals[-1] if vals else None
    return dict(collapsed=bool(last and is_collapsed(last["val_tgt"])), onset=onset, recovered=recovered,
                last_step=last["step"] if last else None)
