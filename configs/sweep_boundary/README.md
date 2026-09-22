# Compressibility sweep configs (intrinsic rank K)

The **resolved** configs the trainer recorded for the 40 runs behind Table 8, Figure 6, the
permutation p and the oracle correlation. They are copied verbatim from each run directory rather
than re-authored, so they are byte-identical to what executed, including defaults the authored
configs leave implicit.

`runs/` is gitignored, so before round 8 these were the only runs in the paper with no committed
specification -- the Setup claimed one for every run and the sweep was the exception. Regenerate
with `scripts/sync_sweep_configs.py`, which fails if a copy has drifted from its run directory.

K=4 is the C4 anchor and keeps its authored config at `configs/c4_factor_{ds,dual}.yaml`.
