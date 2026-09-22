# I-JEPA split on the cell-token PFN: single-seed smoke (2026-09-07/08)

Question: does moving the prediction from the encoder (mask_emb in-fill) to the predictor (context-only encoder +
learned mask token, I-JEPA style) give the latent arm a target that carries the hidden cell's value? Three runs, seed 0,
the v4 latent cell's recipe (emb 256, 6 layers, twoway-2 predictor, real-matched prior, lr 5e-4, mixed masking) with the
target moved from the row CLS to the hidden cell itself:

| run | encoder | column code | target r | steps |
|---|---|---|---|---|
| cell_ctx | context-only (ctx_only) | rope | shared | 20000 |
| cell (control) | in-fill (mask_emb) | rope | shared | 20000 |
| cell_freshr (A) | context-only, no CLS | resample (TabPFN-v2 style) | fresh per branch | stopped at 8250 (collapsed) |
| cell_ctx_diff | context-only (ctx_only) | rope | shared; target = EMA(full) - EMA(masked view) | 20000 |
| dual_split | context-only (ctx_only) | rope | diff target + bar32 head on the twoway-4 predictor (lambda 1 / 1) | 20000 |

## 1. Training log at the last val step

| run | step | val_jepa | tgt erank | tgt dim_std_min | tgt cos | tgt within-table var | pred cos |
|---|---|---|---|---|---|---|---|
| cell_ctx_s0 | 20000 | 0.0158 | 32.0 | 0.3031 | 0.166 | 0.939 | 0.170 |
| cell_s0 | 20000 | 0.2743 | 124.2 | 0.4197 | 0.311 | 0.896 | 0.399 |
| cell_freshr_s0 | 8250 | 0.0005 | 155.1 | 0.0000 | 1.000 | n/a | 1.000 |
| cell_ctx_diff_s0 | 20000 | 0.6183 | 93.8 | 0.4074 | 0.247 | 0.759 | 0.295 |
| dual_split_s0 | 20000 | 0.5183 | 19.9 | 0.1266 | 0.195 | 0.985 | 0.553 |

## 2. What the EMA target at a hidden cell encodes (held-out linear R2, ridge; real prior, any_cell, 128 context rows)

| run | step | R2 target ~ column one-hot | ~ value z | ~ true f | value z from target | value z from prediction |
|---|---|---|---|---|---|---|
| ctx | 20000 | 0.594 | -0.000 | -0.000 | -0.008 | -0.006 |
| infill | 20000 | -0.006 | 0.072 | 0.071 | 0.932 | 0.238 |
| freshr | 6300 | -73.623 | -67.354 | -67.359 | -0.000 | -0.000 |
| diff (mid) | 4700 | 0.387 | 0.089 | 0.088 | 0.915 | 0.034 |
| diff | 20000 | 0.053 | 0.143 | 0.141 | 0.775 | 0.055 |
| dualsplit (mid) | 5975 | -0.003 | 0.379 | 0.372 | 0.936 | 0.096 |
| dualsplit | 20000 | -0.005 | 0.465 | 0.457 | 0.946 | 0.260 |

(freshr at 6300: the target has zero variance, so every R2 is undefined; the row is kept for the record.)

## 3. Masked f-MSE probe (lower is better; constant map 0.966 / random-init floor 0.968 under any_cell)

| run | policy | cell read-out | predictor read-out | mse_x (value retention) |
|---|---|---|---|---|
| ctx | any_cell | 0.979 | 0.976 | 1.002 |
| ctx | mixed | 0.979 | 0.974 | 0.990 |
| infill | any_cell | 0.903 | n/a | 0.925 |
| infill | mixed | 0.967 | n/a | 0.978 |
| ds | any_cell | 0.634 | n/a | 0.655 |
| ds | mixed | 0.772 | n/a | 0.780 |
| diff | any_cell | 1.016 | 0.920 | 1.040 |
| diff | mixed | 1.030 | 0.968 | 1.042 |
| dualsplit | any_cell | 0.970 | 0.734 | 0.992 |
| dualsplit | mixed | 0.968 | 0.836 | 0.978 |

## 4. Value read-out of the dual_split head vs the ds arm (the trainers' identical val_mse / bar-CE definition, same val seed, any_cell)

| step | dual_split val_mse | ds val_mse | dual_split bar CE | ds bar CE | dual_split tgt erank | tgt dim_std_min |
|---|---|---|---|---|---|---|
| 5000 | 0.8001 | 0.8581 | 2.260 | 1.571 | 33.3 | 0.1778 |
| 10000 | 0.7301 | 0.8017 | 2.067 | 1.471 | 30.2 | 0.1681 |
| 20000 | 0.6566 | 0.7725 | 1.930 | 1.416 | 19.9 | 0.1266 |
| ds at 40000 | - | 0.7463 | - | 1.366 | - | - |

Caveat: the architectures are not matched (dual_split = 6 encoder + 4 predictor blocks with the head on the predictor; ds = 6 blocks with the head on the encoder), one seed, training policy mixed vs any_cell. The matched control (same split, lambda_jepa 0: configs/v4_mae_ctx_tw4_lr5e-4_mixed_cell_s0.yaml) has not been run.

## 5. Real data (E9 protocol: CC18 + Grinsztajn + TabArena, 147 datasets, K_FEAT 64, CTX 1024): v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0 vs references, per-dataset exact sign test

| reference | task/metric | n | wins / losses | p (sign) | mean run | mean ref | delta |
|---|---|---|---|---|---|---|---|
| real_ds_lr5e-4_s0 | clf_acc | 115 | 61 / 35 | 0.01035 | 0.6676 | 0.6550 | +0.0126 |
| real_ds_lr5e-4_s0 | clf_auc | 115 | 69 / 46 | 0.03975 | 0.7197 | 0.6976 | +0.0221 |
| real_ds_lr5e-4_s0 | reg_r2 | 32 | 13 / 19 | 0.37709 | 0.4613 | 0.4599 | +0.0013 |
| v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0 | clf_acc | 115 | 28 / 73 | 1e-05 | 0.6676 | 0.7013 | -0.0337 |
| v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0 | clf_auc | 115 | 22 / 93 | 0.0 | 0.7197 | 0.7592 | -0.0395 |
| v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0 | reg_r2 | 32 | 8 / 24 | 0.007 | 0.4613 | 0.5230 | -0.0617 |
| v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s0 | clf_acc | 115 | 18 / 82 | 0.0 | 0.6676 | 0.7151 | -0.0475 |
| v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s0 | clf_auc | 115 | 13 / 102 | 0.0 | 0.7197 | 0.7743 | -0.0547 |
| v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s0 | reg_r2 | 32 | 3 / 29 | 0.0 | 0.4613 | 0.5598 | -0.0986 |
| histgb | clf_acc | 115 | 13 / 101 | 0.0 | 0.6676 | 0.8424 | -0.1748 |
| histgb | clf_auc | 115 | 6 / 109 | 0.0 | 0.7197 | 0.8785 | -0.1588 |
| histgbr | reg_r2 | 32 | 0 / 32 | 0.0 | 0.4613 | 0.7366 | -0.2753 |

Inference path of the run under test: x-encoder (observed cells only) -> predictor (mask tokens) -> bar head (model.jepa.ValueModel); the y-encoder is training-only. real_ds_lr5e-4_s0 = the same-step (20k) ds arm; the 40k rows are ds at twice the steps and E6's latent-init two-stage arm.

## Verdict

1. **cell_ctx (I-JEPA split, rope, shared code)**: the target is a column-address code. Column one-hot explains its variance,
   value and true f explain none, nothing about the value can be read back, and both probe read-outs sit at the constant map.
   The JEPA loss falls toward zero because the predictor reproduces the address from its rope position. Variance-collapse
   monitors (dim_std, cos, within-table var) all read healthy: this is a shortcut, not a collapse.
2. **cell (in-fill control)**: the target is a near-invertible code of the observed value (Issue #19's regime); the predictor
   recovers a quarter of it. The cell read-out equals E1's pure-latent CLS-target cell to three decimals, so the
   CLS-to-cell target swap changes nothing about the latent arm's f read-out; the ds reference stays far ahead.
3. **cell_freshr (A: the target encoder draws its own column codes)**: the target's address component becomes
   unpredictable and the EMA target collapses to a constant within a few thousand steps (dim_std 0, cos 1, prediction
   constant). The shared address was what pulled the target out of its initial near-collapse; removing it leaves collapse
   as the absorbing state. Stopped early; the box was destroyed.
4. **cell_ctx_diff (difference target)**: see the rows above; the verdict for this run is stated in
   docs/current_status.md session (24b) from these numbers.
5. **dual_split (diff target + value head on the predictor)**: sections 2-4; the verdict is stated in
   docs/current_status.md session (24c) from these numbers.

Net: on this substrate a cell latent target is either an address code (predictor-side prediction), a value code
(encoder-side in-fill), or a constant (address decorrelated). None carries information beyond the value, which is the
data-space target already. The structural change asked for at the start of the session was implemented with the
existing arms untouched (flags default off; 54 tests) and did not open a latent advantage.
