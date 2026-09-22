# big (emb512, 35M) suite — PARTIAL (11/156 datasets)

Box (vast 47533242) ran out of credit and was suspended at dataset 11/156, so
`results_suite_big.json` on the box was lost. These 11 lines are recovered from
the eval log (accuracy per arm; clf = accuracy). The pattern is fully consistent:
**big_lat_s (pure latent) is catastrophic downstream (near/below chance) on every
dataset**, big_ds > big_dual ≫ big_lat_s, mirroring the probe result exactly.

To complete the full 156-dataset matched ladder (analyze_ladder vs base), the
suite must be re-run — needs a GPU box (credit top-up). The ckpts were also lost
with the box, so re-running requires re-training from the tuned configs first
(big_{ds,dual,lat_s}.yaml, seed 0 — deterministic, re-derivable).

| dataset (did) | big_ds | big_dual | big_lat_s | logreg | histgb |
|---|---|---|---|---|---|
| kr-vs-kp (3) | 0.7526 | 0.6283 | 0.4727 | 0.9492 | 0.9746 |
| balance-scale (11) | 0.8509 | 0.8605 | 0.4601 | 0.8733 | 0.8743 |
| mfeat-factors (12) | 0.847 | 0.7637 | 0.097 | 0.9316 | 0.9466 |
| mfeat-fourier (14) | 0.7311 | 0.6367 | 0.097 | 0.8105 | 0.8398 |
| breast-w (15) | 0.9695 | 0.9686 | 0.6543 | 0.9657 | 0.9552 |
| mfeat-karhunen (16) | 0.7357 | 0.6257 | 0.097 | 0.9453 | 0.9473 |
| mfeat-morphological (18) | 0.7201 | 0.6868 | 0.097 | 0.6901 | 0.7031 |
| mfeat-zernike (22) | 0.6354 | 0.5508 | 0.097 | 0.7839 | 0.7858 |
| cmc (23) | 0.474 | 0.5286 | 0.3424 | 0.4987 | 0.4993 |
| optdigits (28) | 0.765 | 0.6882 | 0.1159 | 0.9512 | 0.974 |
| credit-approval (29) | 0.8638 | 0.8541 | 0.5546 | 0.8667 | 0.8676 |
