# #24 re-train probe results (2026-08-13, emb512/35M, seed0)

| encoder | edge_auc (chance .50) | fam_acc (maj .329) | mse_f@1 (lower better) | verdict |
|---|---|---|---|---|
| big_ds (data-space)        | 0.583 | 0.601 | 0.082 | encodes mechanism |
| big_dual (JEPA+value)      | 0.557 | 0.609 | 0.065 | encodes mechanism |
| big_lat_s_h (STEELMAN, non-collapsed dim_std 1.13) | 0.494 | 0.314 | 0.955 | ~0 mechanism |
| big_lat_s (collapsed 40k, dim_std 0.0, "erank 274") | 0.497 | 0.329 | 0.948 | ~0 mechanism |
| random_init                | 0.556 | 0.429 | 0.015 | random-feature baseline |

Key: collapsed ≈ steelman failure (not a collapse artifact). Non-collapsed healthy latent
STILL fails => latent objective genuinely doesn't encode SCM mechanism at scale. Prior
session's "erank 274 = full-rank noise" was actually collapse (confirmed: dim_std 0.0).
random_init mse_f 0.015 << latent 0.95: latent training destroys linearly-decodable
function info that a random projection preserves. ds/dual reproduce recorded exactly.
