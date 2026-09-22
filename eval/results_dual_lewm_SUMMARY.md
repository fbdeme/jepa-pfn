# #24 dual+LeWM ablation — results (base scale, emb256)

De-confound for the latent-vs-data-space thesis. Existing arms differed in TWO ways
at once — `base_dual` = EMA target + value head, `base_lat_s` = SIGReg + no head — so a
skeptic could blame the latent arm's failure on EMA-vs-SIGReg instead of the objective.
`base_dual_lewm` = **SIGReg + value head** (holds the head fixed vs base_dual, holds the
anti-collapse fixed vs base_lat_s), splitting the confound into two clean isolations.

## Training note (a finding in itself)

SIGReg + value head is **training-unstable**. At the matched recipe (lr 1e-3, λ_sig 0.25,
batch 4) the encoder **fully collapsed** (dim_std→0, cos→1, SIGReg loss exploded to 6000+)
by ~step 1200; bumping λ_sig→0.4 only delayed collapse to ~step 2000. It trained cleanly
only after **lr→5e-4 AND λ_sig→0.4** (final: dim_std ~1.0, dim_std_min ~0.6, cos ~0.04,
val_jepa 0.48). The headless `base_lat_s` was stable at lr 1e-3 with no such tuning.
→ **EMA is a robust anti-collapse for a value-headed latent model; SIGReg is fragile.**
(This forces dual_lewm's lr/λ_sig to differ from the other arms — the clean comparison is
dual(EMA) vs dual_lewm(SIGReg), value head fixed at λ_ppd 0.1; λ_sig has no EMA analog.)

## Probe (encoder mechanism; N=8 tables, linear+nonlinear)

| encoder | edge_auc (.50) | fam_acc (maj .329) | mse_f@1 ↓ |
|---|---|---|---|
| base_ds (data-space) | **.597** | **.622** | .079 |
| base_dual (EMA + head) | **.569** | **.556** | .025 |
| base_dual_lewm (SIGReg + head) | .496 | .327 | 1.16 |
| base_lat_s (SIGReg, no head) | .498 | .314 | .443 |
| random_init | .552 | .463 | .016 |

base_ds/base_lat_s reproduce results_probe_base.json exactly (deterministic → comparable
to base_dual's stored .569/.556). Both SIGReg arms sit at chance and **below random_init**
on edge — the latent objective *destroys* linearly-decodable structure a random projection keeps.

## Suite (downstream; 13-dataset pilot, PFN value head + point_pred)

| arm | clf_acc (10) | clf_auc (10) | reg_r² (3) |
|---|---|---|---|
| base_ds (data-space) | **.724** | **.758** | **.544** |
| base_dual (EMA + head) | .683 | **.715** | **.467** |
| base_dual_lewm (SIGReg + head) | .634 | .509 | −.005 |
| base_lat_s (SIGReg, no head) | .631 | .499 | .005 |
| logreg / ridge | .717 | .738 | .537 |
| histgb / histgbr | .764 | .793 | .652 |

- **base_dual_lewm == base_lat_s within .001 on 7/13 datasets** (identical elsewhere within
  noise). clf_acc looks non-trivial only via majority-class prediction — **AUC .509 ≈ chance**
  and **reg r² ≈ 0** expose that dual_lewm's *trained* value head is no better than lat_s's
  *untrained* head.
- **base_ds ≥ base_dual on 10/13 datasets.**

## Verdict (two isolations)

1. **Value-head effect (SIGReg fixed):** `base_dual_lewm ≈ base_lat_s` on probe AND suite.
   Adding a value head to the SIGReg arm does **nothing** — lat_s's failure is NOT "it lacks
   a value head."
2. **Anti-collapse effect (value head fixed):** `base_dual (EMA) ≫ base_dual_lewm (SIGReg)`
   on probe (.569 vs .496) and suite (auc .715 vs .509, r² .467 vs −.005). base_dual's
   usability comes from the **EMA target**, not the head; SIGReg cannot support a usable head.

**Confound resolved:** the latent arm's failure tracks the objective/anti-collapse
mechanism, not the presence of a value head. And the best latent arm (EMA dual) is still
**below data-space** (probe .569<.597; suite ≥ on 10/13). Thesis reinforced — even after
stabilizing SIGReg+head training and handing it a value head, latent < data-space.

## Big-scale replication (emb 512, ~35M) — steelman

`big_dual_lewm` at the big trio recipe (lr 2e-4, λ_sig 0.4, λ_ppd 0.5, batch 2). It has a
**healthy plateau up to step ~12000** (val erank 124, dim_std 1.02, cos 0.002) then collapses
~step 12100 — the same pattern as headless `big_lat_s` (collapses ~13.5k). We captured the
peak-healthy ckpt as the **steelman `big_dual_lewm_h`** (step 12000, mirrors `big_lat_s_h`),
so the verdict cannot be dismissed as a collapse artifact.

Probe (mechanism): big_ds .583 / big_dual (EMA) .557 / **big_dual_lewm_h .496 ≈ big_lat_s_h
.494** (both chance) / random_init .557.

Suite (13-dataset pilot):

| arm | clf_acc | clf_auc (.50) | reg_r² |
|---|---|---|---|
| big_ds (data-space) | **.723** | **.752** | **.577** |
| big_dual (EMA + head) | **.711** | **.739** | **.493** |
| big_dual_lewm_h (SIGReg + head, steelman) | .631 | .485 | −.009 |
| big_lat_s_h (SIGReg, no head, steelman) | .628 | .524 | −.017 |

**Same result at headline scale**: big_dual_lewm_h == big_lat_s_h within .001 on **7/13**
datasets (both chance AUC, ~0 r²); big_ds ≥ big_dual on 8/13 + probe. The value head is inert
on a SIGReg encoder even at 35M and even on a **non-collapsed steelman** — the base-scale
de-confound holds at the headline scale.

Artifacts (big): `results_probe_dual_lewm_big.json`, `results_suite_dual_lewm_big.json`. Ckpts:
local `runs/big_dual_lewm_h/` (steelman) + `runs/big_dual_lewm/` (collapsing) + HF `fbdeme/jepa-pfn`.

Artifacts (base): `results_probe_dual_lewm.json`, `results_suite_dual_lewm.json`,
`configs/base_dual_lewm.yaml`. Ckpt: local `runs/base_dual_lewm/` + HF `fbdeme/jepa-pfn`.
