"""verdict.md for the I-JEPA-split single-seed smoke (2026-09-07/08): every number below is read from the JSON /
metrics files in this directory (source-backed). Run: uv run python eval/ijepa_split_2026-09-08/build_verdict.py"""
import json
from pathlib import Path
D = Path(__file__).parent
probe = json.loads((D / "results_probe_masked.json").read_text())["post"]
tc = json.loads((D / "results_target_content.json").read_text())["runs"]
tcf = json.loads((D / "results_target_content_freshr_6k.json").read_text())["runs"]
runs = {"ctx": "v4_dual_ema_lam0_lr5e-4_mixed_cell_ctx_s0", "infill": "v4_dual_ema_lam0_lr5e-4_mixed_cell_s0",
        "freshr": "v4_dual_ema_lam0_lr5e-4_mixed_cell_freshr_s0", "ds": "v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0",
        "diff": "v4_dual_ema_lam0_lr5e-4_mixed_cell_ctx_diff_s0", "dualsplit": "v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0"}
HAVE_DS = (D / "metrics_dual_split_s0").exists()
if HAVE_DS:   # the fifth run (2026-09-08 afternoon, user go): dual on the split = cell_ctx_diff + twoway-4 predictor + bar head ON the predictor
    tcs = json.loads((D / "results_target_content_dualsplit.json").read_text())["runs"]
    tcs_mid = json.loads((D / "results_target_content_dualsplit_mid.json").read_text())["runs"]
    probe_s = json.loads((D / "results_probe_masked_dualsplit.json").read_text())["post"]
    ds_val = {json.loads(l)["step"]: json.loads(l) for l in (D.parents[1] / "runs/v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0/metrics.jsonl").read_text().splitlines() if '"val_mse"' in l}
HAVE_DIFF = (D / "metrics_cell_ctx_diff_s0").exists()
if HAVE_DIFF:   # the fourth run (2026-09-08, user go): target = EMA(full) - EMA(masked view), on top of cell_ctx
    tcd = json.loads((D / "results_target_content_diff.json").read_text())["runs"]
    tcd_mid = json.loads((D / "results_target_content_diff_mid.json").read_text())["runs"]
    probe_d = json.loads((D / "results_probe_masked_diff.json").read_text())["post"]
def val(name, step=None):
    rows = [json.loads(l) for l in (D / f"metrics_{name}").read_text().splitlines() if '"val_jepa"' in l]
    r = rows[-1] if step is None else next(x for x in rows if x["step"] == step)
    return r
def f(x): return "n/a" if x is None else f"{x:.3f}"
L = ["# I-JEPA split on the cell-token PFN: single-seed smoke (2026-09-07/08)", "",
     "Question: does moving the prediction from the encoder (mask_emb in-fill) to the predictor (context-only encoder +",
     "learned mask token, I-JEPA style) give the latent arm a target that carries the hidden cell's value? Three runs, seed 0,",
     "the v4 latent cell's recipe (emb 256, 6 layers, twoway-2 predictor, real-matched prior, lr 5e-4, mixed masking) with the",
     "target moved from the row CLS to the hidden cell itself:", "",
     "| run | encoder | column code | target r | steps |", "|---|---|---|---|---|",
     "| cell_ctx | context-only (ctx_only) | rope | shared | 20000 |",
     "| cell (control) | in-fill (mask_emb) | rope | shared | 20000 |",
     "| cell_freshr (A) | context-only, no CLS | resample (TabPFN-v2 style) | fresh per branch | stopped at %d (collapsed) |" % val("cell_freshr_s0")["step"]]
if HAVE_DIFF:
    L.append("| cell_ctx_diff | context-only (ctx_only) | rope | shared; target = EMA(full) - EMA(masked view) | %d |" % val("cell_ctx_diff_s0")["step"])
if HAVE_DS:
    L.append("| dual_split | context-only (ctx_only) | rope | diff target + bar32 head on the twoway-4 predictor (lambda 1 / 1) | %d |" % val("dual_split_s0")["step"])
L.append("")
L += ["## 1. Training log at the last val step", "", "| run | step | val_jepa | tgt erank | tgt dim_std_min | tgt cos | tgt within-table var | pred cos |", "|---|---|---|---|---|---|---|---|"]
for k in ("cell_ctx_s0", "cell_s0", "cell_freshr_s0") + (("cell_ctx_diff_s0",) if HAVE_DIFF else ()) + (("dual_split_s0",) if HAVE_DS else ()):
    r = val(k); t, p = r["val_tgt"], r["val_pred"]
    L.append(f"| {k} | {r['step']} | {r['val_jepa']:.4f} | {t['erank']:.1f} | {t['dim_std_min']:.4f} | {t['cos']:.3f} | {f(t.get('within_table_var_frac'))} | {p['cos']:.3f} |")
L += ["", "## 2. What the EMA target at a hidden cell encodes (held-out linear R2, ridge; real prior, any_cell, 128 context rows)", "",
      "| run | step | R2 target ~ column one-hot | ~ value z | ~ true f | value z from target | value z from prediction |", "|---|---|---|---|---|---|---|"]
for k, src in (("ctx", tc), ("infill", tc), ("freshr", tcf)) + ((("diff (mid)", tcd_mid), ("diff", tcd)) if HAVE_DIFF else ()) + ((("dualsplit (mid)", tcs_mid), ("dualsplit", tcs)) if HAVE_DS else ()):
    r = src[runs[k.split(" ")[0]]]
    L.append(f"| {k} | {r['step']} | {r['target']['col']:.3f} | {r['target']['z']:.3f} | {r['target']['f']:.3f} | {r['reverse']['z_from_target']:.3f} | {r['reverse']['z_from_pred']:.3f} |")
L += ["", "(freshr at %d: the target has zero variance, so every R2 is undefined; the row is kept for the record.)" % tcf[runs["freshr"]]["step"], "",
      "## 3. Masked f-MSE probe (lower is better; constant map %.3f / random-init floor %.3f under any_cell)" % (probe["const_map_mse_f"]["any_cell"], probe["floor"]["any_cell"]["mean"]), "",
      "| run | policy | cell read-out | predictor read-out | mse_x (value retention) |", "|---|---|---|---|---|"]
for k in ("ctx", "infill", "ds") + (("diff",) if HAVE_DIFF else ()) + (("dualsplit",) if HAVE_DS else ()):
    a = (probe_d if k == "diff" else probe_s if k == "dualsplit" else probe)["arms"][runs[k]]
    for pol in ("any_cell", "mixed"):
        L.append(f"| {k} | {pol} | {a[pol]['cell']['mse_f']:.3f} | {f(a[pol].get('pred', {}).get('mse_f'))} | {a[pol]['cell']['mse_x']:.3f} |")
if HAVE_DS:
    L += ["", "## 4. Value read-out of the dual_split head vs the ds arm (the trainers' identical val_mse / bar-CE definition, same val seed, any_cell)", "",
          "| step | dual_split val_mse | ds val_mse | dual_split bar CE | ds bar CE | dual_split tgt erank | tgt dim_std_min |", "|---|---|---|---|---|---|---|"]
    rows = [json.loads(l) for l in (D / "metrics_dual_split_s0").read_text().splitlines() if '"val_jepa"' in l]
    for r in rows:
        if r["step"] in (5000, 10000, 20000):
            d = ds_val.get(r["step"], {})
            L.append(f"| {r['step']} | {r['val_mse']:.4f} | {d.get('val_mse', float('nan')):.4f} | {r['val_ppd']:.3f} | {d.get('val_nll', float('nan')):.3f} | {r['val_tgt']['erank']:.1f} | {r['val_tgt']['dim_std_min']:.4f} |")
    d40 = ds_val.get(40000, {})
    L.append(f"| ds at 40000 | - | {d40.get('val_mse', float('nan')):.4f} | - | {d40.get('val_nll', float('nan')):.3f} | - | - |")
    L.append("")
    L.append("Caveat: the architectures are not matched (dual_split = 6 encoder + 4 predictor blocks with the head on the predictor; ds = 6 blocks with the head on the encoder), one seed, training policy mixed vs any_cell. The matched control (same split, lambda_jepa 0: configs/v4_mae_ctx_tw4_lr5e-4_mixed_cell_s0.yaml) has not been run.")
SC = D / "results_suite_compare.json"
if SC.exists():
    sc = json.loads(SC.read_text())
    L += ["", f"## 5. Real data (E9 protocol: CC18 + Grinsztajn + TabArena, {sc['n_datasets']} datasets, K_FEAT 64, CTX 1024): {sc['run']} vs references, per-dataset exact sign test", "",
          "| reference | task/metric | n | wins / losses | p (sign) | mean run | mean ref | delta |", "|---|---|---|---|---|---|---|---|"]
    for ref, rec in sc["vs"].items():
        for tm, r in rec.items():
            L.append(f"| {ref} | {tm} | {r['n']} | {r['wins']} / {r['losses']} | {r['p_sign']} | {r['mean_run']:.4f} | {r['mean_ref']:.4f} | {r['mean_delta']:+.4f} |")
    L.append("")
    L.append("Inference path of the run under test: x-encoder (observed cells only) -> predictor (mask tokens) -> bar head (model.jepa.ValueModel); the y-encoder is training-only. real_ds_lr5e-4_s0 = the same-step (20k) ds arm; the 40k rows are ds at twice the steps and E6's latent-init two-stage arm.")
L += ["", "## Verdict", "",
      "1. **cell_ctx (I-JEPA split, rope, shared code)**: the target is a column-address code. Column one-hot explains its variance,",
      "   value and true f explain none, nothing about the value can be read back, and both probe read-outs sit at the constant map.",
      "   The JEPA loss falls toward zero because the predictor reproduces the address from its rope position. Variance-collapse",
      "   monitors (dim_std, cos, within-table var) all read healthy: this is a shortcut, not a collapse.",
      "2. **cell (in-fill control)**: the target is a near-invertible code of the observed value (Issue #19's regime); the predictor",
      "   recovers a quarter of it. The cell read-out equals E1's pure-latent CLS-target cell to three decimals, so the",
      "   CLS-to-cell target swap changes nothing about the latent arm's f read-out; the ds reference stays far ahead.",
      "3. **cell_freshr (A: the target encoder draws its own column codes)**: the target's address component becomes",
      "   unpredictable and the EMA target collapses to a constant within a few thousand steps (dim_std 0, cos 1, prediction",
      "   constant). The shared address was what pulled the target out of its initial near-collapse; removing it leaves collapse",
      "   as the absorbing state. Stopped early; the box was destroyed.",
      *(["4. **cell_ctx_diff (difference target)**: see the rows above; the verdict for this run is stated in",
         "   docs/current_status.md session (24b) from these numbers."] if HAVE_DIFF else []),
      *(["5. **dual_split (diff target + value head on the predictor)**: sections 2-4; the verdict is stated in",
         "   docs/current_status.md session (24c) from these numbers."] if HAVE_DS else []),
      "",
      "Net: on this substrate a cell latent target is either an address code (predictor-side prediction), a value code",
      "(encoder-side in-fill), or a constant (address decorrelated). None carries information beyond the value, which is the",
      "data-space target already. The structural change asked for at the start of the session was implemented with the",
      "existing arms untouched (flags default off; 54 tests) and did not open a latent advantage."]
(D / "verdict.md").write_text("\n".join(L) + "\n")

# ---- summary_ko.md: the numbers docs/current_status.md points at (docs quote none by hand) ----
K = ["# I-JEPA 분리 구조 스모크 — 수치 요약 (build_verdict.py 생성, 손타이핑 0)", ""]
ctx, inf = tc[runs["ctx"]], tc[runs["infill"]]
K.append(f"- cell_ctx (rope, 공유 코드, 20k): 타깃 열주소 R² {ctx['target']['col']:.2f} · 값 R² {ctx['target']['z']:.2f} · 타깃→값 복원 {ctx['reverse']['z_from_target']:.2f}; 프로브 cell {probe['arms'][runs['ctx']]['any_cell']['cell']['mse_f']:.3f} / pred {probe['arms'][runs['ctx']]['any_cell']['pred']['mse_f']:.3f} (상수맵 {probe['const_map_mse_f']['any_cell']:.3f})")
K.append(f"- cell (채움 대조, 20k): 타깃→값 복원 {inf['reverse']['z_from_target']:.2f} · 예측→값 {inf['reverse']['z_from_pred']:.2f}; 프로브 cell {probe['arms'][runs['infill']]['any_cell']['cell']['mse_f']:.3f}; ds 참조 cell {probe['arms'][runs['ds']]['any_cell']['cell']['mse_f']:.3f}")
fr = val("cell_freshr_s0"); K.append(f"- cell_freshr (A, {fr['step']} 에서 중단): 타깃 dim_std {fr['val_tgt']['dim_std']:.3f} · cos {fr['val_tgt']['cos']:.3f} · 예측 cos {fr['val_pred']['cos']:.3f} = 상수 붕괴")
if HAVE_DIFF:
    dd, dm = tcd[runs["diff"]], tcd_mid[runs["diff"]]; dv = val("cell_ctx_diff_s0")
    K.append(f"- cell_ctx_diff (20k): 타깃 열주소 R² 중간 {dm['target']['col']:.2f} → 종점 {dd['target']['col']:.2f} · 타깃→값 복원 {dd['reverse']['z_from_target']:.2f} · 예측→값 {dd['reverse']['z_from_pred']:.2f}; val_jepa {dv['val_jepa']:.2f}; 타깃 erank {dv['val_tgt']['erank']:.0f} · std_min {dv['val_tgt']['dim_std_min']:.2f}; 프로브 pred {probe_d['arms'][runs['diff']]['any_cell']['pred']['mse_f']:.3f}")
if HAVE_DS:
    ds_, dsm = tcs[runs["dualsplit"]], tcs_mid[runs["dualsplit"]]; sv = val("dual_split_s0")
    rows = {json.loads(l)["step"]: json.loads(l) for l in (D / "metrics_dual_split_s0").read_text().splitlines() if '"val_jepa"' in l}
    K.append(f"- dual_split (20k): val_mse 5k {rows[5000]['val_mse']:.3f} / 10k {rows[10000]['val_mse']:.3f} / 20k {rows[20000]['val_mse']:.3f} vs ds {ds_val[5000]['val_mse']:.3f} / {ds_val[10000]['val_mse']:.3f} / {ds_val[20000]['val_mse']:.3f} (ds 40k {ds_val[40000]['val_mse']:.3f}); bar CE {sv['val_ppd']:.2f} vs ds {ds_val[20000]['val_nll']:.2f}; 타깃 열주소 R² {ds_['target']['col']:.2f} · 값 R² {ds_['target']['z']:.2f} · 타깃→값 복원 {ds_['reverse']['z_from_target']:.2f} · 예측→값 {ds_['reverse']['z_from_pred']:.2f}; 타깃 erank {sv['val_tgt']['erank']:.0f} · std_min {sv['val_tgt']['dim_std_min']:.2f} · 표내분산 {sv['val_tgt']['within_table_var_frac']:.2f}; 프로브 pred {probe_s['arms'][runs['dualsplit']]['any_cell']['pred']['mse_f']:.3f}")
CV = D / "results_convergence.json"
if CV.exists():
    cv = json.loads(CV.read_text())["dual_split"]
    def c(name): r = cv[name]["convergence"]; return f"{name} Q3→Q4 {r['rel_drop']*100:+.1f}% ({'수렴' if r['converged'] else '미수렴'})"
    K.append("- dual_split 수렴(E7 규칙): " + " · ".join(c(n) for n in ("train jepa (pred_loss)", "val_jepa", "val_ppd", "val_mse", "val_tgt.erank")))
    cvd = json.loads(CV.read_text())["ds_40k (ref)"]["val_mse"]["convergence"]; K.append(f"- ds 40k 참조 val_mse Q3→Q4 {cvd['rel_drop']*100:+.1f}% ({'수렴' if cvd['converged'] else '미수렴'})")
if SC.exists():
    for ref, rec in sc["vs"].items():
        K.append(f"- 실데이터 {sc['n_datasets']}셋 vs {ref}: " + " · ".join(f"{tm} {r['wins']}:{r['losses']} (p {r['p_sign']}, Δ {r['mean_delta']:+.3f})" for tm, r in rec.items()))
(D / "summary_ko.md").write_text("\n".join(K) + "\n")
print("\n".join(L[:3])); print("... wrote", D / "verdict.md")
