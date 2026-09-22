"""Render docs/reframe_analysis.md: the evidence table is injected from eval/results_*.json (no hand-typed numbers); the prose
is the judge's analysis (2026-09-07). Re-run after any artifact changes:  uv run python scripts/gen_reframe_analysis.py"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
J = lambda p: json.load(open(ROOT / p))


def evidence_rows():
    e1, e3, e6, e7, e8, e9 = (J(f"eval/results_v4_{n}.json") for n in ("e1_collapse_grid", "e3_stress_benefit", "e6_pretrain_finetune", "e7_convergence", "e8_pretrain_budget", "e9_realdata"))
    e10 = J("eval/results_v4_e10_objective_control.json"); eq = J("eval/results_v4_equal_steps.json")
    lat = J("eval/results_lat_pretrain_trajectory.json")["runs"]; b20 = J("eval/results_v4_e7_baseline_20k.json"); post = J("eval/results_probe_masked_all.json")["post"]
    t1, pa, t3 = e1["test"], e3["parity"], e3["test"]; pr = e6["pairs"]; gs = e7["gap_by_steps"]; gb = e8["gap_by_budget"]; sp = e8["spearman"]; A = e9["anchors"]
    c20 = e9["pairs"]["finetune_20k_vs_scratch_20k"]; c40 = e9["pairs"]["finetune_40k_vs_scratch_40k"]; bl = e1["best_cell_lat"]; fl = e1["probe_floor"]
    lr_c = sorted({c["lr"] for c in e1["cells"].values() if c["n_collapsed"]}, key=float); lr_a = sorted({c["lr"] for c in e1["cells"].values() if c["all_alive"]}, key=float)
    dsr = sorted(r for r in post["arms"] if "_ds_" in r and ("lr5e-4" in r or "lr1.7e-4" in r)); ds_mf = [post["arms"][r]["any_cell"]["cell"]["mse_f"] for r in dsr]
    return [
        ("E1 붕괴 격자 (27런)", f"붕괴 λ0 {t1['collapsed'][0]}/{t1['n'][0]} vs λ.1 {t1['collapsed'][1]}/{t1['n'][1]}, Fisher p {t1['p']}; 붕괴가 난 lr = {', '.join(lr_c)}, 전 시드 생존 lr = {', '.join(lr_a)}", "값 헤드는 붕괴를 막지 않는다; 붕괴는 lr 의 함수"),
        ("순수 lat 프로브 (E1 λ0 셀)", f"masked f-MSE any_cell {bl['masked_f_any']} vs floor {fl['any_cell']['mean']} vs ds {min(ds_mf)}–{max(ds_mf)}", "잠재 표현은 값을 거의 담지 않는다(floor 근처)"),
        ("E3 dual vs ds (9+9런)", f"실패 ds {t3['failed'][0]}/{t3['n'][0]} vs dual {t3['failed'][1]}/{t3['n'][1]} (p {t3['p']}, 방향 반대); |값 격차| {pa['mean_abs_gap']} > 구간 {pa['margin']}", "결합 잠재 손실은 안정성도 동등성도 주지 않는다"),
        ("사전학습 궤적 (λ0 20k ×3)", "; ".join(f"s{r[-1]} erank {' → '.join(str(int(round(x))) for x in t['erank'].values())}, val_jepa 최소 {t['val_jepa_min']}@{t['val_jepa_min_step']} → {t['val_jepa']['1.0']}" for r, t in lat.items()), "JEPA 손실은 준붕괴 초반에 최소·이후 상승(비정상); 표현은 20k 에도 진화 중"),
        ("E6 사전학습 → 헤드 (15런)", f"격차(팔−scratch) finetune 5k {pr['finetune_5k']['mean_gap_value']}, finetune 20k {pr['finetune_20k']['mean_gap_value']}, frozen {pr['frozen_5k']['mean_gap_value']} (전 시드 규칙: helps {pr['finetune_5k']['helps']}/{pr['finetune_20k']['helps']}/{pr['frozen_5k']['helps']})", "이득은 초기화에서 온다; 고정 표현은 부족"),
        ("E7 수렴 대조 (12런 + 20k)", "격차 추이 " + " → ".join(f"{int(k) // 1000}k {v['mean']}" for k, v in gs.items()) + f"; 판정 {e7['verdict']} (미수렴 {len(e7['unconverged'])}/6, 20k 기준선 수렴 {b20['n_converged']}/{b20['n']})", "예산 3배에도 격차 유지·확대 = 빠른 도착이 아님(수렴 주장은 불가)"),
        ("E8 사전학습 예산 (21런)", " → ".join(f"{k}: {v['mean']}" for k, v in gb.items()) + f"; Spearman erank ρ {sp['erank']['rho']} (p {sp['erank']['p']}), dim_std_min ρ {sp['dim_std_min']['rho']} (p {sp['dim_std_min']['p']}), val_jepa ρ {sp['val_jepa']['rho']} (p {sp['val_jepa']['p']}); 포화 {e8['verdict']['saturation_budget']}", "이득은 사전학습 예산에 단조; erank 가 예측, 손실은 무관; 손실 최소 스냅샷은 해로움"),
        ("E10 목적 대조 (9런)", f"ds-mixed 사전학습 → 헤드 재초기화 5k finetune: 20k 격차 {e10['by_budget']['20000']['gap_ds_init']} vs 잠재 {e10['by_budget']['20000']['gap_lat_init']}; 40k {e10['by_budget']['40000']['gap_ds_init']} vs {e10['by_budget']['40000']['gap_lat_init']}; 판정 {e10['verdict']}", "초기화 이득은 잠재 특이적이 아니다(ds-mixed ≥ 잠재)"),
        ("같은 총 스텝 비교", "; ".join(f"{r['arm']} vs scratch {r['scratch_steps']//1000}k: 평균 {r['mean_gap']:+.3f}" for r in eq["rows"]), "finetune 5k 두 단계는 같은 총 스텝의 scratch 에 진다; finetune ≥ 20k 잠재 두 단계만 이긴다(ds-mixed 는 미검증 = E10b)"),
        ("E9 실데이터 147셋", f"clf 풀링 승:패 {c20['clf']['pooled']['wins']}:{c20['clf']['pooled']['losses']} (20k, p {c20['clf']['pooled']['p']}) · {c40['clf']['pooled']['wins']}:{c40['clf']['pooled']['losses']} (40k, p {c40['clf']['pooled']['p']}); 시드별 clf p(40k) {[v['p'] for v in c40['clf']['per_seed'].values()]}; reg 풀링 {c40['reg']['pooled']['wins']}:{c40['reg']['pooled']['losses']}; 평균 acc 40k {A['finetune_40k']['clf']['mean_metric']} vs {A['scratch_40k']['clf']['mean_metric']}, HistGB {e9['histgb_mean']['clf']}; 규칙 판정 clf {e9['verdict']['real_gain_clf']}", "전이는 양성이나 작고 clf 시드 비강건; 절대 수준은 GBDT 에 크게 못 미침"),
    ]


PROSE = """
## 2. 데이터가 지지하는 주장 / 지지하지 않는 주장 (E10 이후 갱신)

**지지되는 것**
1. 결합 잠재 손실(dual)은 ds 를 개선·안정화하지 않고(E3, §30), 값 헤드는 붕괴를 막지 않으며(E1), 잠재 표현은 값을 담지 않는다(프로브·frozen).
2. 마스킹을 바꾼 **두 단계 학습**(사전학습 → any_cell finetune)은 finetune 이 길 때(≥ 20k) 같은 총 스텝의 단일 단계 scratch 를 이기고 격차가 예산과 함께 커진다(E7 step-matched, 잠재 사전학습으로 확인).
3. 그 이득은 **잠재 목적에 특이적이지 않다**: 같은 마스킹의 데이터공간 CE 사전학습이 같거나 더 큰 이득을 준다(E10, 5k finetune 체제).
4. 짧은 finetune(5k)에서 보이는 사전학습 예산 곡선(E8)은 대부분 총 학습량의 효과다(같은 총 스텝의 scratch 가 더 좋음).
5. 실데이터 전이는 방향이 양성이나 작고 시드 비강건(E9).

**지지되지 않는 것**
- "잠재 예측의 가치 = 초기화"(E10 로 기각) · "erank 가 이득을 예측"(예산과 공변, 층화 없이는 무효) · 수렴점 비교 · 시드 강건한 실데이터 이득.

## 3. 재프레임 후보와 판단 (갱신)

| 후보 | 골자 | 판단 |
|---|---|---|
| B(이전 추천) "잠재 가치의 위치 = 초기화" | E6–E8 긍정 결말 | **기각** — E10 이 초기화 이득의 잠재 특이성을 부정 |
| **D. 부정 결과 + 일반 두 단계 학습**: "Latent prediction adds nothing over data-space self-supervision in tabular ICL — but a masking-switch two-stage schedule does" | paper_v3 의 부정 척추(E1·E3·프로브) + 두 단계 학습의 긍정(E7 step-matched·E10)을 잠재와 분리해 서술 | **1순위** — 모든 실험이 사전 등록 규칙으로 판정된 정직한 이야기; 단, E10b 로 긴 finetune 에서도 ds-mixed 가 잠재와 같은지 확인해야 문장이 닫힘 |
| E. 순수 부정 논문 | 잠재 예측은 어느 경로에서도 무효 | 2순위 — 두 단계 학습의 긍정 발견을 버림 |

**추천 = D, 단 E10b 뒤에.** E10b 가 "ds-mixed 도 같다"면 D 그대로(제목은 잠재 부정 + 두 단계 긍정), "잠재만 이긴다"면 긴 finetune 한정으로 잠재 특이성이 되살아나 B 의 축소판이 된다.

## 4. D 의 뼈대

1. 질문: 잠재 예측이 표 ICL 에 값을 주는가 — 결합 손실 / 표현 / 초기화 세 경로.
2. 계측(재사용).
3. 결합 손실 무효(E3·§30) · 표현은 값이 아님(프로브·frozen) · 붕괴는 lr(E1).
4. 초기화: 두 단계가 긴 finetune 에서 같은 총 스텝의 scratch 를 이김(E7 step-matched) — 그러나 데이터공간 사전학습도 같거나 더 좋음(E10) → 이득은 마스킹 전환·재시작에서 온다.
5. 사전학습 예산·짧은 finetune(E8)은 총 학습량 효과로 재해석; 실데이터(E9)는 양성·비강건.
6. 한계: 미수렴·시드 3·소형·한 prior; E10b 결과에 따라 4 의 문장 확정.

## 5. 남은 실험

1. **E10b(최우선, ≈$2)**: ds-mixed 20k 스냅샷 → any_cell 40k finetune ×3 (총 60k) vs lat20k+ft40k·scratch 60k. 긴 finetune 체제의 목적 특이성을 닫는다.
2. (선택) 마스킹 전환 자체가 원인인지: any_cell 20k → any_cell 40k(= scratch 60k 와 동일하므로 이미 있음) vs mixed 20k → any_cell 40k(E10b) — E10b 가 곧 이 대조.

## 6. paper_v3 에서 버릴 것

- "dual 이 레시피", "잠재 항이 안정성을 산다", parity, "값 헤드 = 붕괴 방지", 그리고 (E10 이후) "잠재 사전학습 초기화가 특이적 이득" 류 문장.
"""


def main():
    rows = evidence_rows()
    tbl = "| 실험 | 정본 수치(아티팩트) | 한 줄 해석 |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    doc = ("# 논문 재프레임 분석 (보고용, 2026-09-07, E10 반영)\n\n"
           "> 수치는 전부 `eval/results_*.json` 에서 `scripts/gen_reframe_analysis.py` 가 주입(손타이핑 0). 결정론 검증 = `uv run python -m eval.v4_determinism_check` PASS 시점의 아티팩트.\n"
           "> 이 문서는 **분석·제안**이며 논문(`paper_v3/`, `paper_v4/`)은 수정하지 않았다.\n\n## 1. 데이터가 말하는 것 (E1–E9 요약)\n\n" + tbl + "\n" + PROSE)
    (ROOT / "docs/reframe_analysis.md").write_text(doc)
    print("wrote docs/reframe_analysis.md", len(doc), "chars,", len(rows), "evidence rows")


if __name__ == "__main__":
    main()
