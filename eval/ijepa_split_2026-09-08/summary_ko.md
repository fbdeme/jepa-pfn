# I-JEPA 분리 구조 스모크 — 수치 요약 (build_verdict.py 생성, 손타이핑 0)

- cell_ctx (rope, 공유 코드, 20k): 타깃 열주소 R² 0.59 · 값 R² -0.00 · 타깃→값 복원 -0.01; 프로브 cell 0.979 / pred 0.976 (상수맵 0.966)
- cell (채움 대조, 20k): 타깃→값 복원 0.93 · 예측→값 0.24; 프로브 cell 0.903; ds 참조 cell 0.634
- cell_freshr (A, 8250 에서 중단): 타깃 dim_std 0.000 · cos 1.000 · 예측 cos 1.000 = 상수 붕괴
- cell_ctx_diff (20k): 타깃 열주소 R² 중간 0.39 → 종점 0.05 · 타깃→값 복원 0.78 · 예측→값 0.06; val_jepa 0.62; 타깃 erank 94 · std_min 0.41; 프로브 pred 0.920
- dual_split (20k): val_mse 5k 0.800 / 10k 0.730 / 20k 0.657 vs ds 0.858 / 0.802 / 0.772 (ds 40k 0.746); bar CE 1.93 vs ds 1.42; 타깃 열주소 R² -0.01 · 값 R² 0.47 · 타깃→값 복원 0.95 · 예측→값 0.26; 타깃 erank 20 · std_min 0.13 · 표내분산 0.99; 프로브 pred 0.734
- dual_split 수렴(E7 규칙): train jepa (pred_loss) Q3→Q4 +1.0% (수렴) · val_jepa Q3→Q4 +1.9% (수렴) · val_ppd Q3→Q4 +2.9% (미수렴) · val_mse Q3→Q4 +4.0% (미수렴) · val_tgt.erank Q3→Q4 +20.1% (미수렴)
- ds 40k 참조 val_mse Q3→Q4 +2.2% (미수렴)
- 실데이터 147셋 vs real_ds_lr5e-4_s0: clf_acc 61:35 (p 0.01035, Δ +0.013) · clf_auc 69:46 (p 0.03975, Δ +0.022) · reg_r2 13:19 (p 0.37709, Δ +0.001)
- 실데이터 147셋 vs v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0: clf_acc 28:73 (p 1e-05, Δ -0.034) · clf_auc 22:93 (p 0.0, Δ -0.040) · reg_r2 8:24 (p 0.007, Δ -0.062)
- 실데이터 147셋 vs v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s0: clf_acc 18:82 (p 0.0, Δ -0.048) · clf_auc 13:102 (p 0.0, Δ -0.055) · reg_r2 3:29 (p 0.0, Δ -0.099)
- 실데이터 147셋 vs histgb: clf_acc 13:101 (p 0.0, Δ -0.175) · clf_auc 6:109 (p 0.0, Δ -0.159)
- 실데이터 147셋 vs histgbr: reg_r2 0:32 (p 0.0, Δ -0.275)
