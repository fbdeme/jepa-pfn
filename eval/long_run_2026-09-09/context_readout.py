"""Context for the 180k snapshot's real-data pass (2026-09-10 discussion: 'would 30M close the HistGB gap?'): the same 115
classification datasets scored by the paper-prior base_ds (eval/results_suite_full.json), the old big-vs-base ladder on their
shared datasets (eval/results_suite_big.json), and where the snapshot's gap to HistGB sits. All numbers computed from the
suite JSONs. Writes results_context.json.  Run: uv run python eval/long_run_2026-09-09/context_readout.py"""
import json
from pathlib import Path
from statistics import mean
D = Path(__file__).parent; ROOT = D.parents[1]
L = json.loads((D / "results_suite_real_long180k.json").read_text())
F = json.loads((ROOT / "eval/results_suite_full.json").read_text())
B = json.loads((ROOT / "eval/results_suite_big.json").read_text())
SNAP = [m for m in next(iter(L.values())) if m.startswith("v4_")][0]
def m_acc(P, model, keys):
    xs = [P[k][model]["acc"] for k in keys if k in P and model in P[k] and "acc" in P[k][model]]
    return {"mean_acc": round(mean(xs), 4), "n": len(xs)} if xs else None
clf = [k for k in L if k.endswith("|clf")]
shared = [k for k in B if k.endswith("|clf") and k in F]
gaps = sorted(((round(L[k]["histgb"]["acc"] - L[k][SNAP]["acc"], 4), k) for k in clf if "histgb" in L[k]), reverse=True)
out = {"snapshot": SNAP,
       "same_115_clf": {"snapshot_180k": m_acc(L, SNAP, clf), "logreg_this_pass": m_acc(L, "logreg", clf), "histgb_this_pass": m_acc(L, "histgb", clf),
                        "base_ds_paper_prior_20k": m_acc(F, "base_ds", clf), "base_dual_paper_prior_20k": m_acc(F, "base_dual", clf),
                        "logreg_paper_pass": m_acc(F, "logreg", clf), "histgb_paper_pass": m_acc(F, "histgb", clf),
                        "note": "paper pass = K_FEAT 32 / CTX 512 (eval/results_suite_full.json); this pass = K_FEAT 64 / CTX 1024; logreg and histgb agree across passes"},
       "big_vs_base_shared": {"n_datasets": len(shared), **{m: m_acc(B if m.startswith("big") else F, m, shared) for m in ("big_ds", "big_dual", "base_ds", "base_dual", "histgb")},
                              "big_config": "emb 512 / 8 layers / mlp 2048 (configs/big_ds.yaml), <= 40k steps at batch 2"},
       "gap_to_histgb": {"largest": [{"dataset": k, "gap": g} for g, k in gaps[:8]], "snapshot_wins": sum(1 for g, _ in gaps if g < 0), "n": len(gaps)}}
(D / "results_context.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["same_115_clf"], indent=1)); print("big vs base", out["big_vs_base_shared"]); print("gaps", out["gap_to_histgb"]["largest"][:4], out["gap_to_histgb"]["snapshot_wins"])
print("wrote", (D / "results_context.json").relative_to(ROOT))
