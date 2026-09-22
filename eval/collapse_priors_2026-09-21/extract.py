"""The predictor-read wiring (head_on_pred: true) collapsed on every prior it was trained on, not only the paper's
(reviewer round 1, R5 follow-up 2026-09-21): three fixed-budget runs, one per prior, plus the encoder-read run on the
paper's prior as the reference. For each run: held-out value MSE at the end and at its minimum, the constant-map MSE of
its own held-out stream (recomputed as eval/open_horizon_2026-09-14/extract_curves.py does), target rank and cosine at
the end, the latent-loss minimum, and the suite mean accuracy / R^2 from the committed eval/results_suite_<run>.json.
Asserts that the three predictor-read configs differ only in the prior. Writes results_collapse_priors.json next to this file.
Run: .venv/bin/python eval/collapse_priors_2026-09-21/extract.py"""
import json, sys
from pathlib import Path

import yaml

R = Path(__file__).resolve().parents[2]; D = Path(__file__).parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "eval/open_horizon_2026-09-14"))
from extract_curves import held_out_cells  # noqa: E402

RUNS = {"TabularJEPA_v3_scm_w64_s0": "pred", "TabularJEPA_v3_tabicl1_s0": "pred", "TabularJEPA_v3_tabicl2_s0": "pred",
        "TabularJEPA_v3_tabicl2_headenc_s0": "enc"}
PRIOR_KEYS = {"run_name", "prior_source", "prior"}          # the only keys the three predictor-read configs may differ in
RULES = {"back_to_const_frac": 0.98,    # value error is "back at the constant map" at the first validation after its minimum at >= this fraction of the map
         "pred_cos_flat": 0.9,          # the predictor output is "flat" at the first validation whose mean pairwise cosine over held-out cells is back
                                        # at >= this after having been below it (an untrained predictor starts near-constant, so the first crossing is armed by a dip)
         "tgt_rank_collapse": 10.0}     # the target field has "lost rank" at the first validation whose effective rank < this (ceiling = emb)


def first_step(va, pred):
    return next((x["step"] for x in va if pred(x)), None)


def prior_label(cfg):
    if cfg.get("prior_source") == "tabicl":
        return {"mix_scm": "tabicl_v1", "graph_scm": "tabicl_v2"}[cfg["prior"]["prior_type"]]
    return "scm"                                              # this repository's own SCM prior (train/train_jepa.py:_make_prior default)


def suite_means(run):
    S = json.loads((R / f"eval/results_suite_{run}.json").read_text())
    out = {}
    for task, m in (("clf", "acc"), ("reg", "r2")):
        v = [d[run][m] for k, d in S.items() if k.endswith("|" + task) and run in d]
        out[task] = dict(n=len(v), mean=round(sum(v) / len(v), 4))
    return out


def main():
    out, cfgs = {}, {}
    for r, arm in RUNS.items():
        cfg = yaml.safe_load((R / "runs" / r / "config.yaml").read_text()); cfgs[r] = cfg
        va = [json.loads(l) for l in (R / "runs" / r / "metrics.jsonl").read_text().splitlines() if '"val_mse"' in l]
        cells, meta = held_out_cells(cfg, arm)
        import torch
        allz = torch.cat(cells); const = float(allz.var(unbiased=False))
        lo = min(va, key=lambda x: x["val_mse"]); last = va[-1]
        out[r] = dict(arm=arm, prior=prior_label(cfg), head_on_pred=cfg["head_on_pred"], lr=cfg["lr"], steps=cfg["steps"],
                      n_rows=cfg["prior"]["n_rows"], seed=cfg["seed"], val_tables=meta["val_tables"],
                      final_step=last["step"], final_val_mse=last["val_mse"], min_val_mse=lo["val_mse"], min_step=lo["step"],
                      const_map_mse=round(const, 4), final_over_const=round(last["val_mse"] / const, 3),
                      min_over_const=round(lo["val_mse"] / const, 3),
                      final_tgt_erank=last["val_tgt"]["erank"], final_tgt_cos=last["val_tgt"]["cos"],
                      min_val_jepa=min(x["val_jepa"] for x in va), suite=suite_means(r),
                      # when: the three events of a collapse, by the rules above (None = never happened in the run)
                      back_to_const_step=first_step(va, lambda x: x["step"] > lo["step"] and x["val_mse"] >= RULES["back_to_const_frac"] * const),
                      pred_cos_flat_step=(lambda dip: first_step(va, lambda x: dip is not None and x["step"] > dip and (x["val_pred"]["cos"] or 0) >= RULES["pred_cos_flat"]))(
                          first_step(va, lambda x: (x["val_pred"]["cos"] or 0) < RULES["pred_cos_flat"])),
                      tgt_rank_collapse_step=first_step(va, lambda x: x["val_tgt"]["erank"] < RULES["tgt_rank_collapse"]),
                      pred_erank_max=max(x["val_pred"]["erank"] for x in va if x["val_pred"]["erank"] is not None),
                      # how: the series the appendix figure draws (paper_v5/figures/make_fig_collapse.py)
                      val=[dict(step=x["step"], val_mse=x["val_mse"], val_jepa=x["val_jepa"], tgt_cos=x["val_tgt"]["cos"],
                                tgt_erank=x["val_tgt"]["erank"], pred_cos=x["val_pred"]["cos"], pred_erank=x["val_pred"]["erank"],
                                pred_dim_std=x["val_pred"]["dim_std"]) for x in va])
        print(r, out[r]["prior"], "final", last["val_mse"], "const", out[r]["const_map_mse"], "erank", out[r]["final_tgt_erank"], "suite", out[r]["suite"])
    preds = [r for r, a in RUNS.items() if a == "pred"]
    for r in preds[1:]:
        diff = {k for k in set(cfgs[preds[0]]) | set(cfgs[r]) if cfgs[preds[0]].get(k) != cfgs[r].get(k)}
        assert diff <= PRIOR_KEYS, (r, diff)
    out["_rules"] = RULES
    out["_note"] = "the three predictor-read configs differ only in " + ", ".join(sorted(PRIOR_KEYS)) + \
                   "; the encoder-read run differs from the tabicl_v2 predictor-read run in head_on_pred alone (build_docs asserts it)"
    (D / "results_collapse_priors.json").write_text(json.dumps(out, indent=1) + "\n"); print("wrote", D / "results_collapse_priors.json")


if __name__ == "__main__":
    main()
