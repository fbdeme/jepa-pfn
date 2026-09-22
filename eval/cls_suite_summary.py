"""Classification-head verdict on the real suite (docs/real_prior_plan.md section 11, pre-registered).

Inputs (all eval/):
  results_suite_cls_*.json   suite passes for the class-head arms (PFN_RUNS=cls_ds_..., cls_dual_...; SUITE_TASKS=clf; base
                             protocol CTX 512 / K_FEAT 32 so base_ds is comparable) -- env CLS_GLOB overrides the glob
  results_suite_real.json    the real-prior bar-head arms (own protocol, 1024/64) -- ds_mean / dual_mean per key
  results_suite_full.json    base arms (512/32): base_ds, base_dual, logreg, histgb
  results_class_count_census.json   n_classes per clf key
Output eval/results_suite_cls.json: per-dataset table + verdict
  rule 1: cls_ds beats real_ds AND base_ds by paired sign test (p < .05, wins > losses) on the clf datasets, and in the
          6-10-class bucket the gap (linear - arm) at least halves relative to real_ds
  rule 2: cls_dual beats cls_ds by paired sign test (real-data half of the latent question) AND on the synthetic held-out
          stream (results_cls_train_curves.json: dual cls_nll_last < ds nll_last for a majority of seeds and on the mean)
Run: uv run python -m eval.cls_suite_summary
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.probe_real_prior import sign_test          # noqa: E402

OUT = ROOT / "eval/results_suite_cls.json"


def arm_mean(models, prefix):
    v = [m["acc"] for k, m in models.items() if k.startswith(prefix) and "acc" in m]
    return float(np.mean(v)) if v else None


def bucket(k):
    return None if k is None else ("2" if k == 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else ">10")


def paired(rows, a, b):
    d = [r[a] - r[b] for r in rows if r.get(a) is not None and r.get(b) is not None]
    w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return {"n": len(d), "wins": w, "losses": l, "p": sign_test(l, w), "mean_gap": round(float(np.mean(d)), 4) if d else None}


def synthetic_half(ds_lr="5e-4", dual_lr="5e-4"):
    """Same val seed stream (10000+seed), same policy y_target: ds val_nll vs dual cls_nll at the last logged step, per seed,
    with the stream's majority / logistic baselines (eval/results_cls_val_baselines.json) when present."""
    p = ROOT / "eval/results_cls_train_curves.json"
    if not p.exists():
        return None
    runs = json.loads(p.read_text())["runs"]
    bp = ROOT / "eval/results_cls_val_baselines.json"
    base = json.loads(bp.read_text())["seeds"] if bp.exists() else {}
    pairs = []
    for seed in range(3):
        ds, du = runs.get(f"cls_ds_lr{ds_lr}_s{seed}", {}).get("summary"), runs.get(f"cls_dual_lr{dual_lr}_s{seed}", {}).get("summary")
        if ds and du and ds.get("steps") == 20000 and du.get("steps") == 20000 and "nll_last" in ds and "cls_nll_last" in du:
            b = base.get(str(seed), {})
            pairs.append({"seed": seed, "ds_nll": ds["nll_last"], "dual_nll": du["cls_nll_last"],
                          "ds_err": ds["err_last"], "dual_err": du["cls_err_last"], "dual_lower_nll": du["cls_nll_last"] < ds["nll_last"],
                          "majority_err": b.get("majority_err"), "logreg_err": b.get("logreg_err")})
    if not pairs:
        return None
    w = sum(q["dual_lower_nll"] for q in pairs)
    mean_gap = round(float(np.mean([q["ds_nll"] - q["dual_nll"] for q in pairs])), 4)
    return {"pairs": pairs, "dual_seed_wins": w, "n": len(pairs), "mean_nll_gap_ds_minus_dual": mean_gap,
            "dual_wins": bool(w * 2 > len(pairs) and mean_gap > 0)}


def main():
    cls = {}
    for p in sorted(glob.glob(os.environ.get("CLS_GLOB", str(ROOT / "eval/results_suite_cls_*.json")))):
        for k, v in json.loads(Path(p).read_text()).items():
            cls.setdefault(k, {}).update(v)
    real = {r["key"]: r for r in json.loads((ROOT / "eval/results_suite_real.json").read_text())["datasets"]}
    base = json.loads((ROOT / "eval/results_suite_full.json").read_text())
    ncls = {r["key"]: r.get("n_classes") for r in json.loads((ROOT / "eval/results_class_count_census.json").read_text())["datasets"]}
    rows = []
    for k, models in cls.items():
        if not k.endswith("|clf"):
            continue
        r = {"key": k, "n_classes": ncls.get(k), "bucket": bucket(ncls.get(k)),
             "cls_ds": arm_mean(models, "cls_ds_lr5e-4"), "cls_dual": arm_mean(models, "cls_dual_lr5e-4"),       # registered arms only
             "ctrl_ds": arm_mean(models, "cls_ds_lr1.7e-4"), "ctrl_dual": arm_mean(models, "cls_dual_lr1.7e-4"),   # 11.2 lr re-match (ds: 3 seeds; dual: seed 0), never enters rules 1-2
             "grid_ds_5e-5": arm_mean(models, "cls_ds_lr5e-5"),                                                    # 11.2 grid point (seed 0)
             "linear_cls": models.get("logreg", {}).get("acc"), "histgb_cls": models.get("histgb", {}).get("acc"),
             "real_ds": real.get(k, {}).get("ds_mean"), "real_dual": real.get(k, {}).get("dual_mean"),
             "base_ds": base.get(k, {}).get("base_ds", {}).get("acc"), "base_dual": base.get(k, {}).get("base_dual", {}).get("acc"),
             "linear_base": base.get(k, {}).get("logreg", {}).get("acc")}
        rows.append(r)
    v = {"cls_ds_vs_real_ds": paired(rows, "cls_ds", "real_ds"), "cls_ds_vs_base_ds": paired(rows, "cls_ds", "base_ds"),
         "cls_dual_vs_cls_ds": paired(rows, "cls_dual", "cls_ds"), "cls_dual_vs_real_dual": paired(rows, "cls_dual", "real_dual"),
         "ctrl_ds_vs_cls_ds": paired(rows, "ctrl_ds", "cls_ds"), "ctrl_dual_vs_ctrl_ds": paired(rows, "ctrl_dual", "ctrl_ds")}
    by_bucket = {}
    for b in ("2", "3-5", "6-10", ">10"):
        rs = [r for r in rows if r["bucket"] == b and r.get("cls_ds") is not None]
        if not rs:
            continue
        m = lambda key: round(float(np.mean([r[key] for r in rs if r.get(key) is not None])), 4)
        by_bucket[b] = {"n": len(rs), "cls_ds": m("cls_ds"), "cls_dual": m("cls_dual") if any(r.get("cls_dual") is not None for r in rs) else None,
                        "real_ds": m("real_ds"), "base_ds": m("base_ds"), "linear": m("linear_cls"), "histgb": m("histgb_cls"),
                        "gap_linear_cls_ds": round(m("linear_cls") - m("cls_ds"), 4), "gap_linear_real_ds": round(m("linear_cls") - m("real_ds"), 4)}
    g = by_bucket.get("6-10")
    halved = bool(g and g["gap_linear_cls_ds"] <= 0.5 * g["gap_linear_real_ds"])
    r1 = all(v[a]["wins"] > v[a]["losses"] and v[a]["p"] < 0.05 for a in ("cls_ds_vs_real_ds", "cls_ds_vs_base_ds")) and halved
    r2_real = v["cls_dual_vs_cls_ds"]["wins"] > v["cls_dual_vs_cls_ds"]["losses"] and v["cls_dual_vs_cls_ds"]["p"] < 0.05
    syn = synthetic_half()
    r2 = r2_real and bool(syn and syn["dual_wins"])
    # 11.2 lr re-match: ds at 1.7e-4 (3 seeds, column ctrl_ds) against the registered dual (5e-4) and the same references
    rm = {"ctrl_ds_vs_real_ds": paired(rows, "ctrl_ds", "real_ds"), "ctrl_ds_vs_base_ds": paired(rows, "ctrl_ds", "base_ds"),
          "cls_dual_vs_ctrl_ds": paired(rows, "cls_dual", "ctrl_ds")}
    rm_syn = synthetic_half(ds_lr="1.7e-4", dual_lr="5e-4")
    rm_rows = [r for r in rows if r["bucket"] == "6-10" and r.get("ctrl_ds") is not None]
    rm_gap = round(float(np.mean([r["linear_cls"] - r["ctrl_ds"] for r in rm_rows])), 4) if rm_rows else None
    rm_halved = bool(rm_gap is not None and g and rm_gap <= 0.5 * g["gap_linear_real_ds"])
    rm_r1 = all(rm[a]["wins"] > rm[a]["losses"] and rm[a]["p"] < 0.05 for a in ("ctrl_ds_vs_real_ds", "ctrl_ds_vs_base_ds")) and rm_halved
    rm_r2_real = rm["cls_dual_vs_ctrl_ds"]["wins"] > rm["cls_dual_vs_ctrl_ds"]["losses"] and rm["cls_dual_vs_ctrl_ds"]["p"] < 0.05
    rematch = {"ds_lr": "1.7e-4", "dual_lr": "5e-4", "n_ctrl_ds_seeds": len({m for k in cls for m in cls[k] if m.startswith("cls_ds_lr1.7e-4")}),
               "paired": rm, "synthetic_heldout": rm_syn, "gap_linear_ctrl_ds_6to10": rm_gap,
               "verdict": {"rule1_head_is_main_cause": bool(rm_r1), "rule1_6to10_gap_halved": rm_halved,
                           "rule2_real_half": bool(rm_r2_real), "rule2_synthetic_half": bool(rm_syn and rm_syn["dual_wins"]),
                           "rule2_latent_clue": bool(rm_r2_real and rm_syn and rm_syn["dual_wins"])}}
    out = {"protocol": "clf datasets only; cls arms scored at CTX 512 / K_FEAT 32 (base protocol); arms = mean over seeds; paired sign tests",
           "paired": v, "by_class_bucket": by_bucket,
           "synthetic_heldout": syn, "rematch_11_2": rematch,
           "verdict": {"rule1_head_is_main_cause": bool(r1), "rule1_6to10_gap_halved": halved,
                       "rule2_real_half": bool(r2_real), "rule2_synthetic_half": bool(syn and syn["dual_wins"]), "rule2_latent_clue": bool(r2)},
           "datasets": rows}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(v)); print("by bucket:", json.dumps(by_bucket)); print("verdict:", out["verdict"])
    print("rematch 11.2:", json.dumps({k: rematch[k] for k in ("n_ctrl_ds_seeds", "paired", "gap_linear_ctrl_ds_6to10", "verdict")}))
    print("rematch synthetic:", json.dumps(rm_syn)); print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
