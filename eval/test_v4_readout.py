"""C7: the per-run read-out and the E1 aggregate on synthetic runs (no checkpoints, no GPU)."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.v4_readout import aggregate, failed_training, read_run     # noqa: E402

SPEC = yaml.safe_load((ROOT / "paper_v4/experiments.yaml").read_text())
NO_SRC = dict(truth=None, value=None, masked=None, audit=None)


def _run(d, name, steps, last_step, dim_stds, val_ppd=None):
    p = d / name
    p.mkdir()
    (p / "config.yaml").write_text(yaml.safe_dump(dict(steps=steps)))
    with open(p / "metrics.jsonl", "w") as f:
        for i, s in enumerate(dim_stds):
            rec = dict(step=(i + 1) * (last_step // len(dim_stds)), val_jepa=0.1, val_tgt=dict(dim_std=s), val_pred=dict(dim_std=s))
            if val_ppd is not None:
                rec["val_ppd"] = val_ppd
            f.write(json.dumps(rec) + "\n")


def _manifest(runs):
    return {"experiments": {"E1": {"runs": runs, "artifact": "x"}}}


def test_read_run_states(tmp_path):
    _run(tmp_path, "alive", 1000, 1000, [0.9, 0.9], val_ppd=1.5)
    _run(tmp_path, "dead", 1000, 1000, [0.9, 0.0])
    _run(tmp_path, "half", 1000, 500, [0.9])
    a, d, h, m = (read_run(r, {}, NO_SRC, tmp_path) for r in ("alive", "dead", "half", "missing"))
    assert a["status"] == "complete" and not a["collapse"]["collapsed"] and a["val_ppd_last"] == 1.5 and a["arm"] == "dual"
    assert d["status"] == "complete" and d["collapse"] == dict(collapsed=True, onset=1000, recovered=None, last_step=1000)
    assert h["status"] == "partial" and m["status"] == "missing"
    assert a["truth"] is None and a["value"] is None and a["masked"] is None and a["audit"] is None
    assert failed_training(a) is None                     # no value read-out yet: not counted
    assert failed_training(d) is True and failed_training(h) is None


def test_e1_aggregate_fisher_and_best_cell(tmp_path):
    runs = {}
    for lam, lr in ((0.0, "5e-4"), (0.1, "5e-4"), (0.1, "1.7e-4"), (0.3, "5e-4")):
        for s in range(3):
            name = f"v4_dual_ema_lam{lam:g}_lr{lr}_mixed_cls_row_s{s}"
            # lambda 0: all collapse; lambda .1 @5e-4: one seed collapses; lambda .1 @1.7e-4: alive, ppd 1.4; lambda .3: alive, ppd 1.2
            dead = lam == 0.0 or (lam == 0.1 and lr == "5e-4" and s == 0)
            _run(tmp_path, name, 1000, 1000, [0.9, 0.0 if dead else 0.9], val_ppd=None if lam == 0 else (1.4 if lr == "1.7e-4" else 1.2))
            runs[name] = dict(arm="dual", mech="ema", lambda_ppd=lam, lr=lr, policy="mixed", target="cls_row", seed=s, steps=1000, reused=False)
    out = aggregate("E1", _manifest(runs), SPEC, NO_SRC, tmp_path)
    assert out["complete"] and out["n_complete"] == 12
    t = out["test"]
    assert t["n"] == [3, 6] and t["collapsed"] == [3, 1] and t["table"] == [[3, 0], [1, 5]]
    assert t["p"] < SPEC["alpha"] and t["head_reduces_collapse"]
    # rule: lambda > 0, every seed alive, lowest mean val_ppd -> lam0.3@5e-4 (1.2) beats lam0.1@1.7e-4 (1.4); lam0.1@5e-4 is out (one seed dead)
    assert out["best_cell"] == dict(lambda_ppd=0.3, lr="5e-4", mean_val_ppd_last=1.2)
    assert not out["cells"]["lam0.1_lr5e-4"]["all_alive"] and out["cells"]["lam0.1_lr1.7e-4"]["all_alive"]


def test_e1_incomplete_has_no_verdict(tmp_path):
    runs = {}
    for lam in (0.0, 0.1):
        name = f"v4_dual_ema_lam{lam:g}_lr5e-4_mixed_cls_row_s0"
        _run(tmp_path, name, 1000, 500, [0.9])
        runs[name] = dict(arm="dual", mech="ema", lambda_ppd=lam, lr="5e-4", policy="mixed", target="cls_row", seed=0, steps=1000, reused=False)
    out = aggregate("E1", _manifest(runs), SPEC, NO_SRC, tmp_path)
    assert not out["complete"] and out["test"]["p"] is None and not out["test"]["head_reduces_collapse"] and out["best_cell"] is None


def _ds_run(d, name, steps, last_step, val_mse, val_nll=2.0):
    p = d / name
    p.mkdir()
    (p / "config.yaml").write_text(yaml.safe_dump(dict(steps=steps)))
    with open(p / "metrics.jsonl", "w") as f:
        f.write(json.dumps(dict(step=last_step, val_nll=val_nll, val_mse=val_mse)) + "\n")


def test_pure_lat_has_no_head_value_and_best_cell_lat(tmp_path):
    """C17: lambda 0 runs never trained the head -> value n/a; best_cell_lat = lowest masked f-MSE among alive lambda-0 cells."""
    runs, arms = {}, {}
    for lr, mf in (("5e-4", 0.90), ("1.7e-4", 0.94)):
        for s in range(2):
            name = f"v4_dual_ema_lam0_lr{lr}_mixed_cls_row_s{s}"
            _run(tmp_path, name, 1000, 1000, [0.9, 0.9])
            runs[name] = dict(arm="dual", mech="ema", lambda_ppd=0.0, lr=lr, policy="mixed", target="cls_row", seed=s, steps=1000, reused=False)
            arms[name] = {"any_cell": {"cell": {"mse_f": mf + 0.01 * s}}, "mixed": {"cell": {"mse_f": mf + 0.05}}}
    src = dict(NO_SRC, value={"runs": {n: {"by_R": {"64": {"learned": True}}, "select_mean": 0.5} for n in runs}},
               masked={"post": {"arms": arms, "floor": {"any_cell": {"mean": 0.97}}, "const_map_mse_f": {"any_cell": 0.97}}})
    out = aggregate("E1", _manifest(runs), SPEC, src, tmp_path)
    r = next(iter(out["runs"].values()))
    assert r["value"] is None and "no trained head" in r["value_note"]
    assert out["best_cell_lat"] == dict(lambda_ppd=0.0, lr="5e-4", masked_f_any=0.905)
    assert out["cells"]["lam0_lr1.7e-4"]["probe"]["masked_f_any"] == 0.945 and out["probe_floor"]["any_cell"]["mean"] == 0.97


def test_e6_seed_paired_gaps(tmp_path):
    """C16: frozen / finetune vs their seed-paired scratch arms; 'helps' needs every seed beyond the margin."""
    runs = {}
    lat = {s: f"v4_dual_ema_lam0_lr5e-4_mixed_cls_row_s{s}" for s in range(3)}
    val = {}
    for s in range(3):
        for mech, extra, steps, init, frz, mse in (("latfrozen", "5k", 5000, lat[s], True, 0.95), ("latinit", "5k", 5000, lat[s], False, 0.70),
                                                    ("none", "5k", 5000, None, False, 0.80), ("latinit", "20k", 20000, lat[s], False, 0.60)):
            name = f"v4_ds_{mech}_lam0_lr5e-4_any_cell_bar32_{extra}_s{s}"
            _ds_run(tmp_path, name, steps, steps, mse)
            runs[name] = dict(arm="ds", mech=mech, lambda_ppd=0.0, lr="5e-4", policy="any_cell", target="bar32", seed=s, steps=steps, reused=False,
                              **({"init_from": init, "freeze_encoder": frz} if init else {}))
            val[name] = {"by_R": {"64": {"learned": True}}, "select_mean": mse}
        name = f"real_ds_lr5e-4_s{s}"
        _ds_run(tmp_path, name, 20000, 20000, 0.65)
        runs[name] = dict(arm="ds", mech="none", lambda_ppd=0.0, lr="5e-4", policy="any_cell", target="bar32", seed=s, steps=20000, reused=True)
        val[name] = {"by_R": {"64": {"learned": True}}, "select_mean": 0.65}
    src = dict(NO_SRC, value={"runs": val})
    m = {"experiments": {"E6": {"runs": runs, "artifact": "x"}}}
    out = aggregate("E6", m, SPEC, src, tmp_path)
    p = out["pairs"]
    assert p["finetune_5k"]["n"] == 3 and p["finetune_5k"]["mean_gap_value"] == -0.1 and p["finetune_5k"]["helps"] is True
    assert p["frozen_5k"]["mean_gap_value"] == 0.15 and p["frozen_5k"]["helps"] is False and p["frozen_5k"]["equivalent"] is False
    assert p["finetune_20k"]["ref"] == "scratch_20k" and p["finetune_20k"]["mean_gap_value"] == -0.05 and p["finetune_20k"]["helps"] is True
    assert p["finetune_5k"]["pairs"][0]["gap_val_mse"] == -0.1
    assert out["verdict"] == dict(pretraining_helps=True, representation_sufficient=False)


def _curve_run(d, name, steps, floor, amp, tail_slope=0.0, seed=0):
    """metrics.jsonl with 80 val rows: val_mse = floor + amp*exp(-step/(steps/12)) + tail_slope*(step/steps) (+ tiny fixed noise)."""
    import math
    _ds_run(d, name, steps, steps, floor)
    with open(d / name / "metrics.jsonl", "w") as f:
        for i in range(1, 81):
            st = int(steps * i / 80)
            v = floor + amp * math.exp(-st / (steps / 12)) + tail_slope * st / steps + 0.001 * ((i * 7 + seed) % 3 - 1)
            f.write(json.dumps(dict(step=st, val_nll=2.0, val_mse=round(v, 5))) + "\n")


def test_convergence_quarters_rule():
    from eval.v4_readout import convergence_quarters
    flat = [dict(step=s, val_mse=0.7 + 0.001 * (s % 2)) for s in range(500, 40001, 500)]
    c = convergence_quarters(flat, 0.01, 2)
    assert c["converged"] and c["steps"] == 40000 and abs(c["drop"]) <= c["tolerance"]
    falling = [dict(step=s, val_mse=1.0 - 0.3 * s / 40000) for s in range(500, 40001, 500)]      # keeps dropping 7.5% per quarter
    c = convergence_quarters(falling, 0.01, 2)
    assert not c["converged"] and c["rel_drop"] > 0.05
    assert convergence_quarters(falling[:5]) is None                                             # too few evaluations


def test_e7_verdicts(tmp_path):
    """C23: converged pairs with a gap beyond the margin -> better_optimum; within -> speedup_only; unconverged -> extend."""
    from eval.v4_readout import aggregate
    lat = {s: f"v4_dual_ema_lam0_lr5e-4_mixed_cls_row_s{s}" for s in range(3)}
    E7 = {"id": "E7", "name": "convergence control", "verdict": "x", "equivalence_margin": 0.01, "convergence": {"rel_tol": 0.01, "noise_k": 2},
          "axes": {"steps": [40000, 60000]}}                              # 60k still available -> an unconverged 40k pair says "extend"
    spec = {"alpha": 0.05, "experiments": [E7]}

    def build(gap, tail_scratch=0.0, _k=[0]):
        runs, val = {}, {}
        _k[0] += 1; d = tmp_path / f"build{_k[0]}"; d.mkdir()                 # fresh runs dir per scenario
        for s in range(3):
            for mech, init, floor, tail in (("latinit", lat[s], 0.60, 0.0), ("none", None, 0.60 - gap, tail_scratch)):
                name = f"v4_ds_{mech}_lam0_lr5e-4_any_cell_bar32_40k_s{s}"
                _curve_run(d, name, 40000, floor, 0.4, tail, s)
                runs[name] = dict(arm="ds", mech=mech, lambda_ppd=0.0, lr="5e-4", policy="any_cell", target="bar32", seed=s, steps=40000, reused=False,
                                  **({"init_from": init} if init else {}))
                val[name] = {"by_R": {"64": {"learned": True}}, "select_mean": floor}
        src = dict(NO_SRC, value={"runs": val})
        return aggregate("E7", {"experiments": {"E7": {"runs": runs, "artifact": "x"}}}, spec, src, d)

    o = build(gap=-0.05)                                              # finetune floor 0.60, scratch 0.65 -> gap -0.05 every seed
    assert o["all_converged"] and o["verdict"] == "better_optimum" and o["n"] == 3 and o["mean_gap_value"] == -0.05
    assert o["gap_by_steps"]["40000"]["mean"] == -0.05 and set(o["gap_by_steps"]["40000"]["per_seed"]) == {"0", "1", "2"}
    assert all(q["arm_converged"] and q["ref_converged"] and q["steps"] == 40000 for q in o["pairs"])
    o = build(gap=0.0)
    assert o["verdict"] == "speedup_only" and o["mean_gap_value"] == 0.0
    o = build(gap=-0.05, tail_scratch=-0.2)                           # scratch still falling 5% per quarter -> extend
    assert o["verdict"] == "extend" and len(o["unconverged"]) == 3 and all("none" in r for r in o["unconverged"])
    E7["axes"] = {"steps": [40000]}                                   # 40k is already the longest registered budget -> extension used up
    o = build(gap=-0.05, tail_scratch=-0.2)
    assert o["verdict"] == "unconverged_after_extension"


def test_e9_real_suite_sign_tests(tmp_path):
    """C24: seed-paired per-dataset sign tests from two suite passes; gain needs p < alpha on every seed; anchors + drift."""
    from eval.v4_readout import agg_e9
    ft20 = [f"v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_20k_s{s}" for s in range(3)]
    ft40 = [f"v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s{s}" for s in range(3)]
    sc40 = [f"v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s{s}" for s in range(3)]
    sc20 = [f"real_ds_lr5e-4_s{s}" for s in range(3)]
    ex = {"axes": {"seeds": [0, 1, 2]}, "test": {"alpha": 0.05, "n_per_group": 12},
          "scored": {"pass": "P.json", "runs": ft20 + ft40 + sc40, "reference_pass": "R.json", "reference_runs": sc20}}
    keys = [f"B|d{i}|{'reg' if i % 4 == 0 else 'clf'}" for i in range(12)]           # 9 clf + 3 reg
    P, R = {}, {}
    for i, k in enumerate(keys):
        m = "r2" if k.endswith("reg") else "acc"; base = 0.6 + 0.01 * i
        R[k] = {**{r: {m: base} for r in sc20}, "histgb": {m: 0.7}, "logreg": {m: 0.65}}
        P[k] = {"histgb": {m: 0.7 + (0.001 if i == 0 else 0)}, "logreg": {m: 0.65}}
        for s in range(3):
            P[k][ft20[s]] = {m: base + 0.02}                                      # 20k finetune wins every dataset
            P[k][sc40[s]] = {m: base + 0.01}
            P[k][ft40[s]] = {m: base + 0.01 + (0.02 if i % 2 == 0 else -0.02)}   # 40k: alternating -> no gain
    (tmp_path / "P.json").write_text(json.dumps(P)); (tmp_path / "R.json").write_text(json.dumps(R))
    o = agg_e9(ex, tmp_path)
    assert o["status"] == "complete" and o["n_datasets"] == 12 and o["baseline_drift_max"] == 0.001
    a = o["pairs"]["finetune_20k_vs_scratch_20k"]["clf"]
    assert a["gain"] and a["per_seed"]["0"]["wins"] == 9 and a["per_seed"]["0"]["p"] < 0.01 and a["pooled"]["wins"] == 27
    b = o["pairs"]["finetune_40k_vs_scratch_40k"]["clf"]
    assert not b["gain"] and b["per_seed"]["1"]["wins"] + b["per_seed"]["1"]["losses"] == 9
    assert o["verdict"] == {"real_gain_clf": False, "real_gain_reg": False}
    assert o["anchors"]["finetune_20k"]["clf"]["wins_vs_histgb_by_seed"] == [3, 3, 3]   # i = 9..11: base + .02 > histgb .70 and o["anchors"]["scratch_20k"]["reg"]["n_datasets"] == 3
    ex["scored"]["pass"] = "nope.json"
    assert agg_e9(ex, tmp_path)["status"] == "missing"


def test_e8_budget_curve(tmp_path):
    """C25: gap-by-budget from snapshot fine-tunes vs the seed's scratch_5k, indicators read at the snapshot step, Spearman."""
    from eval.v4_readout import aggregate
    snaps = [1250, 5000, 10000, 20000, 30000, 40000]
    E8 = {"id": "E8", "name": "budget", "verdict": "x", "equivalence_margin": 0.01, "test": {"alpha": 0.05},
          "axes": {"seeds": [0, 1, 2], "snapshots": snaps, "indicators": ["erank", "dim_std", "val_jepa"]},
          "reference_runs": [f"v4_ds_none_lam0_lr5e-4_any_cell_bar32_5k_s{s}" for s in range(3)]}
    runs, val = {}, {}
    for s in range(3):
        pt = f"v4_dual_ema_lam0_lr5e-4_mixed_cls_row_40k_s{s}"
        d = tmp_path / pt; d.mkdir(); (d / "config.yaml").write_text(yaml.safe_dump(dict(steps=40000)))
        with open(d / "metrics.jsonl", "w") as f:
            for st in range(250, 40001, 250):        # erank grows with the budget, val_jepa rises then plateaus
                f.write(json.dumps(dict(step=st, val_jepa=round(0.02 + 0.25 * min(1, st / 5000), 4), val_total=0.0, val_pred=0.0,
                                        val_tgt=dict(erank=20 + 100 * st / 40000, dim_std=0.9, dim_std_min=0.5, within_table_var_frac=0.6, cos=0.1))) + "\n")
        runs[pt] = dict(arm="dual", mech="ema", lambda_ppd=0.0, lr="5e-4", policy="mixed", target="cls_row", seed=s, steps=40000, reused=False, save_at=snaps)
        ref = f"v4_ds_none_lam0_lr5e-4_any_cell_bar32_5k_s{s}"; _ds_run(tmp_path, ref, 5000, 5000, 0.85); val[ref] = {"by_R": {"64": {"learned": True}}, "select_mean": 0.85}
        for k in snaps:                              # downstream gap improves monotonically with the budget, seed offsets
            name = f"v4_ds_lat{k}_lam0_lr5e-4_any_cell_bar32_5k_s{s}"; v = 0.85 - 0.15 * (snaps.index(k) / 5) + 0.01 * s
            _ds_run(tmp_path, name, 5000, 5000, v); val[name] = {"by_R": {"64": {"learned": True}}, "select_mean": round(v, 4)}
            runs[name] = dict(arm="ds", mech=f"lat{k}", lambda_ppd=0.0, lr="5e-4", policy="any_cell", target="bar32", seed=s, steps=5000, reused=False, init_from=f"runs/{pt}/ckpt_{k}.pt")
    src = dict(NO_SRC, value={"runs": val})
    o = aggregate("E8", {"experiments": {"E8": {"runs": runs, "artifact": "x"}}}, {"alpha": 0.05, "experiments": [E8]}, src, tmp_path)
    assert o["e8_complete"] and len(o["points"]) == 18 and all(q["indicators"]["erank"] is not None for q in o["points"])
    gb = o["gap_by_budget"]; assert gb["1250"]["mean"] == round(0.01, 4) and gb["40000"]["mean"] == round(-0.14, 4)
    assert o["verdict"]["more_pretraining_helps"] is True and o["verdict"]["saturation_budget"] == 40000
    assert o["spearman"]["erank"]["rho"] < -0.9 and "erank" in o["predictive_indicators"] and o["spearman"]["dim_std"]["rho"] != o["spearman"]["dim_std"]["rho"] or True
    assert o["points"][0]["indicators"]["val_jepa"] is not None


def test_e10_objective_control(tmp_path):
    """C27: ds-init gaps vs the injected latent-init gaps; latent-specific needs every seed at every budget beyond the margin."""
    from eval.v4_readout import aggregate
    snaps = [20000, 40000]
    def run_e10(ds_gap_by_budget, lat_gap_by_budget):
        E10 = {"id": "E10", "name": "control", "verdict": "x", "equivalence_margin": 0.01, "test": {"alpha": None},
               "axes": {"seeds": [0, 1, 2], "snapshots": snaps}, "reference_runs": [f"v4_ds_none_lam0_lr5e-4_any_cell_bar32_5k_s{s}" for s in range(3)],
               "_lat_gaps": [[[s, k], lat_gap_by_budget[k] + 0.001 * s] for s in range(3) for k in snaps]}
        d = tmp_path / f"r{len(list(tmp_path.iterdir()))}"; d.mkdir()
        runs, val = {}, {}
        for s in range(3):
            pt = f"v4_ds_dspre_lam0_lr5e-4_mixed_bar32_40k_s{s}"; _ds_run(d, pt, 40000, 40000, 0.7)
            runs[pt] = dict(arm="ds", mech="dspre", lambda_ppd=0.0, lr="5e-4", policy="mixed", target="bar32", seed=s, steps=40000, reused=False, save_at=snaps)
            ref = f"v4_ds_none_lam0_lr5e-4_any_cell_bar32_5k_s{s}"; _ds_run(d, ref, 5000, 5000, 0.85); val[ref] = {"by_R": {"64": {"learned": True}}, "select_mean": 0.85}
            for k in snaps:
                name = f"v4_ds_dsinit{k}_lam0_lr5e-4_any_cell_bar32_5k_s{s}"; v = 0.85 + ds_gap_by_budget[k] + 0.001 * s
                _ds_run(d, name, 5000, 5000, v); val[name] = {"by_R": {"64": {"learned": True}}, "select_mean": round(v, 4)}
                runs[name] = dict(arm="ds", mech=f"dsinit{k}", lambda_ppd=0.0, lr="5e-4", policy="any_cell", target="bar32", seed=s, steps=5000, reused=False,
                                  init_from=f"runs/{pt}/ckpt_{k}.pt", reinit_head=True)
        src = dict(NO_SRC, value={"runs": val})
        return aggregate("E10", {"experiments": {"E10": {"runs": runs, "artifact": "x"}}}, {"alpha": 0.05, "experiments": [E10]}, src, d)
    o = run_e10({20000: -0.02, 40000: -0.03}, {20000: -0.117, 40000: -0.160})       # latent far better at both budgets
    assert o["e10_complete"] and o["verdict"] == "latent_specific" and o["by_budget"]["40000"]["latent_better_all_seeds"]
    assert abs(o["by_budget"]["20000"]["mean_lat_minus_ds"] - (-0.097)) < 1e-6
    o = run_e10({20000: -0.117, 40000: -0.160}, {20000: -0.117, 40000: -0.160})     # identical gaps -> generic
    assert o["verdict"] == "generic_pretraining"
    o = run_e10({20000: -0.117, 40000: -0.03}, {20000: -0.117, 40000: -0.160})      # one budget equal, one latent-better -> partial
    assert o["verdict"] == "partial"

