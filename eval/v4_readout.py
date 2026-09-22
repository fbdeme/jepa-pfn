"""C7: per-run standard read-out -> per-experiment aggregate (paper_v4/docs/04 section 4), driven by configs/v4_manifest.json.

Every field is derived: runs/<run>/{config.yaml,metrics.jsonl} (curve, collapse trajectory = eval/collapse.py, C1/C12 keys),
eval/results_truth_eval_runs.json (edge AUC / fam_acc / f-MSE on the run's own prior, C3), eval/results_real_readout.json + results_real_readout_v4*.json
(fixed-R value MSE vs the constant map -> learned), eval/results_probe_masked_all.json["post"] (masked f-MSE, both policies),
eval/results_jepa_target_audit.json (compact target audit). A missing source is null, never a guess.

Tests (experiments.yaml): E1 Fisher lambda 0 vs 0.1 pooled over lr + best_cell (best_cell_rule); E1b Fisher sigreg vs ema at
lr 5e-4 pooled over lambda; E3 Fisher training-failure ds vs dual pooled over seeds and lr (strata are NOT pooled into n:
five strata of one run are one observation) + parity on the value gap; E2 / E4 / E5 descriptive.

The pre-registered artifact (experiments.yaml `artifact`) is written ONLY when every run of the experiment is complete;
otherwise the aggregate goes to runs/v4_readout_partial_<E>.json (gitignored), so the status ledger cannot flip early.

Run: uv run python -m eval.v4_readout E1 [E1b E3 ...]
     uv run python -m eval.v4_readout --runs a,b --out FILE      ad-hoc read-out of named runs (C7 end-to-end check)
"""
import json
import sys
from pathlib import Path
from statistics import mean

import yaml
from scipy.stats import fisher_exact

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.collapse import trajectory                            # noqa: E402
from eval.extract_real_curves import convergence                # noqa: E402
from eval.real_readout import SELECT_ROWS                       # noqa: E402
from eval.rows_control import _curve_summary, curve             # noqa: E402

RUNS = ROOT / "runs"
MANIFEST = ROOT / "configs/v4_manifest.json"
SPEC = ROOT / "paper_v4/experiments.yaml"
SOURCES = dict(truth="eval/results_truth_eval_runs.json", value="eval/results_real_readout.json",
               masked="eval/results_probe_masked_all.json", audit="eval/results_jepa_target_audit.json",
               reval="eval/results_revalidate_vb16.json")   # validate() re-run at val_batches 16 (= v4 logs) on ckpts whose logs predate C12


def jload(rel):
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else None


def load_sources():
    src = {k: jload(v) for k, v in SOURCES.items()}
    # the v4 value read-outs live in results_real_readout_v4*.json (READOUT_OUT), merged by run over the POST file
    runs = dict((src["value"] or {}).get("runs") or {})
    origin = {r: Path(SOURCES["value"]).name for r in runs}          # which file each run's value read-out came from (C22 docs)
    for p in sorted(ROOT.glob("eval/results_real_readout_v4*.json")):
        more = json.loads(p.read_text()).get("runs") or {}
        runs.update(more); origin.update({r: p.name for r in more})
    src["value"] = {"runs": runs, "origin": origin} if runs else None
    return src


def read_run(run, axes=None, src=None, runs_dir=None):
    src = src if src is not None else load_sources()
    d = (runs_dir or RUNS) / run
    rec = dict(run=run, axes=axes)
    if not ((d / "config.yaml").exists() and (d / "metrics.jsonl").exists()):
        rec["status"] = "missing"
        return rec
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
    vals = [r for r in rows if "val_jepa" in r]
    last = (vals or [r for r in rows if "val_mse" in r or "val_nll" in r] or rows or [{}])[-1]
    rec.update(status="complete" if last.get("step", 0) >= cfg["steps"] else "partial", last_step=last.get("step"),
               steps=cfg["steps"], arm="dual" if vals else "ds", curve=_curve_summary(curve(run)) if runs_dir is None else None)
    dsv = [r for r in rows if "val_mse" in r or "val_nll" in r]
    if dsv and not vals:                                   # ds trainer: the shared val stream's last read (E6 pairs use these)
        rec.update(val_mse_last=dsv[-1].get("val_mse"), val_nll_last=dsv[-1].get("val_nll"))
    if vals:
        rec.update(collapse=trajectory(vals), val_jepa_last=vals[-1]["val_jepa"], val_ppd_last=vals[-1].get("val_ppd"),
                   val_total_last=vals[-1].get("val_total"), within_table_last=vals[-1]["val_tgt"].get("within_table_var_frac"))
        rv = (src.get("reval") or {}).get(run)
        if rv and rec["val_ppd_last"] is None:      # pre-C12 log: take the re-validated head CE (same val stream; val_batches as recorded)
            rec.update(val_ppd_last=rv.get("val_ppd"), val_total_last=rv.get("val_total"), val_ppd_source=f"revalidate(val_batches={rv.get('val_batches')})")
            if rec["within_table_last"] is None:
                rec["within_table_last"] = rv["val_tgt"].get("within_table_var_frac")
    t = (src["truth"] or {}).get(run)
    rec["truth"] = {"c5": t["c5"], "c4": t["c4"]} if t else None
    v = ((src["value"] or {}).get("runs") or {}).get(run)
    rec["value"] = (dict(by_R=v["by_R"], select_mean=v.get("select_mean"), source=((src["value"] or {}).get("origin") or {}).get(run),
                         learned=all(v["by_R"][str(R)]["learned"] for R in SELECT_ROWS if str(R) in v["by_R"]))
                    if v else None)
    if rec["arm"] == "dual" and (axes or {}).get("lambda_ppd", cfg.get("lambda_ppd", 0)) == 0:
        rec["value"] = None                       # C17: pure lat - the bar head was never trained; read it by probes (masked / truth)
        rec["value_note"] = "lambda_ppd 0: no trained head, head-based value read-out not applicable"
    post = (src["masked"] or {}).get("post") or {}
    a = post.get("arms", {}).get(run)
    rec["masked"] = ({p: a[p] for p in ("any_cell", "mixed") if p in a} | dict(floor=post.get("floor"), const_map_mse_f=post.get("const_map_mse_f"))
                     if a else None)
    au = ((src["audit"] or {}).get("runs") or {}).get(run)
    rec["audit"] = {k: x for k, x in au.items() if len(json.dumps(x)) < 200} if au else None
    return rec


def failed_training(rec):
    """E3's unit: a dual run fails if it collapsed or its value head did not learn; a ds run if it did not learn.
    None = the value read-out is not in yet (never counted)."""
    if rec.get("status") != "complete":
        return None
    if rec["arm"] == "dual" and rec["collapse"]["collapsed"]:
        return True
    return (not rec["value"]["learned"]) if rec.get("value") else None


def fisher(fail_a, n_a, fail_b, n_b):
    p = fisher_exact([[fail_a, n_a - fail_a], [fail_b, n_b - fail_b]])[1]
    return dict(table=[[fail_a, n_a - fail_a], [fail_b, n_b - fail_b]], p=round(float(p), 5))


def agg_e1(recs, ex, alpha):
    cells = {}
    for r in recs:
        k = f"lam{r['axes']['lambda_ppd']:g}_lr{r['axes']['lr']}"
        c = cells.setdefault(k, dict(lambda_ppd=r["axes"]["lambda_ppd"], lr=r["axes"]["lr"], runs=[], n_complete=0, n_collapsed=0, val_ppd_last=[]))
        c["runs"].append(r["run"])
        if r["status"] == "complete":
            c["n_complete"] += 1
            c["n_collapsed"] += int(r["collapse"]["collapsed"])
            if r.get("val_ppd_last") is not None:
                c["val_ppd_last"].append(r["val_ppd_last"])
    by_run = {r["run"]: r for r in recs}
    for c in cells.values():
        c["mean_val_ppd_last"] = round(mean(c["val_ppd_last"]), 4) if c["val_ppd_last"] else None
        c["all_alive"] = c["n_complete"] == len(c["runs"]) and c["n_collapsed"] == 0
        # head-free probes on the frozen target encoder (C17): masked f-MSE both policies, edge AUC, family acc
        mk = [by_run[r]["masked"] for r in c["runs"] if by_run[r].get("masked")]
        tr = [by_run[r]["truth"] for r in c["runs"] if by_run[r].get("truth")]
        c["probe"] = dict(n_masked=len(mk), n_truth=len(tr),
                          masked_f_any=round(mean(m["any_cell"]["cell"]["mse_f"] for m in mk), 4) if mk else None,
                          masked_f_mixed=round(mean(m["mixed"]["cell"]["mse_f"] for m in mk), 4) if mk else None,
                          edge_auc=round(mean(t["c5"]["edge_auc"] for t in tr), 4) if tr else None,
                          fam_acc=round(mean(t["c5"]["fam_acc"] for t in tr), 4) if tr else None)
    pool = lambda lam: [c for c in cells.values() if c["lambda_ppd"] == lam]
    g0, g1 = pool(0.0), pool(0.1)
    n0, n1 = sum(c["n_complete"] for c in g0), sum(c["n_complete"] for c in g1)
    f0, f1 = sum(c["n_collapsed"] for c in g0), sum(c["n_collapsed"] for c in g1)
    test = dict(groups="lambda 0 vs 0.1, pooled over lr", n=[n0, n1], collapsed=[f0, f1], **(fisher(f0, n0, f1, n1) if n0 and n1 else {"p": None}))
    test["head_reduces_collapse"] = bool(n0 and n1 and f0 / n0 > f1 / n1 and test["p"] < alpha)
    elig = [c for c in cells.values() if c["lambda_ppd"] > 0 and c["all_alive"] and c["mean_val_ppd_last"] is not None]
    best = min(elig, key=lambda c: (c["mean_val_ppd_last"], float(c["lr"]))) if elig else None
    # pure-lat selection (best_cell_lat_rule): lambda 0, every seed alive, lowest masked f-MSE (any_cell probe)
    elig_lat = [c for c in cells.values() if c["lambda_ppd"] == 0 and c["all_alive"] and c["probe"]["masked_f_any"] is not None
                and c["probe"]["n_masked"] == len(c["runs"])]
    best_lat = min(elig_lat, key=lambda c: (c["probe"]["masked_f_any"], float(c["lr"]))) if elig_lat else None
    floor = next((by_run[r]["masked"]["floor"] for r in by_run if by_run[r].get("masked")), None)
    return dict(cells=cells, test=test, best_cell_rule=ex.get("best_cell_rule"), best_cell_lat_rule=ex.get("best_cell_lat_rule"),
                best_cell=dict(lambda_ppd=best["lambda_ppd"], lr=best["lr"], mean_val_ppd_last=best["mean_val_ppd_last"]) if best else None,
                best_cell_lat=dict(lambda_ppd=best_lat["lambda_ppd"], lr=best_lat["lr"], masked_f_any=best_lat["probe"]["masked_f_any"]) if best_lat else None,
                probe_floor=floor)


def agg_e1b(recs, ema_recs, alpha):
    sig = [r for r in recs if r["status"] == "complete"]
    ema = [r for r in ema_recs if r["status"] == "complete" and r["axes"]["lr"] == "5e-4" and r["axes"]["lambda_ppd"] in (0.0, 0.1)]
    fs, fe = sum(r["collapse"]["collapsed"] for r in sig), sum(r["collapse"]["collapsed"] for r in ema)
    test = dict(groups="sigreg vs ema at lr 5e-4, pooled over lambda", n=[len(sig), len(ema)], collapsed=[fs, fe],
                **(fisher(fs, len(sig), fe, len(ema)) if sig and ema else {"p": None}))
    test["mechanism_differs"] = bool(sig and ema and test["p"] < alpha)
    return dict(test=test, sigreg_runs=[r["run"] for r in sig], ema_runs=[r["run"] for r in ema])


def agg_e3(recs, dual_recs, ex, alpha):
    """ds (E3) vs its dual counterpart (E1 lambda 0.1 cells), per lr and pooled; parity on the fixed-R value MSE."""
    by_lr, pooled = {}, dict(ds=[0, 0], dual=[0, 0])
    gaps = []
    for lr in sorted({r["axes"]["lr"] for r in recs}, key=float):
        ds = [r for r in recs if r["axes"]["lr"] == lr]
        du = [r for r in dual_recs if r["axes"]["lr"] == lr and r["axes"]["lambda_ppd"] == 0.1]
        f = {}
        for name, grp in (("ds", ds), ("dual", du)):
            fl = [failed_training(r) for r in grp]
            f[name] = dict(n=sum(x is not None for x in fl), failed=sum(bool(x) for x in fl), runs=[r["run"] for r in grp])
            pooled[name][0] += f[name]["failed"]; pooled[name][1] += f[name]["n"]
        vd = [r["value"]["select_mean"] for r in ds if r.get("value") and r["value"].get("select_mean") is not None]
        vu = [r["value"]["select_mean"] for r in du if r.get("value") and r["value"].get("select_mean") is not None]
        gap = round(mean(vd) - mean(vu), 4) if vd and vu else None
        if gap is not None:
            gaps.append(gap)
        by_lr[lr] = dict(**f, value_gap_ds_minus_dual=gap)
    (fd, nd), (fu, nu) = pooled["ds"], pooled["dual"]
    test = dict(groups="training failure ds vs dual, pooled over seeds and lr (strata not pooled: one run = one observation)",
                n=[nd, nu], failed=[fd, fu], **(fisher(fd, nd, fu, nu) if nd and nu else {"p": None}))
    test["latent_buys_stability"] = bool(nd and nu and fd / nd > fu / nu and test["p"] < alpha)
    margin = ex.get("equivalence_margin")
    parity = dict(margin=margin, mean_abs_gap=round(mean(abs(g) for g in gaps), 4) if gaps else None)
    parity["parity"] = bool(gaps and parity["mean_abs_gap"] < margin) if margin is not None else None
    return dict(by_lr=by_lr, test=test, parity=parity)


def agg_e6(recs, ex):
    """Seed-paired gaps (lat-initialised arm minus its scratch arm) on the fixed-R value MSE (select rows mean) and on
    the shared val stream's val_mse / val_nll. A negative gap = the pretrained arm is better."""
    margin = ex.get("equivalence_margin")
    by = {}
    for r in recs:
        st = ("scratch_20k" if not r["axes"].get("init_from") and r["axes"]["steps"] == 20000 else
              "scratch_5k" if not r["axes"].get("init_from") else
              "frozen_5k" if r["axes"].get("freeze_encoder") else
              "finetune_5k" if r["axes"]["steps"] == 5000 else "finetune_20k")
        by[(st, r["axes"]["seed"])] = r
    pairs = {"frozen_5k": "scratch_5k", "finetune_5k": "scratch_5k", "finetune_20k": "scratch_20k"}
    out = {}
    for arm, ref in pairs.items():
        rows = []
        for s in sorted({k[1] for k in by}):
            a, b = by.get((arm, s)), by.get((ref, s))
            if not (a and b and a["status"] == "complete" and b["status"] == "complete"):
                continue
            va = a["value"]["select_mean"] if a.get("value") else None
            vb = b["value"]["select_mean"] if b.get("value") else None
            d = lambda k: round(a[k] - b[k], 4) if a.get(k) is not None and b.get(k) is not None else None
            rows.append(dict(seed=s, arm=a["run"], ref=b["run"], gap_value=round(va - vb, 4) if va is not None and vb is not None else None,
                             gap_val_mse=d("val_mse_last"), gap_val_nll=d("val_nll_last")))
        g = [r["gap_value"] for r in rows if r["gap_value"] is not None]
        gm = [r["gap_val_mse"] for r in rows if r["gap_val_mse"] is not None]
        out[arm] = dict(ref=ref, pairs=rows, n=len(rows),
                        mean_gap_value=round(mean(g), 4) if g else None, mean_gap_val_mse=round(mean(gm), 4) if gm else None,
                        helps=bool(g) and len(g) == len(rows) and all(x < -margin for x in g) if margin is not None else None,
                        equivalent=bool(g) and abs(mean(g)) < margin if margin is not None else None)
    return dict(pairs=out, equivalence_margin=margin,
                verdict=dict(pretraining_helps=any(v["helps"] for v in out.values() if v["helps"] is not None) if any(v["n"] for v in out.values()) else None,
                             representation_sufficient=out.get("frozen_5k", {}).get("equivalent")))


def convergence_quarters(rows, rel_tol=0.01, noise_k=2, key="val_mse"):
    """E7's pre-registered convergence rule (paper_v4/experiments.yaml E7 `convergence`): mean of `key` over the last quarter
    of the run vs the previous quarter; converged if the drop <= max(rel_tol * previous-quarter mean, noise_k * sd(last quarter)).
    None when the run has too few evaluations to form two quarters (< 8)."""
    v = [(r["step"], r[key]) for r in rows if key in r and "step" in r]
    if len(v) < 8:
        return None
    T = v[-1][0]
    q4 = [m for s, m in v if s > 0.75 * T]; q3 = [m for s, m in v if 0.5 * T < s <= 0.75 * T]
    if len(q3) < 2 or len(q4) < 2:
        return None
    m3, m4 = mean(q3), mean(q4)
    sd4 = (sum((x - m4) ** 2 for x in q4) / len(q4)) ** 0.5
    drop = m3 - m4; tol = max(rel_tol * m3, noise_k * sd4)
    return dict(steps=T, n_val=len(v), q3_mean=round(m3, 4), q4_mean=round(m4, 4), q4_sd=round(sd4, 4), drop=round(drop, 4),
                rel_drop=round(drop / m3, 4) if m3 else None, tolerance=round(tol, 4), converged=bool(drop <= tol))


def agg_e7(recs, ex, runs_dir=None, extra_finetune=()):
    """E7 (C23): seed-paired gap finetune (lat init) minus scratch at the longest steps where both arms exist, each run first
    checked against the pre-registered convergence rule. Verdict: better_optimum / speedup_only / inconclusive, or extend."""
    margin = ex.get("equivalence_margin"); cv = ex.get("convergence") or {}
    rd = runs_dir or RUNS
    conv = {}
    for r in recs:
        if r["status"] == "missing":
            continue
        rows = [json.loads(l) for l in open(rd / r["run"] / "metrics.jsonl")]
        conv[r["run"]] = convergence_quarters(rows, cv.get("rel_tol", 0.01), cv.get("noise_k", 2))
    by = {}
    for r in list(recs) + list(extra_finetune):                 # extra = E6's finetune_20k (step-matched 40k row), never paired
        arm = "finetune" if r["axes"].get("init_from") else "scratch"
        by[(arm, r["axes"]["seed"], r["axes"]["steps"])] = r
    pairs, unconverged, by_steps = [], [], {}
    for s in sorted({k[1] for k in by}):
        both = sorted({k[2] for k in by if k[1] == s and k[0] == "finetune"} & {k[2] for k in by if k[1] == s and k[0] == "scratch"})
        if not both:
            continue
        for T0 in both:                                          # gap trend over every common budget (20k lives in E6)
            a0, b0 = by[("finetune", s, T0)], by[("scratch", s, T0)]
            if a0["status"] == "complete" and b0["status"] == "complete" and a0.get("value") and b0.get("value"):
                by_steps.setdefault(str(T0), {})[str(s)] = round(a0["value"]["select_mean"] - b0["value"]["select_mean"], 4)
        T = both[-1]; a, b = by[("finetune", s, T)], by[("scratch", s, T)]
        if not (a["status"] == "complete" and b["status"] == "complete"):
            continue
        ca, cb = conv.get(a["run"]), conv.get(b["run"])
        for run, c in ((a["run"], ca), (b["run"], cb)):
            if not (c and c["converged"]):
                unconverged.append(run)
        va = a["value"]["select_mean"] if a.get("value") else None
        vb = b["value"]["select_mean"] if b.get("value") else None
        pairs.append(dict(seed=s, steps=T, arm=a["run"], ref=b["run"],
                          gap_value=round(va - vb, 4) if va is not None and vb is not None else None,
                          gap_val_mse_q4=round(ca["q4_mean"] - cb["q4_mean"], 4) if ca and cb else None,
                          arm_converged=bool(ca and ca["converged"]), ref_converged=bool(cb and cb["converged"])))
    g = [q["gap_value"] for q in pairs if q["gap_value"] is not None]
    all_conv = bool(pairs) and not unconverged and len(g) == len(pairs)
    reg_steps = (ex.get("axes") or {}).get("steps") or []
    max_steps = max(int(x) for x in reg_steps) if reg_steps else None
    if not pairs or len(g) != len(pairs):
        verdict = None
    elif unconverged and max_steps is not None and all(q["steps"] >= max_steps for q in pairs):
        verdict = "unconverged_after_extension"           # the one pre-registered extension is used up: report the gap trend, no convergence claim
    elif unconverged:
        verdict = "extend"
    elif margin is not None and all(x < -margin for x in g):
        verdict = "better_optimum"
    elif margin is not None and abs(mean(g)) < margin:
        verdict = "speedup_only"
    else:
        verdict = "inconclusive"
    e6p = ROOT / "eval/results_v4_e6_pretrain_finetune.json"
    ref20 = None
    if e6p.exists() and runs_dir is None:
        e6 = json.loads(e6p.read_text()); ref20 = (e6.get("pairs") or {}).get("finetune_20k", {}).get("mean_gap_value")
    gap_by_steps = {T0: dict(per_seed=v, mean=round(mean(v.values()), 4)) for T0, v in sorted(by_steps.items(), key=lambda kv: int(kv[0]))}
    # step-matched secondary read: finetune(T - pretrain_steps) vs scratch(T); the fine-tune's pretraining budget is the init run's steps
    step_matched = {}
    pre_steps = {}
    for r in list(recs) + list(extra_finetune):
        src_run = (r["axes"].get("init_from") or "").split("/")[1] if "/" in (r["axes"].get("init_from") or "") else r["axes"].get("init_from")
        if src_run and src_run not in pre_steps:
            cfgp = rd / src_run / "config.yaml"
            pre_steps[src_run] = yaml.safe_load(cfgp.read_text()).get("steps") if cfgp.exists() else None
    for (arm, s, T), r in by.items():
        if arm != "finetune" or r["status"] != "complete" or not r.get("value"):
            continue
        P = pre_steps.get((r["axes"].get("init_from") or "").split("/")[1] if "/" in (r["axes"].get("init_from") or "") else r["axes"].get("init_from"))
        ref = by.get(("scratch", s, T + P)) if P else None
        if ref and ref["status"] == "complete" and ref.get("value"):
            step_matched.setdefault(str(T + P), {})[str(s)] = round(r["value"]["select_mean"] - ref["value"]["select_mean"], 4)
    step_matched = {T0: dict(per_seed=v, mean=round(mean(v.values()), 4), all_negative=all(x < 0 for x in v.values()),
                             all_beyond_margin=all(x < -margin for x in v.values()) if margin is not None else None)
                    for T0, v in sorted(step_matched.items(), key=lambda kv: int(kv[0]))}
    if ref20 is not None:
        gap_by_steps = {"20000": dict(per_seed=None, mean=ref20, source="E6 finetune_20k vs scratch_20k"), **gap_by_steps}
    return dict(convergence=conv, convergence_rule=cv, pairs=pairs, n=len(pairs), equivalence_margin=margin,
                mean_gap_value=round(mean(g), 4) if g else None, all_converged=all_conv, unconverged=unconverged,
                gap_at_20k_e6=ref20, gap_by_steps=gap_by_steps, step_matched=step_matched, pretrain_steps=pre_steps, verdict=verdict)


def agg_e8(recs, ex, src, runs_dir=None):
    """E8 (C25): for every (seed, snapshot) the 5k fine-tune's fixed-R value minus the seed's E6 scratch_5k value, the
    pretraining indicators logged at that snapshot step, the gap-by-budget curve, saturation budget, Spearman per indicator."""
    from scipy.stats import spearmanr
    from eval.probe_real_prior import holm
    margin = ex.get("equivalence_margin"); alpha = ex["test"]["alpha"]; inds = ex["axes"]["indicators"]; snaps = ex["axes"]["snapshots"]
    rd = runs_dir or RUNS
    pt = {r["axes"]["seed"]: r for r in recs if r["arm"] == "dual" or r["axes"].get("arm") == "dual"}
    refs = {}
    for r in ex.get("reference_runs", []):
        rec = read_run(r, None, src, runs_dir); refs[int(r.rsplit("_s", 1)[1])] = rec
    ind_at = {}
    for s, r in pt.items():
        rows = [json.loads(l) for l in open(rd / r["run"] / "metrics.jsonl")] if r["status"] != "missing" else []
        ind_at[s] = {row["step"]: {**{k: row["val_tgt"].get(k) for k in inds if k != "val_jepa"}, "val_jepa": row.get("val_jepa")}
                     for row in rows if "val_jepa" in row and row["step"] in snaps}
    points = []
    for r in recs:
        if r["arm"] != "ds" or not str(r["axes"].get("mech", "")).startswith("lat"):
            continue
        s, k = r["axes"]["seed"], int(r["axes"]["mech"][3:]); ref = refs.get(s)
        va = r["value"]["select_mean"] if r.get("value") else None
        vb = ref["value"]["select_mean"] if ref and ref.get("value") else None
        points.append(dict(seed=s, budget=k, run=r["run"], ref=ref["run"] if ref else None, status=r["status"], value=va, ref_value=vb,
                           gap=round(va - vb, 4) if va is not None and vb is not None else None, indicators=ind_at.get(s, {}).get(k)))
    by_budget = {}
    for k in snaps:
        g = {str(q["seed"]): q["gap"] for q in points if q["budget"] == k and q["gap"] is not None}
        im = {}
        for name in inds:                                        # seed-mean of every indicator at this budget (docs print these)
            xs = [q["indicators"][name] for q in points if q["budget"] == k and q.get("indicators") and q["indicators"].get(name) is not None]
            im[name] = round(mean(xs), 4) if xs else None
        by_budget[str(k)] = dict(per_seed=g, mean=round(mean(g.values()), 4) if len(g) == len(pt) and g else None, indicator_means=im)
    means = {k: v["mean"] for k, v in by_budget.items() if v["mean"] is not None}
    sat = None
    if len(means) == len(snaps):
        mn = min(means.values()); sat = min(int(k) for k, v in means.items() if v <= mn + margin)
    ok = lambda k: all(str(s) in by_budget[str(k)]["per_seed"] for s in pt)
    helps = None
    if ok(snaps[-1]) and ok(5000) and pt:
        helps = all(by_budget[str(snaps[-1])]["per_seed"][str(s)] < by_budget["5000"]["per_seed"][str(s)] - margin for s in pt)
    sp = {}
    for name in inds:
        xs = [(q["indicators"][name], q["gap"]) for q in points if q.get("indicators") and q["indicators"].get(name) is not None and q["gap"] is not None]
        if len(xs) >= 4:
            rho, pv = spearmanr([x for x, _ in xs], [y for _, y in xs]); sp[name] = dict(rho=round(float(rho), 4), p=round(float(pv), 5), n=len(xs))
    rej = holm([sp[n]["p"] for n in sp], alpha) if sp else []
    names = list(sp); predictive = [names[i] for i in rej]
    complete = bool(points) and len(points) == len(snaps) * len(pt) and all(q["gap"] is not None for q in points) and all(r["status"] == "complete" for r in recs)
    return dict(points=points, gap_by_budget=by_budget, saturation_budget=sat, spearman=sp, predictive_indicators=predictive, equivalence_margin=margin,
                verdict=dict(more_pretraining_helps=helps, saturation_budget=sat, predictive_indicators=predictive), e8_complete=complete)


def agg_e10(recs, ex, src, runs_dir=None):
    """E10 (C27): ds-pretrained (mixed masking, head re-initialised) fine-tunes vs the E8 latent-init fine-tunes, both against
    the seed's E6 scratch_5k: is the initialisation gain latent-specific?"""
    margin = ex.get("equivalence_margin"); snaps = ex["axes"]["snapshots"]
    refs = {int(r.rsplit("_s", 1)[1]): read_run(r, None, src, runs_dir) for r in ex.get("reference_runs", [])}
    e8p = ROOT / "eval/results_v4_e8_pretrain_budget.json"
    lat = {}
    if e8p.exists() and runs_dir is None:
        for q in json.loads(e8p.read_text())["points"]:
            lat[(q["seed"], q["budget"])] = q["gap"]
    if runs_dir is not None and ex.get("_lat_gaps"):               # tests inject the latent gaps
        lat = {tuple(k): v for k, v in ex["_lat_gaps"]}
    points = []
    for r in recs:
        if r["arm"] != "ds" or not str(r["axes"].get("mech", "")).startswith("dsinit"):
            continue
        s, k = r["axes"]["seed"], int(r["axes"]["mech"][6:]); ref = refs.get(s)
        va = r["value"]["select_mean"] if r.get("value") else None
        vb = ref["value"]["select_mean"] if ref and ref.get("value") else None
        g_ds = round(va - vb, 4) if va is not None and vb is not None else None
        g_lat = lat.get((s, k))
        points.append(dict(seed=s, budget=k, run=r["run"], status=r["status"], value=va, scratch_value=vb, gap_ds_init=g_ds, gap_lat_init=g_lat,
                           lat_minus_ds=round(g_lat - g_ds, 4) if g_lat is not None and g_ds is not None else None))
    by_budget = {}
    for k in snaps:
        d = {str(q["seed"]): q["lat_minus_ds"] for q in points if q["budget"] == k and q["lat_minus_ds"] is not None}
        gd = {str(q["seed"]): q["gap_ds_init"] for q in points if q["budget"] == k and q["gap_ds_init"] is not None}
        gl = {str(q["seed"]): q["gap_lat_init"] for q in points if q["budget"] == k and q["gap_lat_init"] is not None}
        by_budget[str(k)] = dict(lat_minus_ds=d, mean_lat_minus_ds=round(mean(d.values()), 4) if d else None,
                                 gap_ds_init=gd, mean_gap_ds_init=round(mean(gd.values()), 4) if gd else None,
                                 gap_lat_init=gl, mean_gap_lat_init=round(mean(gl.values()), 4) if gl else None,
                                 latent_better_all_seeds=bool(d) and all(x < -margin for x in d.values()),
                                 equivalent_all_seeds=bool(d) and all(abs(x) < margin for x in d.values()),
                                 ds_better_all_seeds=bool(d) and all(x > margin for x in d.values()))
    complete = bool(points) and len(points) == len(snaps) * len(refs) and all(q["lat_minus_ds"] is not None for q in points) and all(r["status"] == "complete" for r in recs)
    vals = list(by_budget.values())
    if not complete:
        verdict = None
    elif all(v["latent_better_all_seeds"] for v in vals):
        verdict = "latent_specific"
    elif all(v["equivalent_all_seeds"] or v["ds_better_all_seeds"] for v in vals):
        verdict = "generic_pretraining"
    else:
        verdict = "partial"
    geo = ROOT / "eval/results_encoder_geometry.json"
    geometry = json.loads(geo.read_text()) if geo.exists() and runs_dir is None else None
    return dict(points=points, by_budget=by_budget, equivalence_margin=margin, e10_complete=complete, verdict=verdict, geometry=geometry)


def agg_e9(ex, root=None):
    """E9 (C24): real-data suite scores (eval/suite_bench.py passes) -> seed-paired finetune-minus-scratch per dataset,
    exact sign test per task (clf acc / reg r2) per seed and pooled, wins vs histgb, baseline drift between passes."""
    from eval.probe_real_prior import sign_test
    root = root or ROOT; sc = ex["scored"]; alpha = ex["test"]["alpha"]
    pp, rp = root / sc["pass"], root / sc["reference_pass"]
    if not pp.exists():
        return dict(status="missing", pass_file=sc["pass"])
    P, R = json.loads(pp.read_text()), json.loads(rp.read_text())
    keys = sorted(set(P) & set(R)); kinds = {k: k.split("|")[-1] for k in keys}
    metric = lambda kind: "r2" if kind == "reg" else "acc"
    runs = list(sc["runs"]); missing = [r for r in runs if not all(r in P[k] for k in keys)]
    drift = 0.0
    for k in keys:
        for m in ("histgb", "logreg", "ridge", "histgbr"):
            if m in P[k] and m in R[k]:
                for mk in P[k][m]:
                    drift = max(drift, abs(P[k][m][mk] - R[k][m][mk]))
    seeds = ex["axes"]["seeds"]
    lat20 = {s: next(r for r in runs if "latinit" in r and "_20k_" in r and r.endswith(f"_s{s}")) for s in seeds}
    lat40 = {s: next(r for r in runs if "latinit" in r and "_40k_" in r and r.endswith(f"_s{s}")) for s in seeds}
    sc40 = {s: next(r for r in runs if "_none_" in r and "_40k_" in r and r.endswith(f"_s{s}")) for s in seeds}
    sc20 = {s: next(r for r in sc["reference_runs"] if r.endswith(f"_s{s}")) for s in seeds}
    pairs = {"finetune_20k_vs_scratch_20k": (lat20, sc20, P, R), "finetune_40k_vs_scratch_40k": (lat40, sc40, P, P)}
    out = {}
    for name, (A, B, PA, PB) in pairs.items():
        res = {}
        for kind in ("clf", "reg"):
            ks = [k for k in keys if kinds[k] == kind]; m = metric(kind)
            per_seed, pooled = {}, [0, 0, 0]
            for s in seeds:
                w = l = tie = 0; d = []
                for k in ks:
                    a, b = PA[k].get(A[s]), PB[k].get(B[s])
                    if not (a and b):
                        continue
                    diff = a[m] - b[m]; d.append(diff)
                    w += diff > 0; l += diff < 0; tie += diff == 0
                p = sign_test(l, w)
                per_seed[str(s)] = dict(wins=w, losses=l, ties=tie, p=round(p, 5), mean_delta=round(mean(d), 4) if d else None, n=len(d),
                                        gain=bool(w > l and p < alpha))
                pooled[0] += w; pooled[1] += l; pooled[2] += tie
            pp_ = sign_test(pooled[1], pooled[0])
            res[kind] = dict(n_datasets=len(ks), per_seed=per_seed, pooled=dict(wins=pooled[0], losses=pooled[1], ties=pooled[2], p=round(pp_, 6)),
                             gain=all(v["gain"] for v in per_seed.values()) if per_seed else None)
        out[name] = res
    # anchors: mean metric per arm (seed-averaged) and datasets won vs histgb, per task
    def arm_stats(runmap, PX):
        st = {}
        for kind in ("clf", "reg"):
            ks = [k for k in keys if kinds[k] == kind]; m = metric(kind); hb = "histgb" if kind == "clf" else "histgbr"
            vals, wins = [], []
            for s in seeds:
                r = runmap[s]
                v = [PX[k][r][m] for k in ks if r in PX[k]]; vals.append(mean(v) if v else None)
                wins.append(sum(PX[k][r][m] > PX[k][hb][m] for k in ks if r in PX[k] and hb in PX[k]))
            st[kind] = dict(mean_metric_by_seed=[round(x, 4) if x is not None else None for x in vals],
                            mean_metric=round(mean([x for x in vals if x is not None]), 4) if any(x is not None for x in vals) else None,
                            wins_vs_histgb_by_seed=wins, n_datasets=len(ks))
        return st
    anchors = {"finetune_20k": arm_stats(lat20, P), "scratch_20k": arm_stats(sc20, R), "finetune_40k": arm_stats(lat40, P), "scratch_40k": arm_stats(sc40, P)}
    hb = {}
    for kind in ("clf", "reg"):
        ks = [k for k in keys if kinds[k] == kind]; m = metric(kind); b = "histgb" if kind == "clf" else "histgbr"
        hv = [P[k][b][m] for k in ks if b in P[k]]; hb[kind] = round(mean(hv), 4) if hv else None
    complete = not missing and len(keys) == ex["test"]["n_per_group"]
    verdict = dict(real_gain_clf=bool(out["finetune_20k_vs_scratch_20k"]["clf"]["gain"] and out["finetune_40k_vs_scratch_40k"]["clf"]["gain"]),
                   real_gain_reg=bool(out["finetune_20k_vs_scratch_20k"]["reg"]["gain"] and out["finetune_40k_vs_scratch_40k"]["reg"]["gain"]))
    return dict(status="complete" if complete else "partial", n_datasets=len(keys), n_runs=len(runs), missing_runs=missing,
                baseline_drift_max=round(drift, 4), pairs=out, anchors=anchors, histgb_mean=hb, alpha=alpha, verdict=verdict)


def agg_descriptive(recs, eid):
    out = dict(runs={r["run"]: r for r in recs})
    if eid == "E5":
        out["extension_rule"] = {r["run"]: convergence(curve(r["run"])) for r in recs if r["status"] != "missing"}
    return out


def aggregate(eid, manifest, spec, src, runs_dir=None):
    ex = {e["id"]: e for e in spec["experiments"]}[eid]
    alpha = spec["alpha"]
    reads = lambda e: [read_run(run, ax, src, runs_dir) for run, ax in manifest["experiments"][e]["runs"].items()]
    recs = reads(eid)
    out = dict(experiment=eid, name=ex["name"], n_runs=len(recs), n_complete=sum(r["status"] == "complete" for r in recs),
               complete=all(r["status"] == "complete" for r in recs), verdict_rule=ex["verdict"], runs={r["run"]: r for r in recs})
    if eid == "E1":
        out.update(agg_e1(recs, ex, alpha))
    elif eid == "E1b":
        out.update(agg_e1b(recs, reads("E1") if "E1" in manifest["experiments"] else [], alpha))
    elif eid == "E3":
        out.update(agg_e3(recs, reads("E1") if "E1" in manifest["experiments"] else [], ex, alpha))
    elif eid == "E6":
        out.update(agg_e6(recs, ex))
    elif eid == "E7":
        e6 = manifest["experiments"].get("E6", {}).get("runs", {})
        extra = [read_run(r, ax, src, runs_dir) for r, ax in e6.items() if ax.get("init_from") and ax["steps"] == 20000 and not ax.get("freeze_encoder")]
        out.update(agg_e7(recs, ex, runs_dir, extra_finetune=extra))
    elif eid == "E8":
        out.update(agg_e8(recs, ex, src, runs_dir))
    elif eid == "E10":
        out.update(agg_e10(recs, ex, src, runs_dir))
    else:
        out.update(agg_descriptive(recs, eid))
    return out


def main():
    argv = sys.argv[1:]
    src = load_sources()
    if "--runs" in argv:
        runs = argv[argv.index("--runs") + 1].split(",")
        out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else ROOT / "runs/v4_readout_adhoc.json"
        res = {r: read_run(r, None, src) for r in runs}
        out.write_text(json.dumps(res, indent=1) + "\n")
        for r, rec in res.items():
            print(r, rec["status"], {k: rec.get(k) for k in ("collapse", "val_ppd_last", "within_table_last")}, "truth" if rec["truth"] else "-", "value" if rec["value"] else "-")
        print("wrote", out)
        return
    manifest, spec = json.loads(MANIFEST.read_text()), yaml.safe_load(SPEC.read_text())
    if argv and argv[0] == "E9":
        # E9 scores existing runs on the real-data suite: no configs, no manifest entry - aggregate the suite passes directly
        ex = {e["id"]: e for e in spec["experiments"]}["E9"]
        out = dict(experiment="E9", name=ex["name"], verdict_rule=ex["verdict"], **agg_e9(ex))
        path = ROOT / ex["artifact"] if out["status"] == "complete" else ROOT / "runs/v4_readout_partial_E9.json"
        path.write_text(json.dumps(out, indent=1) + "\n")
        print("E9:", out["status"], f"datasets {out.get('n_datasets')}", "->", path.relative_to(ROOT), json.dumps(out.get("verdict")))
        return
    if "--baseline-20k" in argv:
        # E7 rationale: the E6 20k arms (lat-init finetune vs reused real_ds scratch) under E7's convergence rule
        ex = {e["id"]: e for e in spec["experiments"]}["E7"]; cv = ex.get("convergence") or {}
        e6 = manifest["experiments"]["E6"]["runs"]
        arm = {r: ("finetune 20k" if ax.get("init_from") else "scratch 20k (real_ds)") for r, ax in e6.items() if ax["steps"] == 20000 and not ax.get("freeze_encoder")}
        conv = {r: convergence_quarters([json.loads(l) for l in open(RUNS / r / "metrics.jsonl")], cv.get("rel_tol", 0.01), cv.get("noise_k", 2)) for r in sorted(arm)}
        out = dict(rule=cv, runs=conv, arm=arm, n_converged=sum(bool(c and c["converged"]) for c in conv.values()), n=len(conv))
        path = ROOT / "eval/results_v4_e7_baseline_20k.json"; path.write_text(json.dumps(out, indent=1) + "\n")
        for r, c in conv.items():
            print(f"{r:52} {arm[r]:22} q3 {c['q3_mean']:.4f} -> q4 {c['q4_mean']:.4f} drop {c['drop']:+.4f} tol {c['tolerance']:.4f} converged={c['converged']}")
        print("wrote", path.relative_to(ROOT), f"converged {out['n_converged']}/{out['n']}")
        return
    for eid in argv or list(manifest["experiments"]):
        if eid not in manifest["experiments"]:
            print(f"{eid}: not in the manifest (pending in gen_v4_configs)"); continue
        res = aggregate(eid, manifest, spec, src)
        target = ROOT / manifest["experiments"][eid]["artifact"] if res["complete"] else ROOT / f"runs/v4_readout_partial_{eid}.json"
        target.write_text(json.dumps(res, indent=1) + "\n")
        print(f"{eid}: {res['n_complete']}/{res['n_runs']} complete -> {target.relative_to(ROOT)}", {k: res[k] for k in ("test", "best_cell", "parity") if k in res})


if __name__ == "__main__":
    main()
