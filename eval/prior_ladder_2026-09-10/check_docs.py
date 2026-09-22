"""Deterministic doc check for the v3 ladder: re-derive every headline number from results_judge_v3.json and the local
metrics.jsonl files, and assert the exact strings appear in the docs (source-backed-numbers discipline). Exit 1 on any miss.
Run: uv run python eval/prior_ladder_2026-09-10/check_docs.py"""
import json, sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
JJ = json.load(open(R / "eval/prior_ladder_2026-09-10/results_judge_v3.json")); J, BASE = JJ["results"], JJ["baselines"]
DOCS = {p: (R / p).read_text() for p in ["docs/current_status.md", "docs/prior_v2_plan.md", "docs/history.md"]}
DUAL, RDS, PFN = "v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0", "real_ds_lr5e-4_s0", "PFN_tabicl2_ds_s0"
SNAP, V3, T1, SCM = "TabularJEPA_v3_tabicl2_s0_snap9250", "TabularJEPA_v3_tabicl2_s0", "TabularJEPA_v3_tabicl1_s0", "TabularJEPA_v3_scm_w64_s0"
ALL, PLAN, STAT = list(DOCS), ["docs/prior_v2_plan.md"], ["docs/current_status.md"]

def cell(rule, a, b, task, lab): return J[f"{rule}|{a}|{b}|{task}|{lab}"]
def wl(c): return f"{c['win']}:{c['loss']}"
def m3(x): return ("%.3f" % x).lstrip("0") if x >= 0 else "-" + ("%.3f" % -x).lstrip("0")
def means(c): return f"{m3(c['mean_a'])} vs {m3(c['mean_b'])}"

C = [("prior clf all", wl(cell("prior|ds", PFN, RDS, "clf", "all")), ALL), ("prior clf all means", means(cell("prior|ds", PFN, RDS, "clf", "all")), STAT + PLAN),
     ("prior 6-10", wl(cell("prior|ds", PFN, RDS, "clf", "ncls=6-10")), ALL), ("prior 6-10 means", means(cell("prior|ds", PFN, RDS, "clf", "ncls=6-10")), ALL),
     ("prior reg", wl(cell("prior|ds", PFN, RDS, "reg", "all")), STAT + PLAN),
     ("R1 clf all", wl(cell("R1", SNAP, DUAL, "clf", "all")), STAT + PLAN), ("R1 clf all means", means(cell("R1", SNAP, DUAL, "clf", "all")), STAT + PLAN),
     ("R1 6-10", wl(cell("R1", SNAP, DUAL, "clf", "ncls=6-10")), STAT + PLAN),
     ("R2 clf all", wl(cell("R2", SNAP, PFN, "clf", "all")), STAT + PLAN), ("R2 clf all means", means(cell("R2", SNAP, PFN, "clf", "all")), STAT + PLAN),
     ("R2 6-10", wl(cell("R2", SNAP, PFN, "clf", "ncls=6-10")), STAT + PLAN), ("R2 reg", wl(cell("R2", SNAP, PFN, "reg", "all")), STAT + PLAN),
     ("R3 snap-t1 clf", wl(cell("R3", SNAP, T1, "clf", "all")), STAT + PLAN), ("R3 snap-t1 3-5", wl(cell("R3", SNAP, T1, "clf", "ncls=3-5")), STAT + PLAN),
     ("R3 snap-t1 reg", wl(cell("R3", SNAP, T1, "reg", "all")), STAT + PLAN), ("R3 t1-scm clf", wl(cell("R3", T1, SCM, "clf", "all")), PLAN),
     ("R3 t1-scm reg", wl(cell("R3", T1, SCM, "reg", "all")), PLAN), ("PFN vs HistGB clf", wl(cell("ref", PFN, "histgb", "clf", "all")), STAT)]
h, pr = cell("ref", PFN, "histgb", "clf", "ncls=6-10"), cell("prior|ds", PFN, RDS, "clf", "ncls=6-10")
def m2(x): return ("%.2f" % x).lstrip("0")                        # the docs quote the HistGB gap to 2 decimals
C += [("HistGB 6-10 gap PFN", m2(h["mean_b"] - h["mean_a"]), ALL), ("HistGB 6-10 gap real_ds", m2(h["mean_b"] - pr["mean_b"]), ALL)]
# architecture ladder 2 (plan 5.2, 2026-09-11): headenc arm, L2 pairs; the dae / sigreg arms are appended here when their passes land
HE = "TabularJEPA_v3_tabicl2_headenc_s0"
if f"L2|{HE}|{PFN}|clf|all" in J:
    C += [("L2 headenc-PFN clf", wl(cell("L2", HE, PFN, "clf", "all")), STAT + PLAN), ("L2 headenc-PFN clf means", means(cell("L2", HE, PFN, "clf", "all")), STAT + PLAN),
          ("L2 headenc-PFN 6-10", wl(cell("L2", HE, PFN, "clf", "ncls=6-10")), STAT + PLAN), ("L2 headenc-PFN reg", wl(cell("L2", HE, PFN, "reg", "all")), STAT + PLAN),
          ("L2 headenc-snap clf", wl(cell("L2", HE, SNAP, "clf", "all")), STAT + PLAN), ("L2 headenc-snap reg", wl(cell("L2", HE, SNAP, "reg", "all")), STAT + PLAN),
          ("L2 headenc-dual clf", wl(cell("L2", HE, DUAL, "clf", "all")), STAT + PLAN)]
DS8K = "PFN_tabicl2_ds_s0_snap8000"   # L2t step-matched read (2026-09-13)
if f"L2t|{HE}|{DS8K}|clf|all" in J:
    C += [("L2t headenc-ds8k clf", wl(cell("L2t", HE, DS8K, "clf", "all")), STAT + PLAN), ("L2t headenc-ds8k 6-10", wl(cell("L2t", HE, DS8K, "clf", "ncls=6-10")), STAT + PLAN),
          ("L2t headenc-ds8k reg", wl(cell("L2t", HE, DS8K, "reg", "all")), STAT + PLAN), ("L2t dae-ds8k clf", wl(cell("L2t", "TabularJEPA_v3_tabicl2_dae_headenc_s0", DS8K, "clf", "all")), STAT + PLAN),
          ("L2t ds20k-ds8k clf", wl(cell("L2t", PFN, DS8K, "clf", "all")), STAT + PLAN)]
# headenc_conv snapshots vs the ds_conv snapshot at the same step (the plan-5.3 pair itself): cited in the status trajectory line
for k in sorted(k for k in J if k.startswith("L3|TabularJEPA_v3_tabicl2_headenc_conv_s0") and ("|PFN_tabicl2_ds_conv_s0_snap" in k or k.startswith("L3|TabularJEPA_v3_tabicl2_headenc_conv_s0|PFN_tabicl2_ds_conv_s0|")) and k.endswith("|clf|all")):   # same-step pairs + final vs final
    a, b = k.split("|")[1:3]; where = STAT + PLAN if "_snap" not in a else STAT   # the final-vs-final pair = the plan-5.3 verdict
    C += [(f"L3 {a}-{b} clf", wl(cell("L3", a, b, "clf", "all")), where), (f"L3 {a}-{b} clf means", means(cell("L3", a, b, "clf", "all")), where), (f"L3 {a}-{b} reg", wl(cell("L3", a, b, "reg", "all")), where)]
    if "_snap" not in a:
        C += [(f"L3 {a}-{b} 6-10", wl(cell("L3", a, b, "clf", "ncls=6-10")), where), (f"L3 {a}-{b} reg means", means(cell("L3", a, b, "reg", "all")), where)]
for run in ("PFN_tabicl2_ds_conv_s0", "TabularJEPA_v3_tabicl2_headenc_conv_s0"):   # final ckpts vs HistGB / linear baselines (judge baselines table)
    if run in BASE:
        b = BASE[run]; C += [(f"{run} vs HistGB clf", f"{b['clf']['vs_histgb'][0]}:{b['clf']['vs_histgb'][1]}", STAT), (f"{run} vs HistGB reg", f"{b['reg']['vs_histgb'][0]}:{b['reg']['vs_histgb'][1]}", STAT),
                             (f"{run} vs logreg", f"{b['clf']['vs_logreg'][0]}:{b['clf']['vs_logreg'][1]}", STAT), (f"{run} vs ridge", f"{b['reg']['vs_ridge'][0]}:{b['reg']['vs_ridge'][1]}", STAT)]
CSUM = R / "eval/open_horizon_2026-09-14/conv_summary.json"   # open-horizon training summaries (summarize_conv.py)
if CSUM.exists():
    for run, s in json.load(open(CSUM)).items():
        if s["stop"]:
            C += [(f"{run} stop step", f"{s['stop']['step']:,}", STAT), (f"{run} rule best", ("%.4f" % s["stop"]["best"]).lstrip("0"), STAT), (f"{run} final val_mse", ("%.4f" % s["final_val_mse"]).lstrip("0"), STAT)]
# L3 open-horizon pair (2026-09-14): every scored ds_conv snapshot vs the 20k baseline is cited in the status trajectory line;
# the first (20k) row = same recipe re-run = run-to-run noise floor, also cited in the plan (5.3).
for k in sorted(k for k in J if k.startswith("L3|PFN_tabicl2_ds_conv_s0") and k.endswith(f"|{PFN}|clf|all")):   # snapshots + final
    snap = k.split("|")[1]; where = STAT + PLAN if snap.endswith("_snap20000") else STAT
    C += [(f"L3 {snap}-ds20k clf", wl(cell("L3", snap, PFN, "clf", "all")), where), (f"L3 {snap}-ds20k clf means", means(cell("L3", snap, PFN, "clf", "all")), where),
          (f"L3 {snap}-ds20k reg", wl(cell("L3", snap, PFN, "reg", "all")), where)]
DA = "TabularJEPA_v3_tabicl2_dae_headenc_s0"
if f"L2|{DA}|{PFN}|clf|all" in J:
    C += [("L2 dae-PFN clf", wl(cell("L2", DA, PFN, "clf", "all")), STAT + PLAN), ("L2 dae-PFN clf means", means(cell("L2", DA, PFN, "clf", "all")), STAT + PLAN),
          ("L2 dae-PFN 6-10", wl(cell("L2", DA, PFN, "clf", "ncls=6-10")), STAT + PLAN), ("L2 dae-PFN reg", wl(cell("L2", DA, PFN, "reg", "all")), STAT + PLAN),
          ("L2x headenc-dae clf", wl(cell("L2x", HE, DA, "clf", "all")), STAT + PLAN), ("L2x headenc-dae reg", wl(cell("L2x", HE, DA, "reg", "all")), STAT + PLAN)]
SG = "TabularJEPA_v3_tabicl2_sigreg_headenc_s0"
if f"L2|{SG}|{PFN}|clf|all" in J:
    C += [("L2 sigreg-PFN clf", wl(cell("L2", SG, PFN, "clf", "all")), STAT + PLAN), ("L2 sigreg-PFN reg", wl(cell("L2", SG, PFN, "reg", "all")), STAT + PLAN),
          ("L2x headenc-sigreg clf", wl(cell("L2x", HE, SG, "clf", "all")), STAT + PLAN)]
TCP = R / "eval/arch_ladder_2026-09-11/target_content.json"
if TCP.exists():                                        # target -> value decodability (plan 5.2 dae judge)
    TC = json.load(open(TCP))["runs"]
    for run, key, lab in ((HE, "target", "headenc target"), (DA, "target", "dae target"), (DA, "target_noise_free", "dae noise-free"), (SG, "target", "sigreg target")):
        if run in TC:
            C += [(f"R2 z<-{lab}", m3(TC[run][key]["z_from_target"]), STAT + PLAN)]

def vals(run):
    return [(d["step"], d["val_mse"]) for d in map(json.loads, open(R / f"runs/{run}/metrics.jsonl")) if "val_mse" in d]
if (R / f"runs/{V3}/metrics.jsonl").exists():          # runs/ is gitignored: metrics checks only where the fetched copies exist
    v = vals(V3); mn = min(v, key=lambda x: x[1])
    C += [("V3 min val_mse", m3(round(mn[1], 3)), STAT), ("V3 20k val_mse", m3(round(v[-1][1], 3)), STAT),   # curve details live in current_status only
          ("PFN 20k val_mse", m3(round(vals(PFN)[-1][1], 3)), STAT),
          ("tabicl1 const", "%.4f" % min(x[1] for x in vals(T1)), STAT + PLAN),
          ("scm 2k", m3(round(dict(vals(SCM))[2000], 3)), STAT + PLAN), ("scm 20k", m3(round(dict(vals(SCM))[20000], 3)), STAT + PLAN)]
if (R / f"runs/{HE}/metrics.jsonl").exists():
    C += [("headenc 20k val_mse", "%.4f" % vals(HE)[-1][1], STAT + PLAN)]
if (R / f"runs/{SG}/metrics.jsonl").exists():
    C += [("sigreg 20k val_mse", "%.4f" % vals(SG)[-1][1], STAT + PLAN)]
bad = [(lab, exp, d) for lab, exp, where in C for d in where if exp not in DOCS[d]]
for lab, exp, d in bad:
    print("MISSING", lab, repr(exp), "in", d)
print(f"{len(C)} claims checked across docs, missing {len(bad)}")
sys.exit(1 if bad else 0)
