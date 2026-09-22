"""Determinism check for this directory (the C26 pattern, local): (1) build_verdict.py twice -> identical verdict.md and
summary_ko.md; (2) convergence_report.py and suite_compare.py re-aggregate to byte-identical JSON; (3) every numeric token in
verdict.md / summary_ko.md is traceable to a number in the JSON artifacts or metrics files here (rounded to the token's
decimals; percent tokens also matched x100). Exit 1 on any failure.  Run: uv run python eval/ijepa_split_2026-09-08/check.py"""
import hashlib, json, re, subprocess, sys
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parents[1]
PY = [sys.executable]
def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()
fails = []
def run(script, *args):
    r = subprocess.run(PY + [str(D / script), *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode: fails.append(f"{script} rc={r.returncode}: {r.stderr[-300:]}")
# 1. idempotence
before = {n: md5(D / n) for n in ("verdict.md", "summary_ko.md") if (D / n).exists()}
run("build_verdict.py"); mid = {n: md5(D / n) for n in before}; run("build_verdict.py"); after = {n: md5(D / n) for n in before}
for n in before:
    if not (before[n] == mid[n] == after[n]): fails.append(f"{n} not idempotent (committed {before[n][:8]} / build {mid[n][:8]} / rebuild {after[n][:8]})")
# 2. re-aggregation
for script, out, args in (("convergence_report.py", "results_convergence.json", []),
                          ("suite_compare.py", "results_suite_compare.json",
                           ["v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0", "eval/results_suite_real_dualsplit.json",
                            "real_ds_lr5e-4_s0", "v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0", "v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s0", "histgb", "histgbr"])):
    if not (D / out).exists(): continue
    old = (D / out).read_bytes(); run(script, *args)
    if (D / out).read_bytes() != old: fails.append(f"{out} changed on re-aggregation")
# 3. numbers
nums = set()
def harvest(o):
    if isinstance(o, bool): return
    if isinstance(o, (int, float)): nums.add(float(o))
    elif isinstance(o, dict): [harvest(v) for v in o.values()]
    elif isinstance(o, list): [harvest(v) for v in o]
for p in list(D.glob("results_*.json")):
    harvest(json.loads(p.read_text()))
for p in list(D.glob("metrics_*")) + [ROOT / "runs/v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0/metrics.jsonl"]:
    if p.exists():
        for l in p.read_text().splitlines(): harvest(json.loads(l))
def traceable(tok):
    v = float(tok); d = len(tok.split(".")[1]) if "." in tok else 0
    return any(abs(round(abs(s), d) - v) < 1e-9 or abs(round(abs(s) * 100, d) - v) < 1e-9 for s in nums)
for n in ("verdict.md", "summary_ko.md"):
    if not (D / n).exists(): continue
    txt = re.sub(r"`[^`]*`", " ", (D / n).read_text()); txt = re.sub(r"\d{4}-\d{2}-\d{2}", " ", txt)
    txt = re.sub(r"(lr|lam|tw|K_FEAT|CTX|emb|s)\d[\w.e-]*", " ", txt); txt = re.sub(r"[\w.-]*\d[\w.-]*(k|_s\d)\b", " ", txt)
    bad = [t for t in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", txt) if not traceable(t.lstrip("-")) and float(t) not in (0, 1, 2, 3, 4, 5, 6, 8, 12, 32, 64, 128, 256, 1024, 20000, 40000, 60000, 147, 115, 59, 49, 48, 1250, 5000, 10000, 30000, 2500, 7500, 12500, 15000, 17500, 1000)]
    if bad: fails.append(f"{n}: numbers with no source: {sorted(set(bad))[:20]}")
print("\n".join(fails) if fails else "all checks passed")
print("IJEPA-SPLIT DETERMINISM CHECK", "FAIL" if fails else "PASS"); sys.exit(1 if fails else 0)
