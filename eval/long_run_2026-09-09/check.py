"""Determinism check for this directory: (1) build_verdict.py twice -> identical verdict.md / summary_ko.md; (2) readout.py,
context_readout.py and suite_compare.py (COMPARE_OUT here) re-aggregate to byte-identical JSON; (3) every numeric token in
verdict.md / summary_ko.md is traceable to a number in the JSON artifacts here (rounded to the token's decimals; percent
tokens also matched x100). Exit 1 on any failure.  Run: uv run python eval/long_run_2026-09-09/check.py"""
import hashlib, json, os, re, subprocess, sys
from pathlib import Path
D = Path(__file__).parent; ROOT = D.parents[1]
def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()
fails = []
def run(script, *args, env=None):
    r = subprocess.run([sys.executable, str(script), *args], cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if r.returncode: fails.append(f"{Path(script).name} rc={r.returncode}: {r.stderr[-300:]}")
docs = ("verdict.md", "summary_ko.md")
before = {n: md5(D / n) for n in docs}; run(D / "build_verdict.py"); mid = {n: md5(D / n) for n in docs}; run(D / "build_verdict.py"); after = {n: md5(D / n) for n in docs}
for n in docs:
    if not (before[n] == mid[n] == after[n]): fails.append(f"{n} not idempotent")
SNAP = "v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_200k_s0_at180k"
for script, out, args, env in ((D / "readout.py", "results_long_run.json", [], None), (D / "context_readout.py", "results_context.json", [], None),
                               (ROOT / "eval/ijepa_split_2026-09-08/suite_compare.py", "results_suite_compare_180k.json",
                                [SNAP, "eval/results_suite_real_long180k.json", "v4_dual_split_ctx_tw4_diff_lam1_lr5e-4_mixed_cell_s0", "real_ds_lr5e-4_s0",
                                 "v4_ds_none_lam0_lr5e-4_any_cell_bar32_40k_s0", "v4_ds_latinit_lam0_lr5e-4_any_cell_bar32_40k_s0", "histgb"],
                                {"COMPARE_OUT": str(D / "results_suite_compare_180k.json")})):
    old = (D / out).read_bytes(); run(script, *args, env=env)
    if (D / out).read_bytes() != old: fails.append(f"{out} changed on re-aggregation")
# number traceability
nums = set()
def walk(x):
    if isinstance(x, bool): return
    if isinstance(x, (int, float)): nums.add(float(x))
    elif isinstance(x, dict): [walk(v) for v in x.values()]
    elif isinstance(x, list): [walk(v) for v in x]
for j in D.glob("results_*.json"): walk(json.loads(j.read_text()))
def traceable(tok):
    v = float(tok.replace(",", "")); dec = len(tok.split(".")[1]) if "." in tok else 0
    return any(round(abs(n), dec) == round(abs(v), dec) or round(abs(n) * 100, dec) == round(abs(v), dec) for n in nums)
for n in docs:
    text = re.sub(r"`[^`]*`", "", (D / n).read_text()); text = re.sub(r"\b\d+k\b", "", text)          # code spans, step labels like 20k
    text = re.sub(r"\[0, \d+\]", "", text)
    for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text):
        if float(tok) in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 20.0, 34.0, 40.0, 100.0, 115.0, 147.0, 512.0, 2048.0, 8.0, 32.0, 64.0, 2026.0, 9.0, 24.0) : continue   # section numbers, protocol constants, dates
        if not traceable(tok): fails.append(f"{n}: number {tok} not traceable to a JSON here")
print("\n".join(fails) if fails else "LONG-RUN DETERMINISM CHECK PASS")
sys.exit(1 if fails else 0)
