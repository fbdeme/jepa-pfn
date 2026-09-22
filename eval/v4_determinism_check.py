"""Are the paper_v4 results deterministic and fully source-backed?  Exit 1 on any failure.

1. every experiment artifact re-aggregates from its inputs to the same JSON (E1 E3 E6 E7 E8 via eval.v4_readout.aggregate,
   E9 via agg_e9)                                   -> a stale artifact (code changed after it was written) is a failure
2. paper_v4/build_docs.py is idempotent: rendering again changes no byte of paper_v4/docs/*.md
3. every number printed in paper_v4/docs/03_experiments_done.md section 4 (the v4 results) exists as a numeric leaf of some
   results JSON (artifacts + the readout / census / trajectory files build_docs reads)  -> no hand-typed result numbers
4. paper_v4/check_docs.py passes (and its --selftest), scripts/status_check.py reports no drift

Run: uv run python -m eval.v4_determinism_check
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.v4_readout import MANIFEST, SPEC, agg_e9, aggregate, load_sources     # noqa: E402

SOURCES_FOR_NUMBERS = ["eval/results_v4_*.json", "eval/results_real_readout*.json", "eval/results_revalidate*.json", "eval/results_jepa_collapse_census.json",
                       "eval/results_probe_masked_all.json", "eval/results_truth_eval_runs.json", "eval/results_lat_pretrain_trajectory.json",
                       "eval/results_suite_real_e9.json", "eval/results_suite_real_ds.json", "eval/results_cls_val_baselines.json", "eval/results_suite_cls.json"]


def leaves(x, out):
    if isinstance(x, dict):
        for v in x.values():
            leaves(v, out)
    elif isinstance(x, list):
        for v in x:
            leaves(v, out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.add(x)
    return out


def norm(o):
    return json.loads(json.dumps(o, sort_keys=True))


def check_artifacts():
    spec = yaml.safe_load(SPEC.read_text()); manifest = json.loads(MANIFEST.read_text()); src = load_sources()
    rows = []
    for ex in spec["experiments"]:
        eid, art = ex["id"], ROOT / ex["artifact"]
        if not art.exists():
            rows.append((eid, "no artifact", None)); continue
        stored = norm(json.loads(art.read_text()))
        if eid == "E9":
            fresh = norm(dict(experiment="E9", name=ex["name"], verdict_rule=ex["verdict"], **agg_e9(ex)))
        elif eid in manifest["experiments"]:
            fresh = norm(aggregate(eid, manifest, spec, src))
        else:
            rows.append((eid, "not in manifest", None)); continue
        same = fresh == stored
        diff = None
        if not same:
            ka, kb = set(fresh), set(stored)
            diff = f"keys only fresh {sorted(ka - kb)[:5]} only stored {sorted(kb - ka)[:5]}" if ka != kb else \
                   "values differ at " + ", ".join(k for k in fresh if fresh[k] != stored.get(k))[:200]
        rows.append((eid, "identical" if same else "STALE", diff))
    return rows


def docs_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "paper_v4/docs").glob("*.md"))}


def check_docs_idempotent():
    """Build twice: the second build must change nothing (the first may legitimately apply a template/builder edit)."""
    subprocess.run([sys.executable, "paper_v4/build_docs.py"], cwd=ROOT, capture_output=True, text=True)
    before = docs_hashes()
    r = subprocess.run([sys.executable, "paper_v4/build_docs.py"], cwd=ROOT, capture_output=True, text=True)
    after = docs_hashes()
    changed = [k for k in after if before.get(k) != after[k]]
    return r.returncode == 0 and not changed, changed, r.stdout.strip().splitlines()[-1:] + r.stderr.strip().splitlines()[-1:]


def check_numbers():
    vals = set()
    for pat in SOURCES_FOR_NUMBERS:
        for p in ROOT.glob(pat):
            try:
                leaves(json.loads(p.read_text()), vals)
            except Exception:
                pass
    strs = set()
    for v in vals:
        strs.add(str(v)); strs.add(f"{v:g}"); strs.add(str(float(v)))    # a set keeps one of {1, 1.0}: render both
        if isinstance(v, float):
            strs.add(f"{v:.4f}".rstrip("0").rstrip("."))
    doc = (ROOT / "paper_v4/docs/03_experiments_done.md").read_text()
    sec = doc[doc.index("## 4."):]
    sec = re.sub(r"`[^`]*`", " ", sec)                               # run names, step tokens, paths
    sec = re.sub(r"\(\d{4}-\d{2}-\d{2}[^)]*\)", " ", sec)             # dates
    sec = re.sub(r"\bE\d+b?\b|\bC\d+\b|§\d+(?:\.\d+)?|\bs[0-2]\b|\bR\d+\b|\bK[_ ]?FEAT \d+|\bCTX \d+", " ", sec)
    sec = re.sub(r"\b\d(?:\.\d+)?e-\d\b", " ", sec)                 # lr labels (5e-4, 1.7e-4, 1.5e-3) are config tokens
    nums = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?(?![\w.%])", sec)
    missing = sorted({n for n in nums if n not in strs and n.lstrip("-") not in strs and not (n.lstrip("-").isdigit() and int(n) in vals)})
    return len(nums), missing


def main():
    fails = []
    print("1. artifacts re-aggregate identically:")
    for eid, st, diff in check_artifacts():
        print(f"   {eid:4} {st}" + (f"  ({diff})" if diff else ""))
        if st == "STALE":
            fails.append(f"{eid} artifact stale")
    ok, changed, tail = check_docs_idempotent()
    print(f"2. build_docs idempotent: {'yes' if ok else 'NO ' + str(changed)}  {' '.join(tail)}")
    if not ok:
        fails.append("build_docs not idempotent")
    n, missing = check_numbers()
    print(f"3. numbers in 03 section 4: {n} checked, not found in any results JSON: {missing or 'none'}")
    if missing:
        fails.append(f"hand-typed numbers {missing}")
    for cmd in (["paper_v4/check_docs.py"], ["paper_v4/check_docs.py", "--selftest"], ["scripts/status_check.py"]):
        r = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or [""])[-1][:90]
        print(f"4. {' '.join(cmd)}: rc {r.returncode}  {last}")
        if r.returncode != 0:
            fails.append(" ".join(cmd))
    print("DETERMINISM CHECK", "FAIL: " + "; ".join(fails) if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
