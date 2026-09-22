"""Reviewer round 1, R1 W1: the suite concatenates OpenML-CC18, Grinsztajn and TabArena, and some dataset names appear
under more than one benchmark (different OpenML ids, i.e. different curations of one source; ids from
results_suite_ids.json, suite_ids.py). The pooled sign test counts such entries twice. This re-runs every stratum of the final pair on a
pool that keeps ONE entry per name, under two rules (keep the earliest benchmark's entry / keep the latest), and writes
results_dedup.json next to this file. Numbers: same suite JSONs and the same exact two-sided sign test as judge_v3.py.
Run: python3 eval/dedup_2026-09-20/dedup_names.py"""
import json, statistics as st
from math import comb
from pathlib import Path

R = Path(__file__).resolve().parents[2]; D = Path(__file__).parent
DS, HE = "PFN_tabicl2_ds_conv_s0", "TabularJEPA_v3_tabicl2_headenc_conv_s0"
ORDER = ["CC18", "Grinsztajn", "TabArena"]                       # curation order, oldest first
S = {r: json.loads((R / f"eval/results_suite_{r}.json").read_text()) for r in (DS, HE)}
census = {r["key"]: r for r in json.loads((R / "eval/results_class_count_census.json").read_text())["datasets"]}
IDS = json.loads((D / "results_suite_ids.json").read_text())["ids"]
STRATA = [("clf", "all", lambda c: True), ("clf", "ncls=2", lambda c: c == 2), ("clf", "ncls=3-5", lambda c: 3 <= c <= 5),
          ("clf", "ncls=6-10", lambda c: 6 <= c <= 10), ("reg", "all", lambda c: True)]


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return None if n == 0 else round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n), 5)


def compare(keys, task):
    m = "acc" if task == "clf" else "r2"
    xa = [S[HE][k][HE][m] for k in keys]; xb = [S[DS][k][DS][m] for k in keys]
    w = sum(x > y for x, y in zip(xa, xb)); l = sum(x < y for x, y in zip(xa, xb))
    return dict(n=len(keys), win=w, loss=l, tie=len(keys) - w - l, p=sign_p(w, l), mean_a=round(st.mean(xa), 4), mean_b=round(st.mean(xb), 4))


def main():
    keys = sorted(S[DS]); assert set(keys) == set(S[HE])
    by = {}
    for k in keys:
        b, name, task = k.split("|"); by.setdefault((name, task), []).append(k)
    dups = {nt: sorted(ks, key=lambda k: ORDER.index(k.split("|")[0])) for nt, ks in by.items() if len(ks) > 1}
    out = {"_note": "one entry per dataset name; 'earliest' keeps the first benchmark in " + ">".join(ORDER) + ", 'latest' the last",
           "duplicates": {f"{n}|{t}": dict(entries=ks, ids=[IDS[k] for k in ks]) for (n, t), ks in sorted(dups.items())},
           "n_dup_names": {t: sum(1 for (n, tt) in dups if tt == t) for t in ("clf", "reg")},
           "n_dropped": {t: sum(len(ks) - 1 for (n, tt), ks in dups.items() if tt == t) for t in ("clf", "reg")}}
    ids = [v["ids"] for v in out["duplicates"].values()]
    out["dup_ids_all_distinct"] = all(len(set(i)) == len(i) for i in ids)
    for rule, pick in (("earliest", lambda ks: ks[0]), ("latest", lambda ks: ks[-1])):
        keep = set(keys) - {k for ks in dups.values() for k in ks if k != pick(ks)}
        rows = {}
        for task, lab, f in STRATA:
            ks = [k for k in keys if k in keep and k.endswith("|" + task) and (task == "reg" or f(census[k]["n_classes"]))]
            rows[f"{task}|{lab}"] = compare(ks, task)
        out[rule] = rows
    (D / "results_dedup.json").write_text(json.dumps(out, indent=1) + "\n")
    for rule in ("earliest", "latest"):
        for k, r in out[rule].items():
            print(f"{rule:8s} {k:14s} n={r['n']:3d} {r['win']:3d}:{r['loss']:<3d} tie={r['tie']:2d} p={r['p']}  {r['mean_a']:.3f} vs {r['mean_b']:.3f}")
    print("dup names", out["n_dup_names"], "dropped", out["n_dropped"], "dup ids distinct", out["dup_ids_all_distinct"])


if __name__ == "__main__":
    main()
