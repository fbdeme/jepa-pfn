"""Re-review round 2, NEW-10: the OpenML id behind every suite record. eval/suite_bench.py keys its records
bench|name|task and writes them in ascending-did order, so where a benchmark lists one name twice the record with
the higher id overwrote the other. This reproduces that enumeration (network: OpenML listings only, no data) and
writes results_suite_ids.json: the id per record key and the ids each key overwrote.
Run: uv run --with openml python eval/dedup_2026-09-20/suite_ids.py"""
import inspect, json, sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]; D = Path(__file__).parent
sys.path.insert(0, str(R))
from eval.suite_bench import SUITES, pick_full  # noqa: E402
import openml  # noqa: E402

S = json.loads((R / "eval/results_suite_PFN_tabicl2_ds_conv_s0.json").read_text())
caps = inspect.signature(pick_full).parameters
CAP_ROWS, CAP_FEAT = caps["cap_rows"].default, caps["cap_feat"].default
S = json.loads((R / "eval/results_suite_PFN_tabicl2_ds_conv_s0.json").read_text())
ids, over = {}, {}
for bench, sids in SUITES.items():
    member = {}
    for sid in sids:
        for d in openml.study.get_suite(sid).data:
            member.setdefault(int(d), []).append(sid)
    dids = sorted(member)
    df = openml.datasets.list_datasets(data_id=dids, output_format="dataframe")
    assert list(df.did) == sorted(df.did)
    for _, r in df.iterrows():                                   # ascending did, the order pick_full keeps
        n, f, c = r.NumberOfInstances, r.NumberOfFeatures, r.NumberOfClasses
        if n > CAP_ROWS or f > CAP_FEAT or not (c == 0 or 2 <= c <= 10):
            continue
        guess = "reg" if c == 0 else "clf"                        # the eval decides from the loaded target; where the records
        have = {t for t in ("clf", "reg") if f"{bench}|{r['name']}|{t}" in S}   # hold one task for the name, that task is the eval's
        task = guess if (guess in have or len(have) != 1) else have.pop()
        key = f"{bench}|{r['name']}|{task}"
        if key in ids:
            assert member[ids[key]] != member[int(r.did)], (key, "twice in one suite")   # each pair spans two suites
            over.setdefault(key, []).append(ids[key])
        ids[key] = int(r.did)
census = {r["key"]: r["id"] for r in json.loads((R / "eval/results_class_count_census.json").read_text())["datasets"]}
out = {"_note": "id per suite record key from the eval/suite_bench.py enumeration (ascending did; a later id overwrites an "
                "earlier one of the same key); 'overwritten' lists the ids a key replaced",
       "ids": dict(sorted(ids.items())), "overwritten": dict(sorted(over.items())),
       "n_overwritten_by_bench": {b: sum(1 for k in over if k.startswith(b + "|")) for b in SUITES},
       "record_keys_missing_id": sorted(set(S) - set(ids)), "enumerated_not_in_records": sorted(set(ids) - set(S)),
       "class_census_id_mismatch": {k: [ids.get(k), v] for k, v in census.items() if ids.get(k) != v}}
(D / "results_suite_ids.json").write_text(json.dumps(out, indent=1) + "\n")
print("ids", len(ids), "overwritten", out["n_overwritten_by_bench"], "missing", out["record_keys_missing_id"],
      "extra", out["enumerated_not_in_records"], "census mismatch", out["class_census_id_mismatch"])
