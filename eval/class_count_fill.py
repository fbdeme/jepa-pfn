"""Complete eval/results_class_count_census.json for every classification dataset of the real-data suite.

The 2026-09-04 census resolved OpenML ids through the shape census, which keyed datasets by name alone, so the sets
whose name occurs under two ids (credit-g in CC18 and TabArena; electricity twice inside the Grinsztajn suites, where
the later id overwrote the earlier in the results dict) were left unresolved and fell out of every class stratum.
Here the id of each results key comes from the suite run's own log, which prints `[task] name did=ID ... logreg=X
histgb=Y` per dataset under a `=== bench ===` header: the key's baseline scores in the results file must equal that
line's (the baselines are deterministic, so any suite log will do). The class count is then read from the cached
target column (class_count_census.load_dataset_info), and every entry the old census already had is re-derived
the same way and must agree. Run: uv run --with pandas --with pyarrow python eval/class_count_fill.py"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.class_count_census import load_dataset_info   # noqa: E402

CENSUS = ROOT / "eval/results_class_count_census.json"
SUITE = ROOT / "eval/results_suite_PFN_tabicl2_ds_conv_s0.json"
LOG = ROOT / "logs/e9_TabularJEPA_v3_tabicl2_headenc_s0.log"      # any full suite log: same baselines, same ids
LINE = re.compile(r"^  \[(clf|reg)\] (.+?)\s+did=(\d+)\s+.*?(?:logreg|ridge)=([-\d.]+) (?:histgb|histgbr)=([-\d.]+)")


def main():
    ids = {}   # (bench, name, task, baseline scores) -> id
    bench = None
    for line in LOG.read_text().splitlines():
        if line.startswith("=== "):
            bench = line.split()[1]
        m = LINE.match(line)
        if m:
            task, name, did, lin, gb = m.groups()
            ids[(bench, name, task, float(lin), float(gb))] = int(did)
    cen = json.loads(CENSUS.read_text())
    have = {r["key"]: r for r in cen["datasets"]}
    added, agree = [], 0
    for key, res in json.loads(SUITE.read_text()).items():
        b, name, task = key.split("|")
        if task != "clf":
            continue
        did = ids[(b, name[:28], task, res["logreg"]["acc"], res["histgb"]["acc"])]   # the log prints name[:28]; KeyError = no line with these scores
        n, err = load_dataset_info(did)
        assert err is None, f"{key} id {did}: {err}"
        if key in have:
            assert (have[key]["id"], have[key]["n_classes"]) == (did, n), f"{key}: census {have[key]['id']}/{have[key]['n_classes']} vs log {did}/{n}"
            agree += 1
        else:
            cen["datasets"].append(dict(key=key, id=did, n_classes=n)); added.append((key, did, n))
    cen["unresolved"] = [u for u in cen["unresolved"] if u[0] not in {k for k, _, _ in added}]
    cen.pop("summary", None)   # the original census summarised only the 100 sets it had resolved; nothing reads the block and its counts are stale
    cen["fill_note"] = f"eval/class_count_fill.py: all {len(cen['datasets'])} entries derived or re-derived by matching the suite log's per-dataset baseline scores"   # final state only, so reruns are byte-identical
    CENSUS.write_text(json.dumps(cen, indent=1) + "\n")
    for a in added:
        print("added", *a)
    print(f"agree {agree}, added {len(added)}, unresolved left {len(cen['unresolved'])}")


if __name__ == "__main__":
    main()
