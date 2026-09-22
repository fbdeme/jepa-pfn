"""Class count census for classification datasets in the real-data suite.

For each classification dataset, counts the number of distinct target classes
to understand how the DS head performance varies by task complexity.

Run: uv run --with openml --with pyarrow python eval/class_count_census.py
Reads: eval/results_suite_real.json, eval/results_realdata_shapes.json,
       ~/.cache/openml/org/openml/www/datasets/<id>/dataset_<id>.pkl.py3,
       ~/.cache/openml/org/openml/www/datasets/<id>/description.xml
Writes: eval/results_class_count_census.json
        paper_v3/review/round13/cls_head_2026-09-04/class_counts.md
"""
import glob
import json
import os
import pickle
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parents[1]
SUITE_RESULTS = ROOT / "eval/results_suite_real.json"
SHAPES_DATA = ROOT / "eval/results_realdata_shapes.json"
OUT_JSON = ROOT / "eval/results_class_count_census.json"
OUT_MD = ROOT / "paper_v3/review/round13/cls_head_2026-09-04/class_counts.md"
CACHE = os.path.expanduser("~/.cache/openml/org/openml/www/datasets")


def load_dataset_info(dataset_id):
    """Load target attribute name and count distinct classes."""
    dataset_dir = Path(CACHE) / str(dataset_id)
    desc_path = dataset_dir / "description.xml"
    pkl_files = glob.glob(str(dataset_dir / "dataset_*.pkl.py3"))

    if not desc_path.exists():
        return None, "description.xml not found"
    if not pkl_files:
        return None, "pkl file not found"

    # Read target attribute from description.xml
    desc_text = desc_path.read_text()
    match = re.search(r"<oml:default_target_attribute>([^<]*)<", desc_text)
    if not match:
        return None, "no default_target_attribute in description.xml"
    target_attr = match.group(1)

    try:
        # Load pickle: (df, categorical_list, attribute_names)
        df, cat_list, attr_names = pickle.load(open(pkl_files[0], "rb"))
    except Exception as e:
        return None, f"pickle load failed: {e}"

    # Verify target attribute exists in df
    if target_attr not in df.columns:
        return None, f"target attribute '{target_attr}' not in DataFrame columns"

    try:
        n_classes = int(df[target_attr].nunique(dropna=True))
        return n_classes, None
    except Exception as e:
        return None, f"nunique failed: {e}"


def main():
    # Load suite results
    suite_data = json.loads(SUITE_RESULTS.read_text())
    suite_datasets = suite_data.get("datasets", [])

    # Load shapes data to create lookup
    shapes_data = json.loads(SHAPES_DATA.read_text())
    shapes_by_bench_name = {}
    for row in shapes_data.get("rows", []):
        key = (row["bench"], row["name"])
        shapes_by_bench_name[key] = row

    # Process classification datasets
    census_rows = []
    unresolved = []

    for ds in suite_datasets:
        if ds.get("task") != "clf":
            continue

        key = ds.get("key", "")
        parts = key.split("|")
        if len(parts) != 3:
            unresolved.append((key, "invalid key format"))
            continue

        bench, name, task = parts

        # Find in shapes data
        shapes_row = shapes_by_bench_name.get((bench, name))
        if not shapes_row:
            unresolved.append((key, "not in results_realdata_shapes.json"))
            continue

        dataset_id = shapes_row.get("id")

        # Load class count
        n_classes, error = load_dataset_info(dataset_id)

        rec = {
            "key": key,
            "id": dataset_id,
            "n_classes": n_classes,
            "ds_mean": ds.get("ds_mean"),
            "dual_mean": ds.get("dual_mean"),
            "linear": ds.get("linear"),
            "histgb": ds.get("histgb"),
        }
        census_rows.append(rec)

        if error:
            unresolved.append((key, error))

    # Sort by n_classes (descending, nulls last)
    def sort_key(rec):
        nc = rec["n_classes"]
        return (nc is None, -(nc or 0))
    census_rows_sorted = sorted(census_rows, key=sort_key)

    # Create summary by class bucket
    def make_buckets():
        buckets = {
            "binary": [],
            "3-5": [],
            "6-10": [],
            ">10": [],
            "null": [],
        }
        for rec in census_rows:
            nc = rec["n_classes"]
            if nc is None:
                buckets["null"].append(rec)
            elif nc == 2:
                buckets["binary"].append(rec)
            elif 3 <= nc <= 5:
                buckets["3-5"].append(rec)
            elif 6 <= nc <= 10:
                buckets["6-10"].append(rec)
            else:
                buckets[">10"].append(rec)
        return buckets

    buckets = make_buckets()

    def bucket_summary(bucket_list):
        if not bucket_list:
            return None
        return {
            "count": len(bucket_list),
            "mean_ds": round(st.mean([r["ds_mean"] for r in bucket_list if r["ds_mean"] is not None]), 4),
            "mean_dual": round(st.mean([r["dual_mean"] for r in bucket_list if r["dual_mean"] is not None]), 4),
            "mean_linear": round(st.mean([r["linear"] for r in bucket_list if r["linear"] is not None]), 4),
            "mean_histgb": round(st.mean([r["histgb"] for r in bucket_list if r["histgb"] is not None]), 4),
            "ds_ge_linear": sum(1 for r in bucket_list if r["ds_mean"] is not None and r["linear"] is not None and r["ds_mean"] >= r["linear"]),
            "ds_ge_histgb": sum(1 for r in bucket_list if r["ds_mean"] is not None and r["histgb"] is not None and r["ds_mean"] >= r["histgb"]),
        }

    summary = {
        "binary": bucket_summary(buckets["binary"]),
        "3-5": bucket_summary(buckets["3-5"]),
        "6-10": bucket_summary(buckets["6-10"]),
        ">10": bucket_summary(buckets[">10"]),
        "null": {"count": len(buckets["null"])},
    }

    output = {
        "summary": summary,
        "datasets": census_rows_sorted,
        "unresolved": unresolved,
    }

    OUT_JSON.write_text(json.dumps(output, indent=1) + "\n")

    # Generate markdown report
    md_lines = ["# Classification Dataset Class Counts\n"]

    # Summary table
    md_lines.append("## Summary by Class Bucket\n")
    md_lines.append("| Bucket | Count | Mean DS | Mean Dual | Mean Linear | Mean HistGB | DS≥Linear | DS≥HistGB |")
    md_lines.append("|--------|-------|---------|-----------|-------------|-------------|-----------|-----------|")

    for bucket_name in ["binary", "3-5", "6-10", ">10"]:
        s = summary[bucket_name]
        if s is None:
            continue
        line = f"| {bucket_name} | {s['count']} | {s['mean_ds']:.4f} | {s['mean_dual']:.4f} | {s['mean_linear']:.4f} | {s['mean_histgb']:.4f} | {s['ds_ge_linear']} | {s['ds_ge_histgb']} |"
        md_lines.append(line)

    if summary["null"]["count"] > 0:
        md_lines.append(f"| null | {summary['null']['count']} | — | — | — | — | — | — |")

    md_lines.append("")

    # Full per-dataset table
    md_lines.append("## Full Dataset Census\n")
    md_lines.append("| Key | n_classes | DS | Dual | Linear | HistGB |")
    md_lines.append("|-----|-----------|----|----|--------|--------|")

    for rec in census_rows_sorted:
        nc = rec["n_classes"] if rec["n_classes"] is not None else "—"
        ds_val = f"{rec['ds_mean']:.4f}" if rec["ds_mean"] is not None else "—"
        dual_val = f"{rec['dual_mean']:.4f}" if rec["dual_mean"] is not None else "—"
        lin_val = f"{rec['linear']:.4f}" if rec["linear"] is not None else "—"
        hgb_val = f"{rec['histgb']:.4f}" if rec["histgb"] is not None else "—"
        line = f"| {rec['key']} | {nc} | {ds_val} | {dual_val} | {lin_val} | {hgb_val} |"
        md_lines.append(line)

    md_lines.append("")
    md_lines.append("DONE\n")

    # Unresolved summary
    if unresolved:
        md_lines.append("## Unresolved Datasets\n")
        for key, reason in unresolved:
            md_lines.append(f"- {key}: {reason}")

    OUT_MD.write_text("\n".join(md_lines))

    print(f"Written {len(census_rows)} census rows to {OUT_JSON}")
    print(f"Written markdown report to {OUT_MD}")
    if unresolved:
        print(f"Warning: {len(unresolved)} unresolved datasets")


if __name__ == "__main__":
    main()
