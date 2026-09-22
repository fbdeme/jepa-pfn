"""Aggregate the masked-cell probe into the numbers the paper will quote.

Nothing downstream should read eval/results_probe_masked.json directly and average it by
hand. This writes eval/results_probe_masked_summary.json, and the paper's build scripts and
any write-up read that.

Two things it derives rather than assumes:
  * which seeds are non-degenerate. A collapsed arm returns Var[f] exactly, so the rule is
    "mse_f below DEGENERATE x the floor's own mse_f" -- read off the random-init encoder in
    the same run, not typed in.
  * the old-probe / new-probe comparison, restricted to the seeds that were above the edge
    floor in results_probe_seeds.json, so the two probes are compared on the same runs.

Run: uv run python -m eval.masked_probe_summary
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

NEW = ROOT / "eval/results_probe_masked.json"
OLD = ROOT / "eval/results_probe_seeds.json"
OUT = ROOT / "eval/results_probe_masked_summary.json"

GROUPS = {"ds": "base_ds_s{}", "dual": "base_dual_s{}",
          "lat": "base_lat_s_s{}", "ema_nohead": "base_ema_nohead_s{}"}
OLD_KEY = {"ds": "ds", "dual": "dual", "lat": "lat_s"}   # ema_nohead lives in another file
SEEDS = range(5)
DEGENERATE = 0.98        # fraction of the floor's mse_f at/above which an arm is a constant map
READS = ("cell", "cls_mean", "cls_concat")


def agg(xs):
    return {"mean": round(st.mean(xs), 4),
            "sd": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0,
            "n": len(xs)}


def main():
    new = json.loads(NEW.read_text())
    sig = [s for s in new["_protocol"]["sigmas"]]
    sig = [str(s) for s in sig]
    floor = {s: {rd: new["random_init"][s][rd] for rd in READS} for s in sig}
    # Round-6: the floor here is a SINGLE random-init draw (probe_masked.py seeds it once), and
    # the margin from it to a constant map is smaller than the spread across draws -- a different
    # draw would admit collapsed seeds. Grade against the constant map instead, the same anchor
    # every other table now uses, computed with no encoder at all. Under this protocol the two
    # rules give the same partition, so no reported number moves; what moves is the guarantee.
    from eval.probe_masked_all import const_mse
    const = {s: const_mse(f"headline:{int(float(s))}", "any_cell") for s in sig}
    # Round-9 artifact audit: this table was left on the one-anchor rule while every other table
    # moved to both anchors, and the paper meanwhile claimed "one rule governs every table". The
    # leak is not hypothetical here: .98 x the constant map is .9430, ABOVE this table's own
    # random-init floor of .9327, so the anchor the rule says it excludes would itself pass.
    # Take the stricter of the two, as eval/masked_all_summary.py does.
    cut = {s: round(min(DEGENERATE * const[s], floor[s]["cell"]["mse_f"]), 4) for s in sig}

    out = {"note": "masked-cell probe (eval/probe_masked.py). 'alive' = seeds whose cell "
                   "mse_f is below BOTH null anchors at the same sigma -- DEGENERATE x the "
                   "constant map, and the random-init floor. Aggregates are over seeds.",
           "degenerate_rule": {"fraction_of_constant_map": DEGENERATE,
                               "constant_map_by_sigma": {s: const[s] for s in sig},
                               "floor_by_sigma": {s: floor[s]["cell"]["mse_f"] for s in sig},
                               "cutoff_by_sigma": {s: cut[s] for s in sig},
                               "note": "min(DEGENERATE x constant map, random-init floor): the "
                                       "same two-anchor rule every other table uses, so neither "
                                       "anchor can satisfy the strict inequality"},
           "floor": floor, "sigmas": sig, "per_group": {}}

    for g, pat in GROUPS.items():
        runs = [pat.format(i) for i in SEEDS if pat.format(i) in new]
        rec = {"runs": runs}
        alive = [r for r in runs if new[r]["1.0"]["cell"]["mse_f"] < cut["1.0"]]
        rec["alive"] = alive
        rec["n_alive"] = f"{len(alive)}/{len(runs)}"
        for s in sig:
            rec[s] = {}
            for rd in READS:
                allv = [new[r][s][rd]["mse_f"] for r in runs]
                rec[s][rd] = {"all_seeds": agg(allv),
                              "alive": agg([new[r][s][rd]["mse_f"] for r in alive])
                              if alive else None,
                              "mse_x_alive": agg([new[r][s][rd]["mse_x"] for r in alive])
                              if alive else None}
        out["per_group"][g] = rec

    # old (unmasked, context-cell) vs new (masked) on the SAME runs: those above the edge floor
    old = json.loads(OLD.read_text())
    fl_hi = max(r["edge_auc"] for r in old["floor"])
    cmp_ = {}
    for g, okey in OLD_KEY.items():
        rows = old["trained"][okey]
        idx = [i for i, r in enumerate(rows) if r["edge_auc"] > fl_hi]
        if not idx:
            cmp_[g] = {"n": 0, "note": "no seed above the edge floor"}
            continue
        pat = GROUPS[g]
        cmp_[g] = {"n": len(idx), "seeds": idx,
                   "old_unmasked_mse_f": round(st.mean(rows[i]["mse_f"]["1.0"] for i in idx), 4),
                   "new_masked_mse_f": round(
                       st.mean(new[pat.format(i)]["1.0"]["cell"]["mse_f"] for i in idx), 4)}
    out["old_vs_new_above_edge_floor"] = cmp_
    out["old_vs_new_note"] = ("selection is by the edge-AUC floor rule of "
                              "results_probe_seeds.json, so both probes are read on the "
                              "same runs; the orderings disagree.")

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))
    for g, r in out["per_group"].items():
        c = r["1.0"]["cell"]
        a = c["alive"]
        print(f"  {g:11s} alive {r['n_alive']}  cell mse_f all {c['all_seeds']['mean']:.4f}"
              + (f"  alive {a['mean']:.4f}+/-{a['sd']:.4f}" if a else "  alive --"))
    print(f"  floor       cell mse_f {floor['1.0']['cell']['mse_f']:.4f}"
          f"  cls_mean {floor['1.0']['cls_mean']['mse_f']:.4f}")
    for g, r in cmp_.items():
        if r.get("n"):
            print(f"  {g:11s} old {r['old_unmasked_mse_f']:.4f} -> new "
                  f"{r['new_masked_mse_f']:.4f}  (n={r['n']})")


if __name__ == "__main__":
    main()
