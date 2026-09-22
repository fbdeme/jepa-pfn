"""Extract the shuffled-target control's collapse record into a committed artifact.

The appendix states that all four arbaux seeds are collapsed from the first logged step to
the last. That claim rested on runs/*/metrics.jsonl, which is gitignored -- the same
clean-clone defect round 12 caught in a macro that read runs/. Same fix as the collapse
curve and the ds training curves: extract the load-bearing summary into a committed JSON
with a reproducer, and let the paper anchor macros on that.

Run: uv run python -m eval.extract_arbaux_collapse
Writes eval/results_arbaux_collapse.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "eval/results_arbaux_collapse.json"
COS_COLLAPSED = 0.95        # the paper's own collapse criterion (batch cosine similarity)


def main():
    out = {"criterion": f"target-side batch cosine > {COS_COLLAPSED} at EVERY logged step",
           "source": "runs/c4_factor_arbaux_s*/metrics.jsonl (gitignored; this file is the "
                     "committed summary)", "per_seed": {}}
    for d in sorted((ROOT / "runs").glob("c4_factor_arbaux_s*")):
        rows = [json.loads(l) for l in open(d / "metrics.jsonl")]
        pts = [(r["step"], r["tgt"]["cos"], r["tgt"]["dim_std"]) for r in rows if r.get("tgt")]
        cos = [c for _, c, _ in pts]
        out["per_seed"][d.name] = {
            "n_logged": len(pts), "first_step": pts[0][0], "last_step": pts[-1][0],
            "min_cos": round(min(cos), 4), "last_cos": round(pts[-1][1], 4),
            "last_dim_std": round(pts[-1][2], 4),
            "collapsed_at_every_logged_step": bool(min(cos) > COS_COLLAPSED)}
    n = len(out["per_seed"])
    out["n_seeds"] = n
    out["n_collapsed_throughout"] = sum(
        1 for v in out["per_seed"].values() if v["collapsed_at_every_logged_step"])
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  {out['n_collapsed_throughout']}/{n} seeds collapsed at every logged step")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
