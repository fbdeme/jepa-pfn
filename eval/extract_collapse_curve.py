"""Extract the base SIGReg latent arm's collapse curve into a committed JSON.

The Fig 5 curve data lives in runs/bsw_lat_n1_s0/metrics.jsonl (gitignored, vanishes
when the run dir is cleaned). Pull the per-eval (step, dim_std, cos, erank) into
eval/results_collapse_curve.json so the figure is reproducible from a committed source
(source-backed-numbers). Key steps (peak/onset/permanent) are DERIVED, then cross-checked
against the \\numBaseCollapse* macros.

Run: python -m eval.extract_collapse_curve
"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "runs" / "bsw_lat_n1_s0" / "metrics.jsonl"
OUT = ROOT / "eval" / "results_collapse_curve.json"


def main():
    evals = []
    for line in SRC.read_text().splitlines():
        r = json.loads(line)
        vt = r.get("val_tgt")
        if vt:
            evals.append({"step": r["step"], "dim_std": vt["dim_std"],
                          "cos": vt["cos"], "erank": vt.get("erank")})
    evals.sort(key=lambda e: e["step"])
    peak = max(e["dim_std"] for e in evals)
    peak_step = next(e["step"] for e in evals if e["dim_std"] == peak)
    for e in evals:
        e["dim_std_norm"] = round(e["dim_std"] / peak, 4)

    # derived key steps
    onset = next((e["step"] for e in evals
                  if e["step"] > peak_step and e["dim_std_norm"] < 0.5), None)
    # permanent = first step dim_std collapses to ~0 (the caption's "dim_std -> 0"
    # definition; matches \numBaseCollapsePermanent). The raw curve still shows a minor
    # late bump, which the figure plots honestly -- this only anchors the region label.
    permanent = next((e["step"] for e in evals
                      if e["step"] > peak_step and e["dim_std_norm"] < 0.05), None)

    out = {"run": "bsw_lat_n1_s0", "scale": "6.6M", "metric": "val_tgt (held-out cells)",
           "dim_std_peak": round(peak, 4), "peak_step": peak_step,
           "onset_step": onset, "permanent_step": permanent,
           "evals": evals}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}  ({len(evals)} evals)")
    print(f"peak dim_std={peak:.3f} @ step {peak_step}; onset={onset}; permanent={permanent}")
    print("cross-check vs macros: onset should be 1250, permanent 2250")


if __name__ == "__main__":
    main()
