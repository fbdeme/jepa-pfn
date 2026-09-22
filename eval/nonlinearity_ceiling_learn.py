"""Merge the GPU learnability smoke into the difficulty ladder JSON (source-backed): reads the
fetched runs/nonlin_*/metrics.jsonl and writes per-rung learnability into
eval/results_nonlinearity_ceiling.json. No hand-typed val_mse.

Learnability criterion:
  - ds arm: val_mse descends well below 1.0 (mean-predictor MSE ~= 1.0 on unit-variance data).
    learned := final val_mse < 0.90 AND final < first (descending).
  - lat arm: NON-collapse (dim_std stays > 0, cos < 1). A dim_std -> 0 / cos -> 1 arm has
    collapsed to a constant regardless of JEPA loss (rank can stay high -- not a health check).

Run (after `remote.sh fetch`):  uv run python -m eval.nonlinearity_ceiling_learn
"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
JSON = ROOT / "eval" / "results_nonlinearity_ceiling.json"


def _val_mse(run):
    p = ROOT / "runs" / run / "metrics.jsonl"
    if not p.exists():
        return None
    vs = [json.loads(l) for l in p.open() if '"val_mse"' in l]
    return [(r["step"], r["val_mse"]) for r in vs] if vs else None


def _lat_collapse(run):
    p = ROOT / "runs" / run / "metrics.jsonl"
    if not p.exists():
        return None
    vs = [json.loads(l) for l in p.open() if "val_tgt" in l]
    if not vs:
        return None
    t = vs[-1]["val_tgt"]
    return dict(dim_std=t["dim_std"], cos=t["cos"], erank=t["erank"],
                collapsed=bool(t["dim_std"] < 1e-3 and t["cos"] > 0.99))


def main():
    d = json.loads(JSON.read_text())
    for r in d["rungs"]:
        curve = _val_mse(f"nonlin_{r['rung']}_ds")
        if curve:
            final = curve[-1][1]
            r["ds_val_mse_curve"] = [[s, round(v, 4)] for s, v in curve]
            r["ds_val_mse_final"] = round(final, 4)
            r["ds_learned"] = bool(final < 0.90 and final < curve[0][1])
    lat = _lat_collapse("nonlin_L3_lat")
    if lat:
        d["lat_smoke"] = dict(rung="L3", batch=4, mode="sigreg", lambda_sig=0.25, steps=3000,
                              seed=0, note="pure-latent arm on the L3 nonlinear prior", **lat)
    d["smoke"] = dict(arch="base (emb256 heads8 mlp1024 layers6)", ds_batch=8, steps=3000, seed=0,
                      gpu="3x L40 48GB, 1 run/GPU")
    JSON.write_text(json.dumps(d, indent=2))
    print("ds learnability:")
    for r in d["rungs"]:
        if "ds_learned" in r:
            print(f"  {r['rung']:6s} headroom {r['headroom_median']:+.3f}  "
                  f"val_mse {r['ds_val_mse_final']}  learned={r['ds_learned']}")
    if lat:
        print(f"lat L3: dim_std {lat['dim_std']} cos {lat['cos']} -> collapsed={lat['collapsed']}")


if __name__ == "__main__":
    main()
