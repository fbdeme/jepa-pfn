"""Issue #19 — does moving the latent target to the row summary close the gap?

The 2x2 of SSOT V3.3 axis 2. Rows = CLS width k (k=0 is the banked pair),
columns = objective family. Both scoring and loading already exist for the
Issue #18 grid, so this only assembles the pairs:

    gap = data-space - latent, inside a k. Absolute numbers move for dull
    reasons when the architecture changes; only the gap is the measurement.

The pre-registered bar (SSOT V3.4) is C4 + 2-of-3 C5, ON TOP of the noise floor
measured in docs/noise_grid.md - fam_acc within ~.02 is not a win.

Run: uv run python -m eval.cls_grid  ->  eval/results_cls_grid.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.noise_grid import ROOT, load_any, present, score

# k -> (data-space arm, latent arm). k=0 is banked: p2_anycell / p3b_mixed_tw2
# are exactly the pair this experiment adds CLS columns to.
PAIRS = {0: ("p2_anycell", "p3b_mixed_tw2"),
         1: ("p6cls_ds_k1", "p6cls_lat_k1"),
         4: ("p6cls_ds_k4", "p6cls_lat_k4")}
EXTRA = {"p6cls_dual_k4": 4, "p6cls_dual10_k4": 4}  # scored vs their k's ds arm
EVAL_SCALE = 1.0
NOISE_FLOOR = dict(fam_acc=0.02, mse_f=0.007, edge_auc=0.003)


LOSSES = {"mse_f", "mse_x", "deg_mse"}  # lower is better; the rest higher


def gap(ds, lat):
    """POSITIVE = data-space better, on every metric. The losses are flipped to
    get there (lat - ds), the rest are ds - lat.

    NOTE this differs from docs/noise_grid.md, which prints ds - lat throughout
    and leaves the reader to flip the sign on mse_f. Same numbers, and k=0 here
    must match that table up to sign."""
    return {m: round(lat[m] - ds[m] if m in LOSSES else ds[m] - lat[m], 4)
            for m in ds}


def beats_floor(g):
    """Latent wins a metric only by clearing the hardware noise floor
    (docs/noise_grid.md §6) - i.e. gap below -floor, since positive is a loss."""
    return {m: -g[m] > f for m, f in NOISE_FLOOR.items()}


def main():
    out, scores = {}, {}
    for k, (ds, lat) in PAIRS.items():
        missing = [n for n in (ds, lat) if not present(n)]
        if missing:
            print(f"k={k}: missing {missing}, skipped", flush=True)
            continue
        for n in (ds, lat):
            scores.setdefault(n, score(load_any(n), EVAL_SCALE))
        g = gap(scores[ds], scores[lat])
        out[f"k={k}"] = dict(ds=scores[ds], lat=scores[lat], gap=g,
                             latent_wins=beats_floor(g))
        print(f"k={k:<2} {ds} vs {lat}\n  gap {g}\n  latent clears floor: "
              f"{beats_floor(g)}", flush=True)
    for name, k in EXTRA.items():
        if not present(name):
            continue
        ds = PAIRS[k][0]
        scores.setdefault(name, score(load_any(name), EVAL_SCALE))
        g = gap(scores[ds], scores[name])
        out[name] = dict(lat=scores[name], gap=g, latent_wins=beats_floor(g))
        print(f"{name} vs {ds}\n  gap {g}\n  latent clears floor: "
              f"{beats_floor(g)}", flush=True)
    (ROOT / "eval/results_cls_grid.json").write_text(json.dumps(out, indent=1))
    print(f"wrote eval/results_cls_grid.json ({len(out)} cells)")


if __name__ == "__main__":
    main()
