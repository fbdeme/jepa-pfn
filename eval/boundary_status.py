"""Deterministic progress + result tracker for the factor boundary-mapping experiment.

Do NOT hand-assert "N of the grid points are done". Derive it: scan for each K-point's aggregated
probe result and report done/pending, the trained dual-ds gap, and -- side by side -- the CPU oracle
index that predicted it. This is the status_check pattern (docs-pattern) applied to an experiment's
GRID PROGRESS: the artifacts are the single source of truth for "how far along" and "what it shows".

Grid (pre-registered): K in KS at sigma=1.0, arms {ds, dual}, SEEDS seeds. K=4 reuses the C4 run.
Each other K-point aggregates to eval/results_probe_boundary_K{K}.json with keys
c4_factor_{ds,dual}_K{K}_s{seed} (+ random_init). Oracle map = eval/results_factor_boundary.json.

Run: uv run python -m eval.boundary_status
"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KS = [1, 2, 4, 6, 8, 12]     # intrinsic rank sweep at sigma=1.0 (K=4 = the C4 anchor)
SIGMA = 1.0
SEEDS = [0, 1, 2, 3]         # n=4 uniform (K=4 reuses the C4 4-seed aggregate)


def _oracle_index(K):
    p = ROOT / "eval/results_factor_boundary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    cell = d.get(f"K{K}_s{SIGMA}")
    return cell["index"] if cell else None


def _sd(xs):
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _trained_gap(K):
    """Return (ds, dual, gap=dual-ds, n_seeds, ds_sd) from the K-point, or None if pending. ds_sd
    surfaces the K=1 instability (ds is bimodal there) deterministically. K=4 reuses the C4 aggregate."""
    if K == 4:
        p = ROOT / "eval/results_probe_c4_seeds.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        ds = d["per_arm"]["ds"]["mse_f"]["mean"]
        du = d["per_arm"]["dual"]["mse_f"]["mean"]
        return ds, du, du - ds, d["per_arm"]["dual"]["mse_f"]["n"], d["per_arm"]["ds"]["mse_f"]["sd"]
    p = ROOT / f"eval/results_probe_boundary_K{K}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    dss, dus = [], []
    for s in SEEDS:
        ds, du = f"c4_factor_ds_K{K}_s{s}", f"c4_factor_dual_K{K}_s{s}"
        if ds in d and du in d:
            dss.append(d[ds]["mse_f"]["1.0"])
            dus.append(d[du]["mse_f"]["1.0"])
    if not dss:
        return None
    dsm, dum = sum(dss) / len(dss), sum(dus) / len(dus)
    return dsm, dum, dum - dsm, len(dss), _sd(dss)


def main():
    print(f"factor boundary map @ sigma={SIGMA}, arms {{ds,dual}}, seeds {SEEDS}")
    print(f"{'K':>3}  {'oracle_idx':>10}  {'ds(sd)':>12}  {'dual':>6}  {'dual-ds':>8}  status")
    done = 0
    rows, pts = [], []
    for K in KS:
        idx = _oracle_index(K)
        g = _trained_gap(K)
        if g is None:
            print(f"{K:>3}  {idx if idx is None else f'{idx:+.3f}':>10}  {'':>12}  {'':>6}  {'':>8}  PENDING")
            continue
        done += 1
        ds, du, gap, n, ds_sd = g
        flag = "  <- ds unstable" if ds_sd > 0.1 else ""
        print(f"{K:>3}  {idx:+.3f}      {ds:.3f}(±{ds_sd:.2f})  {du:.3f}  {gap:+.3f}    done(n={n}){flag}")
        rows.append((idx, gap, ds_sd))
        pts.append(dict(K=K, oracle=idx, ds=round(ds, 4), ds_sd=round(ds_sd, 4),
                        dual=round(du, 4), gap=round(gap, 4), n=n))
    print(f"\n{done}/{len(KS)} K-points trained.")

    def _corr(pts):
        idxs, gaps = [p[0] for p in pts], [p[1] for p in pts]
        mi, mg = sum(idxs) / len(idxs), sum(gaps) / len(gaps)
        cov = sum((a - mi) * (b - mg) for a, b in zip(idxs, gaps))
        vi = sum((a - mi) ** 2 for a in idxs) ** 0.5
        vg = sum((b - mg) ** 2 for b in gaps) ** 0.5
        return cov / (vi * vg) if vi and vg else float("nan")

    # Correlation between the oracle index and the trained gap. Expect NEGATIVE (higher index ->
    # more negative gap -> latent credited more). Report the full set AND the ds-stable subset --
    # the K=1 point has bimodal ds (ds_sd high) so the oracle over-predicts it and drags the full
    # correlation; the stable regime is where the oracle actually predicts.
    if len(rows) >= 3:
        print(f"corr(oracle_index, dual-ds gap): all = {_corr(rows):+.2f}")
        stable = [r for r in rows if r[2] < 0.1]
        if len(stable) >= 3:
            print(f"                                 ds-stable only (n={len(stable)}) = {_corr(stable):+.2f}"
                  f"  <- where the oracle predicts; K=1 (bimodal ds) excluded")
        # source-backed summary for paper macros (build_tables.py). Numbers derived, not hand-typed.
        stable_pts = [p for p in pts if p["ds_sd"] < 0.1]
        # how many seeds recover f at the unstable point (ds < 0.1) -> the bimodality
        p1 = ROOT / "eval/results_probe_boundary_K1.json"
        k1_solved = k1_total = None
        if p1.exists():
            d = json.loads(p1.read_text())
            dss = [d[f"c4_factor_ds_K1_s{s}"]["mse_f"]["1.0"] for s in SEEDS
                   if f"c4_factor_ds_K1_s{s}" in d]
            k1_total, k1_solved = len(dss), sum(1 for v in dss if v < 0.1)
        summary = dict(
            seeds=list(SEEDS), sigma=SIGMA, k_values=KS, points=pts,
            corr_all=round(_corr(rows), 2), corr_stable=round(_corr(stable), 2),
            n_stable=len(stable),
            gap_peak=min(p["gap"] for p in stable_pts),          # most negative (K=2)
            gap_min=max(p["gap"] for p in stable_pts),           # least negative (K=12)
            k1_ds_solved=k1_solved, k1_ds_total=k1_total, k1_ds_sd=next((p["ds_sd"] for p in pts if p["K"] == 1), None))
        (ROOT / "eval/results_boundary_summary.json").write_text(json.dumps(summary, indent=1))
        print("wrote eval/results_boundary_summary.json")
    else:
        print("(need >=3 trained points for the oracle-vs-trained correlation)")


if __name__ == "__main__":
    main()
