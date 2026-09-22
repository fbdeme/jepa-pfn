"""Masked-cell mechanism read-out for every arm the paper's result tables report.

Round 5, finding 1 (5 of 6 reviewers): `eval/probe_masked.py` repaired the read-out for the
headline table, and the other result tables still quote `eval/probe_base.py`'s unmasked,
context-cell number -- the read-out Section 3.3 convicts of scoring value retention. The
repair was announced globally and applied in one place. This measures the masked read-out
for the remaining reported arms, in one committed artifact.

It also answers round 5's probe-alignment attack by measurement instead of by argument:
`any_cell` is exactly the policy every data-space arm trains under, so a data-space win
under `any_cell` alone is not admissible evidence. Every arm here is measured under BOTH
`any_cell` and `mixed` (the policy the latent arms train under), and both are reported.

One family per table. Each family mirrors its own table's original protocol -- same prior,
same context size, same batch count, same eval seed -- changing only the three things under
repair:
  * the target cells are HIDDEN before encoding (masking policy + the real input_mask),
  * f is read AT those hidden cells, not at context cells the encoder was shown,
  * the CLS columns are kept, so the columns a latent loss writes to are read as well.

  recipe     tab_recipe_grid    base scale, 21 arms (+ the 5 ema-nohead and dual+LeWM
                                cells that probe_masked.py did not cover)
  sigma      tab_sigma          the pilot noise sweep, each arm at its own training noise
  addr       tab_addr_body (a)  the addressing-scheme pairs, truth_eval's protocol
  substrate  tab_substrate      the low-rank factor prior; the lat arm at the same
                                steelman checkpoint results_c4_lat_full.json selected

NOT covered: tab_nonlinear. The matched_L4_* checkpoints are not on this machine (only the
per-seed eval JSONs survived), so that table cannot be re-measured without retraining.

Run: uv run python -m eval.probe_masked_all              (CPU, forward-only, frozen ckpts)
     MASKED_FAMILIES=recipe,sigma uv run python -m eval.probe_masked_all
     MASKED_FAMILIES=post [POST_RUNS=a,b] [MASKED_OUT=...]   the POST arms on their own (real) prior (C3)
Writes eval/results_probe_masked_all.json (or MASKED_OUT)
"""
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from eval.latent_probe import load_auto, load_cellpfn, load_cfg, load_jepa, probe_prior, random_init, truth_cells   # noqa: E402,F401
from eval.truth_eval import ridge_mse              # noqa: E402
from model.jepa import JEPA                        # noqa: E402
from model.pfn import CellPFN                      # noqa: E402
from prior.scm import FuncFamily, PriorConfig, SCMPrior   # noqa: E402
from train.data import make_batch                  # noqa: E402

OUT = Path(os.environ.get("MASKED_OUT", ROOT / "eval/results_probe_masked_all.json"))
EVAL_SEED = 30_000
POLICIES = ("any_cell", "mixed")
N_FLOOR = int(os.environ.get("N_FLOOR", "3"))   # random-init draws per family, for a band
MAX_N = 24000
CLEAN = dict(p_categorize=0.0, p_missing_table=0.0)


# ---------------------------------------------------------------- priors

def scm_prior(noise, ctx, nq, ncols, mlp_depth=1, mlp_gain=1.0):
    return SCMPrior(PriorConfig(n_rows=(ctx + nq, ctx + nq), n_cols=ncols,
                                noise_scale=noise, mlp_depth=(mlp_depth, mlp_depth),
                                mlp_gain=mlp_gain, **CLEAN), seed=EVAL_SEED)


def factor_prior(ctx, nq, k=4):
    from prior.factor import FactorConfig, FactorPrior
    return FactorPrior(FactorConfig(n_rows=(ctx + nq, ctx + nq), n_cols=(12, 12),
                                    n_factors=k, sigma=1.0), seed=EVAL_SEED)


# ---------------------------------------------------------------- loaders (eval/latent_probe.py)

_cfg = load_cfg


def load_at_step(run, ckpt_file):
    """A JEPA arm at a mid-training checkpoint -- the steelman path, same construction as
    eval/c4_lat_full_probe.build so the two read the same weights."""
    ck = torch.load(ROOT / "runs" / run / ckpt_file, map_location="cpu",
                    weights_only=False)
    c = _cfg(run)
    m = JEPA(c["emb"], c["heads"], c["mlp"], c["layers"], c["n_bins"],
             c["n_reg_tokens"], c["ema"], c["lambda_ppd"],
             c.get("predictor", "mlp"), c.get("mode", "ema"),
             c.get("lambda_sig", 0.05), c.get("sig_proj", 128),
             c.get("r_invariant", False), c.get("n_cls", 0),
             c.get("cls_target", False), c.get("r_scheme", "resample"),
             ctx_only=c.get("ctx_only", False))
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    return m.encoder().eval()


# ---------------------------------------------------------------- the measurement

@torch.no_grad()
def masked_read(enc, prior_fn, policy, ctx, n_batches, batch):
    """Encode with the target cells hidden; return the read-outs at those cells.

    `cell` is the hidden cell's own embedding, `cls_mean` its row's CLS columns averaged
    (dimension-matched to `cell`, so a CLS win cannot be a feature-count artefact), `pred`
    (ctx_only arms only) the predictor's latent at the hidden cell - there the encoder never
    had to fill the cell in, so `cell` doubles as the structural control (expected ~floor).
    `mse_x` against the observed noisy value is the value-retention control: an arm whose
    mse_f tracks its mse_x is preserving values, not isolating mechanism.
    """
    torch.manual_seed(EVAL_SEED)       # paired tables AND column identities across arms
    prior = prior_fn()
    rng = np.random.default_rng(EVAL_SEED)
    K = getattr(enc, "n_cls", 0) or 0
    cell, cmean, pred_, f_, x_ = [], [], [], [], []
    jepa = getattr(enc, "jepa", None)
    for _ in range(n_batches):
        bt = make_batch(prior, batch, policy, rng, split=ctx, return_truth=True)
        tm = bt["target_mask"]
        h = enc.encode(bt["z"], bt["input_mask"], bt["split"], keep_cls=True)
        hc = h[:, :, K:] if K else h
        sel = tm & truth_cells(bt).expand_as(tm)          # C4: non-root AND continuous
        if not sel.any():
            continue
        b, r, d = sel.nonzero(as_tuple=True)
        cell.append(hc[b, r, d])
        if K:
            cmean.append(h[:, :, :K][b, r].mean(1))
        if jepa is not None:
            pred_.append(jepa.predict_latent(bt["z"], bt["input_mask"], tm, bt["split"])[sel[tm]])
        f_.append(bt["z_f"][b, r, d].clamp(-3.05, 3.05))
        x_.append(bt["z"][b, r, d].clamp(-3.05, 3.05))
    f_ = torch.cat(f_).numpy()[:MAX_N]
    x_ = torch.cat(x_).numpy()[:MAX_N]
    feats = {"cell": torch.cat(cell).numpy()[:MAX_N]}
    if K:
        feats["cls_mean"] = torch.cat(cmean).numpy()[:MAX_N]
    if pred_:
        feats["pred"] = torch.cat(pred_).numpy()[:MAX_N]
    return {rd: {"mse_f": round(ridge_mse(X, f_), 4), "mse_x": round(ridge_mse(X, x_), 4),
                 "dim": int(X.shape[1]), "n": int(X.shape[0])}
            for rd, X in feats.items()}


# ---------------------------------------------------------------- families
# (label, loader, extra) per arm; `extra` rides into the record so the table builder does
# not have to re-derive which scheme / noise level / group an arm belongs to.

RECIPE_ARMS = [(f"base_{run}_s{i}", g, i)
               for run, g in (("ds", "ds"), ("dual", "dual"), ("lat_s", "lat"),
                              ("ema_nohead", "ema_nohead"))
               for i in range(5)] + [("base_dual_lewm", "dual_lewm", 0)]

SIGMA_ARMS = [(f"sw_{k}_n{n}_s{s}", k, n, s)
              for n in (1, 3, 10) for k in ("ds", "lat") for s in (0, 1)]

ADDR_PAIRS = [("resample", "p2_anycell", "p3b_mixed_tw2"),
              ("fixed", "p6adr_ds_fixed", "p6adr_lat_fixed"),
              ("rope", "p6adr_ds_rope", "p6adr_lat_rope"),
              ("rope+cls", "p6rc_ds", "p6rc_lat"),
              # The last cell is the one the unmasked probe reported as a reversal, and the
              # paper answers that with a three-seed confirmation. Measure all three seeds
              # here too, so the masked verdict on it rests on the same evidence base.
              ("rope+cls_s1", "p6rc_ds_s1", "p6rc_lat_s1"),
              ("rope+cls_s2", "p6rc_ds_s2", "p6rc_lat_s2")]

SUBSTRATE_ARMS = [(f"c4_factor_{a}_s{s}", a, s)
                  for a in ("ds", "dual", "arbaux") for s in range(4)]

# Round-6 convergent finding: the masked read-out reached no arm above 6.6M and none of the
# target-location grid, so the scale route and the paper's stated dominant cause were still being
# read under the probe Section 3.3 convicts. These three families close that.
BIG_ARMS = [("big_ds", "ds"), ("big_dual", "dual"), ("big_lat_s", "lat"),
            ("big_lat_s_h", "lat_steelman"), ("big_dual_lewm_h", "dual_lewm_steelman")]
# the CLS-location grid: k = how many row-summary columns the latent target is written onto.
CLS_ARMS = [("p2_anycell", "ds", 0), ("p3b_mixed_tw2", "lat", 0),
            ("p6cls_ds_k1", "ds", 1), ("p6cls_lat_k1", "lat", 1),
            ("p6cls_ds_k4", "ds", 4), ("p6cls_lat_k4", "lat", 4),
            ("p6cls_dual_k4", "dual", 4), ("p6cls_dual10_k4", "dual10", 4)]
# the de-confound row set, at base scale, single-seed (the names without _sN)
DECONF_ARMS = [("base_ds", "ds"), ("base_dual", "dual"),
               ("base_lat_s", "lat"), ("base_dual_lewm", "dual_lewm")]

BOUNDARY_K = (1, 2, 4, 6, 8, 12)


def boundary_arms(k):
    """K=4 is the headline substrate and its runs carry no K in the name."""
    tag = "" if k == 4 else f"_K{k}"
    return [(f"c4_factor_{a}{tag}_s{s}", a, s) for a in ("ds", "dual") for s in range(4)]


# The protocol of each family, in one place, so the measurement and everything derived from
# it (the degeneracy anchor below, the summary) cannot drift apart.
PROTOCOL = {
    "recipe": (lambda: scm_prior(1.0, 128, 50, (4, 32)), 128, 8, 8),
    "sigma:1": (lambda: scm_prior(1.0, 128, 50, (4, 32)), 128, 8, 8),
    "sigma:3": (lambda: scm_prior(3.0, 128, 50, (4, 32)), 128, 8, 8),
    "sigma:10": (lambda: scm_prior(10.0, 128, 50, (4, 32)), 128, 8, 8),
    "addr": (lambda: scm_prior(1.0, 100, 50, PriorConfig().n_cols), 100, 24, 16),
    "substrate": (lambda: factor_prior(128, 50), 128, 8, 8),
    # the headline table's own protocol (eval/probe_masked.py), which sweeps read-out
    # noise. Needed so the headline can be graded against the constant map like every
    # other table instead of against a single random-init draw.
    **{f"headline:{int(x)}": (lambda x=x: scm_prior(x, 128, 50, (4, 32)), 128, 8, 8)
       for x in (1.0, 2.0, 4.0, 8.0)},
    # the compressibility sweep: same substrate, intrinsic rank K varied. Each K is its own
    # prior and therefore its own constant-map anchor -- Var[f] moves with K, so a single
    # anchor across the sweep would mis-grade five of the six points.
    **{f"boundary:{k}": (lambda k=k: factor_prior(128, 50, k), 128, 8, 8)
       for k in (1, 2, 4, 6, 8, 12)},
}


def const_mse(key, policy):
    """What a constant map scores on this family+policy: the degeneracy anchor.

    A collapsed encoder gives ridge a feature matrix with no usable variation, so the fit
    keeps only the bias and predicts mean(f) of the training half. `ridge_mse` splits the
    sample in half, so that arm's score is exactly the second half's mean squared deviation
    from the first half's mean -- computable here with no encoder at all.

    This replaces "a fraction of the random-init floor" as the aliveness rule. Under
    `any_cell` the two agree, because there the floor happens to sit just below Var[f]; under
    `mixed` the random-init floor is WORSE than a constant map, so the floor rule would
    certify collapsed arms as alive. The anchor has to be the constant map, not the floor.
    """
    prior_fn, ctx, n_batches, batch = PROTOCOL[key]
    prior = prior_fn()
    rng = np.random.default_rng(EVAL_SEED)
    f_ = []
    for _ in range(n_batches):
        bt = make_batch(prior, batch, policy, rng, split=ctx, return_truth=True)
        tm = bt["target_mask"]
        sel = tm & truth_cells(bt).expand_as(tm)          # C4: non-root AND continuous
        if not sel.any():
            continue
        b, r, d = sel.nonzero(as_tuple=True)
        f_.append(bt["z_f"][b, r, d].clamp(-3.05, 3.05))
    y = torch.cat(f_).numpy()[:MAX_N]
    n = len(y) // 2
    return round(float(np.mean((y[n:] - y[:n].mean()) ** 2)), 4)


def substrate_lat():
    """The lat arm at exactly the checkpoint results_c4_lat_full.json reports: the
    pre-collapse steelman where the seed collapsed, the 20k checkpoint where it did not.
    Reading a different checkpoint would compare a different model, not a different probe."""
    d = json.loads((ROOT / "eval/results_c4_lat_full.json").read_text())
    out = []
    for s in range(4):
        run = f"c4_factor_lat_s{s}_full"
        e = d.get(run)
        if not e:
            continue
        step = e.get("steelman_step")
        out.append((run, f"ckpt_{step}.pt" if step else "ckpt.pt", s, step))
    return out


def post_family(runs, ctx=128, n_batches=8, batch=8):
    """C3: the POST arms (real-prior runs, later the v4 grid) on the prior THEY trained on - built from the
    first run's ckpt cfg by probe_prior, so one family = one prior = one constant-map anchor (asserted)."""
    cfgs = {r: load_cfg(r) for r in runs}
    keys = {(c.get("prior_source"), json.dumps(c.get("prior", {}), sort_keys=True)) for c in cfgs.values()}
    if len(keys) != 1:
        raise SystemExit(f"post family needs one prior, got {keys}")
    ref = runs[0]
    PROTOCOL["post"] = (lambda: probe_prior(cfgs[ref], ctx, EVAL_SEED, n_query=50), ctx, n_batches, batch)
    jobs = [(r, (lambda r=r: load_auto(r)), {"group": "ds" if "_ds_" in r else "dual", "prior_source": cfgs[r].get("prior_source") or "paper"})
            for r in runs]
    floor_ref = next((r for r in runs if "_ds_" in r), ref)
    return run_family("post", jobs, *PROTOCOL["post"], floor_ref, "post")


def _agg(xs):
    return {"mean": round(st.mean(xs), 4),
            "sd": round(st.stdev(xs), 4) if len(xs) > 1 else 0.0, "n": len(xs)}


def run_family(name, jobs, prior_fn, ctx, n_batches, batch, floor_ref, key=None):
    """jobs: list of (arm_id, encoder_factory, meta_dict). Returns the family record."""
    rec = {"arms": {}, "protocol": {"ctx": ctx, "n_batches": n_batches, "batch": batch,
                                    "eval_seed": EVAL_SEED, "policies": list(POLICIES),
                                    "floor_ref_arch": floor_ref, "n_floor": N_FLOOR,
                                    "protocol_key": key}}
    if key:      # the constant-map score: the degeneracy anchor, no encoder involved
        rec["const_map_mse_f"] = {p: const_mse(key, p) for p in POLICIES}
    for arm, factory, meta in jobs:
        t = time.time()
        enc = factory()
        r = {"meta": meta}
        for p in POLICIES:
            r[p] = masked_read(enc, prior_fn, p, ctx, n_batches, batch)
        rec["arms"][arm] = r
        print(f"  {name}/{arm:24} "
              + "  ".join(f"{p}: f={r[p]['cell']['mse_f']:.4f}/x={r[p]['cell']['mse_x']:.4f}"
                          for p in POLICIES)
              + f"   [{time.time()-t:.0f}s]", flush=True)
    draws = {p: [] for p in POLICIES}
    for d in range(N_FLOOR):
        enc = random_init(floor_ref, 10_000 + d)
        for p in POLICIES:
            draws[p].append(masked_read(enc, prior_fn, p, ctx, n_batches,
                                        batch)["cell"]["mse_f"])
    rec["floor"] = {p: dict(_agg(draws[p]), draws=draws[p]) for p in POLICIES}
    print(f"  {name}/floor                    "
          + "  ".join(f"{p}: f={rec['floor'][p]['mean']:.4f}" for p in POLICIES), flush=True)
    return rec


def main():
    want = os.environ.get("MASKED_FAMILIES", "recipe,sigma,addr,substrate,boundary,big,cls,deconf").split(",")
    # Merge into whatever is already measured, so running one family (e.g. to deepen its
    # floor band) does not silently drop the other three from the artifact.
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()

    if "recipe" in want:
        # probe_base's base protocol: ctx 128, wide-D prior, 8x8, noise 1.0
        jobs = [(a, (lambda a=a: load_auto(a)), {"group": g, "seed": s})
                for a, g, s in RECIPE_ARMS if (ROOT / "runs" / a / "ckpt.pt").exists()]
        res["recipe"] = run_family("recipe", jobs, *PROTOCOL["recipe"], "base_ds_s0", "recipe")

    if "sigma" in want:
        # sigma_sweep_probe's protocol: each arm probed at its OWN training noise.
        fam = {"arms": {}, "by_noise": {}}
        for n in (1, 3, 10):
            arms = [(a, k, s) for a, k, nn, s in SIGMA_ARMS if nn == n]
            jobs = [(a, (lambda a=a: load_auto(a)), {"kind": k, "noise_scale": n, "seed": s})
                    for a, k, s in arms if (ROOT / "runs" / a / "ckpt.pt").exists()]
            sub = run_family(f"sigma_n{n}", jobs, *PROTOCOL[f"sigma:{n}"],
                             f"sw_ds_n{n}_s0", f"sigma:{n}")
            fam["arms"].update(sub["arms"])
            fam["by_noise"][str(n)] = {"floor": sub["floor"], "protocol": sub["protocol"]}
        res["sigma"] = fam

    if "addr" in want:
        # truth_eval's protocol (what results_addr_grid.json was scored under): ctx 100,
        # the DEFAULT n_cols pilot prior, 24x16.
        # One floor per scheme: the schemes differ in r_scheme, so a single random-init
        # arch would be the wrong null for three of the four rows.
        fam = {"arms": {}, "by_scheme": {}}
        for scheme, ds, lat in ADDR_PAIRS:
            jobs = [(a, (lambda a=a: load_auto(a)), {"scheme": scheme, "kind": k})
                    for a, k in ((ds, "ds"), (lat, "lat"))
                    if (ROOT / "runs" / a / "ckpt.pt").exists()]
            if not jobs:
                continue
            sub = run_family(f"addr[{scheme}]", jobs, *PROTOCOL["addr"], ds, "addr")
            fam["arms"].update(sub["arms"])
            fam["by_scheme"][scheme] = {"floor": sub["floor"], "protocol": sub["protocol"],
                                        "ds": ds, "lat": lat}
        res["addr"] = fam

    if "substrate" in want:
        jobs = [(a, (lambda a=a: load_auto(a)), {"group": g, "seed": s})
                for a, g, s in SUBSTRATE_ARMS if (ROOT / "runs" / a / "ckpt.pt").exists()]
        for run, ckpt, s, step in substrate_lat():
            jobs.append((run, (lambda r=run, c=ckpt: load_at_step(r, c)),
                         {"group": "lat", "seed": s, "ckpt": ckpt, "steelman_step": step}))
        res["substrate"] = run_family("substrate", jobs, *PROTOCOL["substrate"],
                                      "c4_factor_ds_s0", "substrate")

    if "big" in want:
        jobs = [(a, (lambda a=a: load_auto(a)), {"group": g})
                for a, g in BIG_ARMS if (ROOT / "runs" / a / "ckpt.pt").exists()]
        res["big"] = run_family("big", jobs, *PROTOCOL["recipe"], "big_ds", "recipe")

    if "cls" in want:
        jobs = [(a, (lambda a=a: load_auto(a)), {"group": g, "k": k})
                for a, g, k in CLS_ARMS if (ROOT / "runs" / a / "ckpt.pt").exists()]
        res["cls"] = run_family("cls", jobs, *PROTOCOL["addr"], "p2_anycell", "addr")

    if "deconf" in want:
        jobs = [(a, (lambda a=a: load_auto(a)), {"group": g})
                for a, g in DECONF_ARMS if (ROOT / "runs" / a / "ckpt.pt").exists()]
        res["deconf"] = run_family("deconf", jobs, *PROTOCOL["recipe"], "base_ds", "recipe")

    if "boundary" in want:
        # The compressibility sweep, on the corrected read-out. What this measures is not a
        # margin but a SUCCESS RATE: on this substrate the data-space arm is bimodal, either
        # solving the task or never leaving the constant map, and the rate is what moves with K.
        fam = {"arms": {}, "by_k": {}}
        for k in BOUNDARY_K:
            arms = [(a, g, s) for a, g, s in boundary_arms(k)
                    if (ROOT / "runs" / a / "ckpt.pt").exists()]
            if not arms:
                print(f"  boundary K={k}: no checkpoints, skipped", flush=True)
                continue
            jobs = [(a, (lambda a=a: load_auto(a)), {"group": g, "seed": s, "K": k})
                    for a, g, s in arms]
            ref = next(a for a, g, _ in arms if g == "ds")
            sub = run_family(f"boundary_K{k}", jobs, *PROTOCOL[f"boundary:{k}"],
                             ref, f"boundary:{k}")
            fam["arms"].update(sub["arms"])
            fam["by_k"][str(k)] = {"floor": sub["floor"],
                                   "const_map_mse_f": sub["const_map_mse_f"],
                                   "protocol": sub["protocol"]}
        res["boundary"] = fam

    if "post" in want:
        # POST_RUNS=a,b,c to name them; default = every real-prior run with a checkpoint
        named = os.environ.get("POST_RUNS")
        runs = named.split(",") if named else sorted(p.name for p in (ROOT / "runs").glob("real_*_lr*_s*")
                                                     if (p / "ckpt.pt").exists())
        res["post"] = post_family(runs)

    res["_note"] = (
        "Masked-cell read-out for the arms behind tab_recipe_grid, tab_sigma, "
        "tab_addr_body(a) and tab_substrate. Each family mirrors the protocol its own "
        "unmasked table was scored under, changing only: target cells hidden before "
        "encoding, f read at those hidden cells, CLS columns kept. Both masking policies "
        "are reported -- any_cell is the data-space arms' training policy and mixed is the "
        "latent arms', so neither alone is a neutral test. tab_nonlinear is absent: the "
        "matched_L4_* checkpoints are not on this machine.")
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
