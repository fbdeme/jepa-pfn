"""Issue #22 pilot: latent rollout vs value rollout vs one-shot on FROZEN models.

The grid (#19/#20) closed the "mechanism recovery" axis negative. This axis is
different: does a latent WM do the one thing a data-space model structurally
cannot - carry an intermediate as a LATENT across a multi-step SCM chain instead
of collapsing it to a scalar - and does that pay off as the chain grows?

Five computations per chain A->...->D (A observed, rest masked; score terminal D
against noise-free z_f):
  one_shot        predict D directly, no stepping
  value_rollout   fill each intermediate with its point_pred SCALAR, re-embed
  latent_rollout  inject each intermediate's ENCODER-OUTPUT latent (no scalar)
  oracle_value    fill intermediates with their TRUE value  (mechanism probe)
  oracle_latent   inject intermediates' TRUE cell latent    (mechanism probe)

The two oracles split "idea wrong" from "injection untrained (nan-jeom B)":
true value and true latent carry the same info, so oracle_latent << oracle_value
means the model just cannot read a latent as input (see docs/rollout_sim.md S6).

Models: p6rc_dual (latent objective) vs p6rc_ds (data-space control). Both have a
trained value head - required for point_pred. eval-only, CPU, free.

Two seeds (30k select, 31k confirm) and chains up to L=6 (in-distribution). Both
models see IDENTICAL chains, so the nan-jeom-B gap (oracle_latent - oracle_value)
is compared PER CHAIN, paired: diff = gap_dual - gap_ds tests whether the latent
objective degrades the injection pathway LESS than data-space, and if that grows
with L (docs/rollout_sim.md S10).

Run:  uv run python -m eval.rollout_sim [--quick]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.noise_grid import load_any, present
from prior.scm import PriorConfig, SCMPrior
from train.data import make_batch

ROOT = Path(__file__).parents[1]
QUICK = "--quick" in sys.argv
def _argv(flag, n):                    # read n args after `flag`, or None
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1:i + 1 + n]
    return None
_t = _argv("--temporal", 2)            # --temporal V T => eval on the time-unrolled prior
TEMPORAL = (int(_t[0]), int(_t[1])) if _t else None
SQUASH = "--squash" in sys.argv        # eval prior tail-stabilized (match T1 training)
_hd = _argv("--hidden", 1)             # --hidden H => PO axis: H hidden variables (POMDP)
HIDDEN = int(_hd[0]) if _hd else 0
_ot = _argv("--tag", 1); OUT_TAG = _ot[0] if _ot else None   # distinct output filename
MODELS = _argv("--models", 2) or (["t0rc_dual", "t0rc_ds"] if TEMPORAL
                                  else ["p6rc_dual", "p6rc_ds"])
SEEDS = [30_000] if QUICK else [30_000, 31_000]        # select -> confirm (SSOT V3.4)
_np = _argv("--n", 1); NPER = int(_np[0]) if _np else 200   # chains per L (lower = faster under load)
TARGETS = ({1: 8, 2: 8, 3: 8} if QUICK
           else {L: NPER for L in range(1, TEMPORAL[1] + 1)} if TEMPORAL
           else {1: 200, 2: 200, 3: 200, 4: 150, 5: 150, 6: 90})  # temporal: 100% chains; struct: L6 rare
LS = list(TARGETS)
BATCH = 16
N_ROWS = 128
SPLIT = 96                              # 96 context / 32 query rows
MODES = ["one_shot", "value_rollout", "latent_rollout", "oracle_value", "oracle_latent"]


def _prior(seed):
    """Eval prior: structural by default, time-unrolled DBN when --temporal is set."""
    cfg = dict(n_rows=(N_ROWS, N_ROWS), p_missing_table=0.0)
    if TEMPORAL:
        cfg["temporal"] = TEMPORAL
        cfg["temporal_squash"] = SQUASH
        if HIDDEN:
            cfg["temporal_hidden"] = (HIDDEN, HIDDEN)      # PO: match training partial obs
    return SCMPrior(PriorConfig(**cfg), seed=seed)


def observed_adj(adj, obs):
    """(D,D) adjacency over columns: adj restricted to observed nodes, in column
    order, so path indices ARE column indices."""
    return adj[np.ix_(obs, obs)]


def find_path(adj_obs, L, rng):
    """A directed path of exactly L edges (L+1 columns) in the observed subgraph,
    or None. N<=8 so brute DFS is trivial; random order avoids always the same path."""
    N = adj_obs.shape[0]
    order = list(rng.permutation(N))

    def dfs(node, depth, seen):
        if depth == L:
            return [node]
        for nxt in order:
            if adj_obs[node, nxt] and nxt not in seen:
                sub = dfs(nxt, depth + 1, seen | {nxt})
                if sub is not None:
                    return [node] + sub
        return None

    for s in order:
        p = dfs(s, 0, {s})
        if p is not None:
            return p
    return None


@torch.no_grad()
def run_modes(enc, z, zf, chain):
    """z, zf: (1,R,D) standardized value / noise-free truth. chain: column indices
    [source, ..., terminal]. Returns {mode: mse of terminal on query rows}."""
    R, D = z.shape[1], z.shape[2]
    E = enc.value_proj.out_features
    term = chain[-1]
    inter = chain[1:-1]                          # intermediates (empty if L==1)
    truth = zf[0, SPLIT:, term]                  # (Rq,) noise-free terminal
    base = torch.zeros(1, R, D, dtype=torch.bool)
    base[0, SPLIT:, chain[1:]] = True            # mask chain[1..L] in query rows

    def latent(z_in, mask, im=None, ie=None):
        return enc.encode(z_in, mask, SPLIT, inject_mask=im, inject_emb=ie)

    def val(h, col):                             # point prediction at one column
        return enc.point_pred(enc.head(h[:, :, col:col + 1]))[0, :, 0]  # (R,)

    def mse(v):
        return ((v[SPLIT:] - truth) ** 2).mean().item()

    out = {}

    # one-shot: whole chain masked, predict terminal directly
    out["one_shot"] = mse(val(latent(z, base), term))

    # value rollout: scalar-fill each intermediate, then predict terminal
    zw, mask = z.clone(), base.clone()
    for col in chain[1:]:
        h = latent(zw, mask)
        if col == term:
            out["value_rollout"] = mse(val(h, col))
        else:
            zw = zw.clone(); zw[0, :, col] = val(h, col)
            mask = mask.clone(); mask[0, :, col] = False

    # latent rollout: inject each intermediate's OUTPUT latent (no scalar collapse)
    im = torch.zeros(1, R, D, dtype=torch.bool)
    ie = torch.zeros(1, R, D, E)
    for col in chain[1:]:
        h = latent(z, base, im if bool(im.any()) else None, ie if bool(im.any()) else None)
        if col == term:
            out["latent_rollout"] = mse(val(h, col))
        else:
            ie[0, :, col, :] = h[0, :, col, :]
            im[0, :, col] = True

    # oracle-value: intermediates = TRUE value (unmask), terminal masked
    mask = base.clone()
    mask[0, SPLIT:, inter] = False               # true z used at intermediates
    out["oracle_value"] = mse(val(latent(z, mask), term))

    # oracle-latent: intermediates = TRUE cell latent injected, terminal masked
    h_full = latent(z, torch.zeros(1, R, D, dtype=torch.bool))
    im = torch.zeros(1, R, D, dtype=torch.bool)
    ie = torch.zeros(1, R, D, E)
    for col in inter:
        ie[0, :, col, :] = h_full[0, :, col, :]
        im[0, :, col] = True
    out["oracle_latent"] = mse(val(latent(z, base, im if bool(im.any()) else None,
                                          ie if bool(im.any()) else None), term))
    return out


@torch.no_grad()
def geom_probe(enc, prior, rng, n_batches=10):
    """nan-jeom B a-priori signal: is the encoder OUTPUT space geometrically near
    the value_proj INPUT space? Low cosine => injecting an output latent as input
    is off-distribution. (S10: cos turned out not to predict the actual penalty.)"""
    cos, nr = [], []
    for _ in range(n_batches):
        bt = make_batch(prior, BATCH, "y_only", rng, split=SPLIT)
        z, no = bt["z"], torch.zeros_like(bt["input_mask"])
        inp = enc.value_proj(z.unsqueeze(-1))
        h = enc.encode(z, no, bt["split"])
        cos.append((F.normalize(inp, dim=-1) * F.normalize(h, dim=-1)).sum(-1).mean().item())
        nr.append((h.norm(dim=-1).mean() / inp.norm(dim=-1).mean()).item())
    return dict(cos=round(float(np.mean(cos)), 4), norm_ratio=round(float(np.mean(nr)), 3))


def paired_collect(encs, seed, targets):
    """Run every chain through ALL models (identical chains, same rng) so gaps can be
    paired per-chain. Returns data[L][name][mode] = per-chain list, aligned across names."""
    prior = _prior(seed)
    rng = np.random.default_rng(seed)
    Ls = list(targets)
    data = {L: {n: {m: [] for m in MODES} for n in encs} for L in Ls}
    need = dict(targets)
    guard = 0
    while any(need[L] > 0 for L in Ls) and guard < 4_000_000:
        guard += 1
        bt = make_batch(prior, BATCH, "y_only", rng, split=SPLIT, return_truth=True)
        adj, obs = bt["adjacency"].numpy(), bt["observed_idx"].numpy()
        for i in range(BATCH):
            ao = observed_adj(adj[i], obs[i])
            for L in Ls:
                if need[L] <= 0:
                    continue
                chain = find_path(ao, L, rng)
                if chain is None:
                    continue
                z, zf = bt["z"][i:i + 1], bt["z_f"][i:i + 1]
                for name, enc in encs.items():
                    errs = run_modes(enc, z, zf, chain)
                    for m in MODES:
                        data[L][name][m].append(errs[m])
                need[L] -= 1
    return data


def stats(data):
    """Per-model mode means (curve) + the paired nan-jeom-B gap test. gap =
    oracle_latent - oracle_value within a model; diff = gap_dual - gap_ds per chain."""
    curve = {n: {} for n in next(iter(data.values()))}
    paired, counts = {}, {}
    for L, per in data.items():
        for n, modes in per.items():
            curve[n][L] = {m: round(float(np.mean(modes[m])), 5) if modes[m] else None
                           for m in MODES}
        counts[L] = len(next(iter(per.values()))["one_shot"])
        if len(MODELS) == 2 and all(m in per for m in MODELS) and counts[L] > 1:
            dual, ds = MODELS                    # [latent/dual, data-space] by convention
            gd = (np.array(per[dual]["oracle_latent"])
                  - np.array(per[dual]["oracle_value"]))
            gs = (np.array(per[ds]["oracle_latent"])
                  - np.array(per[ds]["oracle_value"]))
            d = gd - gs
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            paired[L] = dict(n=len(d), gap_dual=round(float(gd.mean()), 4),
                             gap_ds=round(float(gs.mean()), 4),
                             diff=round(float(d.mean()), 4), se=round(se, 4),
                             t=round(float(d.mean() / se), 2) if se else None)
    return curve, paired, counts


def selfcheck():
    """L==1 has no intermediate, so all five modes must be identical; every error
    finite and positive. The correctness backstop for the injection path."""
    enc = load_any(MODELS[0])
    prior = _prior(1)
    rng = np.random.default_rng(1)
    checked = 0
    while checked < 5:
        bt = make_batch(prior, BATCH, "y_only", rng, split=SPLIT, return_truth=True)
        adj, obs = bt["adjacency"].numpy(), bt["observed_idx"].numpy()
        for i in range(BATCH):
            chain = find_path(observed_adj(adj[i], obs[i]), 1, rng)
            if chain is None:
                continue
            e = run_modes(enc, bt["z"][i:i + 1], bt["z_f"][i:i + 1], chain)
            v = list(e.values())
            assert all(np.isfinite(x) and x >= 0 for x in v), e
            assert max(v) - min(v) < 1e-6, f"L=1 modes must match: {e}"
            checked += 1
            if checked >= 5:
                break
    print("selfcheck OK: L=1 modes identical, errors finite", flush=True)


def run():
    encs = {n: load_any(n) for n in MODELS if present(n)}
    for n in MODELS:
        if n not in encs:
            print(f"skip {n}: not trained", flush=True)
    geom = {n: geom_probe(e, _prior(SEEDS[0]), np.random.default_rng(SEEDS[0]))
            for n, e in encs.items()}
    out = dict(seeds=SEEDS, targets=TARGETS, geom=geom, results={})
    for seed in SEEDS:
        curve, paired, counts = stats(paired_collect(encs, seed, TARGETS))
        out["results"][str(seed)] = dict(counts=counts, curve=curve, paired_gapB=paired)
        print(f"\n#### seed={seed}", flush=True)
        for n in encs:
            print(f"  [{n}] geom(in~out)={geom[n]}", flush=True)
            for L in LS:
                print(f"    L={L} (n={counts[L]}): "
                      + "  ".join(f"{m}={curve[n][L][m]}" for m in MODES), flush=True)
        print("  paired nan-jeom-B gap (dual-ds, neg=dual less broken):", flush=True)
        for L in LS:
            p = paired.get(L)
            if p:
                print(f"    L={L}: gap_dual={p['gap_dual']} gap_ds={p['gap_ds']} "
                      f"diff={p['diff']} se={p['se']} t={p['t']} (n={p['n']})", flush=True)
    tag = (("_temporal" if TEMPORAL else "") + (f"_{OUT_TAG}" if OUT_TAG else "")
           + ("_quick" if QUICK else ""))
    (ROOT / f"eval/results_rollout_sim{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote eval/results_rollout_sim{tag}.json", flush=True)
    return out


if __name__ == "__main__":
    selfcheck()
    run()
