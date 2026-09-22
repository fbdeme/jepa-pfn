"""Issue #15 gate: is the address shortcut actually TAKEN, not just available?

docs/address_leak.md measured that r explains 43.6% of target-embedding
variance for the pure-latent arm. That says the shortcut exists. This says
whether the predictor rides it.

Method: hold the data and the mask fixed, redraw r K times. Each masked
cell's target splits into
    s = s_bar (mean over r draws = CONTENT) + delta (deviation = ADDRESS)
and the prediction splits the same way. Then R^2 on each half separately.

R2_address >> R2_content  =>  the predictor is mostly reproducing addresses.

Run: uv run python -m eval.address_leak   (CPU, forward-only)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.latent_probe import load_cfg, load_jepa, probe_prior
from prior.scm import PriorConfig, SCMPrior
from train.data import make_batch

ROOT = Path(__file__).parents[1]
ARMS = sys.argv[1:] or ["p3b_mixed_tw2", "p3e_rinv", "p3d_ppd_01", "p3d_ppd_10", "p3c_sig_5m4"]
OUT = Path(os.environ.get("LEAK_OUT", ROOT / "eval/results_address_leak.json"))   # named arms -> give an OUT, or the
                                                                                   # committed artifact is overwritten
K_DRAWS, N_BATCHES, BATCH = 8, 12, 8
SEED = 40_000


@torch.no_grad()
def pred_and_target(m, b, r):
    """One (prediction, target) pair at masked cells under a GIVEN r."""
    z, im, tm, split = b["z"], b["input_mask"], b["target_mask"], b["split"]
    h = m._encode_online(z, im, split, r, False)[0]   # ctx_only arms: mask token + key mask; others: unchanged
    pred = m._predict(h, tm, split)
    no_mask = torch.zeros_like(im)
    if m.mode == "ema":
        tgt = m.target.encode(z, no_mask, split, r=r)[tm]
        tgt = F.layer_norm(tgt, tgt.shape[-1:])
    else:
        tgt = m.projector(m.online.encode(z, no_mask, split, r=r))[tm]
    return pred, tgt


@torch.no_grad()
def slot_share(enc, b, k_perm=8):
    """Address share measured by PERMUTING column slots, not by redrawing r.

    Why not reuse the redraw metric: it asks "how much does the target move
    when r is drawn again", which is undefined for a scheme whose column code
    never changes (TabPFN v2's fixed embedding, v3's RoPE) - it would read 0
    for them, not because they carry no address but because the ruler only
    sees moving ones.

    Without any column code our encoder is exactly permutation-equivariant on
    the feature axis (attention over columns has no positional term, row
    attention is per column, the MLP is per cell). So: encode the SAME table
    with columns in a different order, un-permute the output, and whatever
    fails to come back is the slot code's doing. Comparable across all three
    address schemes.
    """
    z, im, split = b["z"], b["input_mask"], b["split"]
    B, R, D = z.shape
    r = enc.sample_r(B, D, z.device)
    outs = []
    for i in range(k_perm):
        p = torch.arange(D) if i == 0 else torch.randperm(D)
        inv = torch.argsort(p)
        h = enc.encode(z[:, :, p], im[:, :, p], split, r=r)
        outs.append(h[:, :, inv])  # back to the original column order
    S = torch.stack(outs)
    var_slot = S.var(0).mean()                 # what the slot code moved
    var_content = S.mean(0).var(dim=(1, 2)).mean()
    return float(var_slot / (var_slot + var_content))


def r2(pred, tgt):
    """Variance of the target explained by the prediction, over cells."""
    resid = (pred - tgt).var(0).sum()
    total = tgt.var(0).sum()
    return float(1 - resid / (total + 1e-12))


def run():
    out = {}
    for arm in ARMS:
        if not (ROOT / "runs" / arm / "ckpt.pt").exists():
            print(f"skip {arm}: no checkpoint", flush=True)
            continue
        m = load_jepa(arm).eval()
        rng = np.random.default_rng(SEED)
        c = load_cfg(arm)
        prior = probe_prior(c, 128, SEED, n_query=50) if c.get("prior_source") else SCMPrior(PriorConfig(), seed=SEED)
        acc = []
        for _ in range(N_BATCHES):
            b = make_batch(prior, BATCH, "any_cell", rng)
            B, _, D = b["z"].shape
            P, T = [], []
            for _ in range(K_DRAWS):
                r = m.online.sample_r(B, D, b["z"].device)
                p, t = pred_and_target(m, b, r)
                P.append(p)
                T.append(t)
            P, T = torch.stack(P), torch.stack(T)          # (K, n_cells, E)
            Pc, Tc = P.mean(0), T.mean(0)                  # content component
            Pd, Td = (P - Pc).flatten(0, 1), (T - Tc).flatten(0, 1)  # address
            acc.append(dict(
                mse=float(F.mse_loss(P, T)),
                r2_all=r2(P.flatten(0, 1), T.flatten(0, 1)),
                r2_content=r2(Pc, Tc),
                r2_address=r2(Pd, Td),
                # share of target variance sitting in each component
                share_address=float(Td.var(0).sum()
                                    / (Td.var(0).sum() + Tc.var(0).sum())),
                # scheme-agnostic twin of the above; the two should agree while
                # the address is a fresh draw, and only slot_share stays defined
                # once it is a fixed code or a RoPE position
                # NOTE: measured on the backbone, so for a SIGReg arm this is a
                # different space than share_address (which includes the
                # projector) - their divergence there is that, not a finding.
                # truth_eval probes the backbone for SIGReg arms too.
                slot_share=slot_share(m.online, b),
            ))
        out[arm] = {k: round(float(np.mean([a[k] for a in acc])), 4) for k in acc[0]}
        print(f"{arm:16} " + "  ".join(f"{k}={v:+.4f}" for k, v in out[arm].items()),
              flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    return out


def _selfcheck():
    """r2() must read 1.0 on a perfect prediction and ~0 on a constant one."""
    t = torch.randn(50, 4)
    assert abs(r2(t, t) - 1.0) < 1e-5
    assert abs(r2(torch.zeros_like(t), t)) < 0.2


if __name__ == "__main__":
    _selfcheck()
    run()
