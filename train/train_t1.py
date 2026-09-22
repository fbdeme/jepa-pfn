"""Issue #23 T1a: single-step consumption training (docs/temporal_design.md S12.3).

nan-jeom B = the encode() inject path (pfn.py:150) was never in a training loss, so
the model cannot read a latent at an input slot. T1a puts it there, minimally:

Markov-1 => slice t+1's parents live only in slice t. So give slice t as a LATENT
only (scalar masked, its TRUE cell latent injected, detached like the oracle) and
predict slice t+1 from observed x. The gradient flows through the inject path =>
the consumption path is trained. Teacher-forced, single step, no BPTT (that is T1b).

Total loss = base + consume_weight * consume, base = the arm's own objective:
  base: jepa       -> JEPA latent (+ lambda_ppd value head)   == t0rc_dual
  base: dataspace  -> bar-distribution CE                      == t0rc_ds
Both arms get the SAME consumption term; only the base objective differs (paired).

Truth (z_f, DAG) never enters the loss: targets are observed x, the injected latent
is encode(observed x).detach(), and slice indexing uses observed_idx (task metadata,
like knowing which column is y - not the noise-free f).

Run:  uv run python -m train.train_t1 configs/t1a_dual.yaml [key=value ...]
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.jepa import JEPA
from model.pfn import CellPFN
from prior.scm import PriorConfig, SCMPrior
from train.data import make_batch

DEFAULTS = dict(seed=0, steps=20000, batch_size=16, lr=2e-3, curriculum_frac=0.2,
                emb=96, heads=4, mlp=192, layers=3, n_bins=32, n_reg_tokens=0,
                ema=0.996, lambda_ppd=0.1, predictor="twoway-2", policy_main="mixed",
                policy="any_cell", mode="ema", lambda_sig=0.05, sig_proj=128,
                r_invariant=False, n_cls=4, cls_target=True, r_scheme="rope",
                base="jepa", consume_weight=1.0,
                consume_mode="single", ss_anneal_frac=0.5,   # T1b: rollout + sched-sampling
                log_every=25, val_every=250, val_batches=8, run_name="run", prior={})


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def consume_loss(enc, z, observed_idx, split, V, t):
    """Single-step consumption: slice t given as its TRUE cell latent (detached),
    predict slice t+1's observed value. observed_idx//V maps each (shuffled) column
    to its time slice. Returns CE over slice t+1 query cells."""
    B, R, D = z.shape
    dev = z.device
    sc = (observed_idx // V).unsqueeze(1)              # (B,1,D) slice of each column
    qrow = torch.zeros(B, R, D, dtype=torch.bool, device=dev)
    qrow[:, split:, :] = True
    input_mask = qrow & (sc >= t)                      # query: hide slice >= t (t re-injected)
    inject_mask = qrow & (sc == t)                     # slice t provided as latent
    tgt = qrow & (sc == t + 1)                         # predict slice t+1
    with torch.no_grad():
        h_full = enc.encode(z, torch.zeros_like(input_mask), split)   # true cell latents
    E = enc.value_proj.out_features
    inject_emb = torch.zeros(B, R, D, E, device=dev)
    inject_emb = torch.where(inject_mask.unsqueeze(-1), h_full.detach(), inject_emb)
    h2 = enc.encode(z, input_mask, split, inject_mask=inject_mask, inject_emb=inject_emb)
    logits = enc.head(h2)[tgt]
    return F.cross_entropy(logits, enc.to_bins(z[tgt]))


def rollout_consume_loss(enc, z, observed_idx, split, V, T, eps, rng, scale=None):
    """T1b free-running consumption: matches eval latent_rollout exactly. slice 0 is
    observed, slices>=1 masked; roll t=1..tt injecting each slice's OWN masked-cell
    encoder output h[slice t] (detached) as the next step's input, mixed with the
    TRUE latent h_full w.p. eps (scheduled sampling). Loss = CE at each rolled slice.

    scale not None (train): backward each step's loss immediately (scaled) so injected
    slices are detached between steps => O(1) graph memory over the roll (no BPTT; that
    is T1c). scale None (no_grad validate): just sum the floats. Returns mean float loss.

    Fixes the T1a exposure bias: T1a only ever injected h_full (observed-cell output),
    but eval feeds back h (masked-cell output). Here the model consumes its OWN latent."""
    B, R, D = z.shape
    dev = z.device
    sc = (observed_idx // V).unsqueeze(1)              # (B,1,D) slice of each column
    qrow = torch.zeros(B, R, D, dtype=torch.bool, device=dev)
    qrow[:, split:, :] = True
    input_mask = qrow & (sc >= 1)                      # slice 0 observed, >=1 masked (fixed)
    tt = int(rng.integers(2, T + 1))                   # horizon: >=1 injected step (compounding)
    E = enc.value_proj.out_features
    with torch.no_grad():
        h_full = enc.encode(z, torch.zeros_like(input_mask), split)   # true cell latents
    IM = torch.zeros(B, R, D, dtype=torch.bool, device=dev)
    IE = torch.zeros(B, R, D, E, device=dev)
    total, n = 0.0, 0
    for t in range(1, tt + 1):
        have = bool(IM.any())
        h = enc.encode(z, input_mask, split, inject_mask=IM if have else None,
                       inject_emb=IE if have else None)
        tgt = qrow & (sc == t)
        l = F.cross_entropy(enc.head(h)[tgt], enc.to_bins(z[tgt]))
        total += l.item(); n += 1
        if scale is not None:
            (scale * l).backward()
        if t < tt:                                     # inject slice t for the next step
            coin = (torch.rand(B, 1, D, device=dev) < eps).unsqueeze(-1)   # sched sampling
            src = torch.where(coin, h_full, h.detach())
            sel = (qrow & (sc == t)).unsqueeze(-1)
            IE = torch.where(sel, src, IE)
            IM = IM | (qrow & (sc == t))
    return total / n


def base_loss(model, is_jepa, bt):
    if is_jepa:
        loss, _, _, _ = model(bt["z"], bt["input_mask"], bt["target_mask"], bt["split"])
        return loss
    logits = model(bt["z"], bt["input_mask"], bt["split"])
    return model.loss(logits, bt["z"], bt["target_mask"])


@torch.no_grad()
def validate(model, enc, is_jepa, cfg, V, T, device):
    val_seed = 10_000 + cfg["seed"]
    prior = SCMPrior(PriorConfig(**cfg["prior"]), seed=val_seed)
    rng = np.random.default_rng(val_seed)
    pol = cfg["policy_main"] if is_jepa else cfg["policy"]
    rollout = cfg["consume_mode"] == "rollout"
    b = c = n = 0
    for _ in range(cfg["val_batches"]):
        bt = make_batch(prior, cfg["batch_size"], pol, rng, device, return_truth=True)
        b += base_loss(model, is_jepa, bt).item()
        if rollout:                                    # eps=0: measure pure free-running
            c += rollout_consume_loss(enc, bt["z"], bt["observed_idx"], bt["split"], V, T, 0.0, rng)
        else:
            t = int(rng.integers(1, T))
            c += consume_loss(enc, bt["z"], bt["observed_idx"], bt["split"], V, t).item()
        n += 1
    return b / n, c / n


def main():
    cfg_path, *overrides = sys.argv[1:]
    cfg = {**DEFAULTS, **yaml.safe_load(open(cfg_path))}
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(__file__).parents[1] / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"))
    metrics = open(run_dir / "metrics.jsonl", "a", buffering=1)

    is_jepa = cfg["base"] == "jepa"
    V, T = cfg["prior"]["temporal"]
    prior = SCMPrior(PriorConfig(**cfg["prior"]), seed=cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    if is_jepa:
        model = JEPA(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], cfg["n_bins"],
                     cfg["n_reg_tokens"], cfg["ema"], cfg["lambda_ppd"], cfg["predictor"],
                     cfg["mode"], cfg["lambda_sig"], cfg["sig_proj"], cfg["r_invariant"],
                     cfg["n_cls"], cfg["cls_target"], cfg["r_scheme"]).to(device)
        enc = model.online
    else:
        model = CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], cfg["n_bins"],
                        n_reg_tokens=cfg["n_reg_tokens"], n_cls=cfg["n_cls"],
                        r_scheme=cfg["r_scheme"]).to(device)
        enc = model
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = schedulefree.AdamWScheduleFree(trainable, lr=cfg["lr"], weight_decay=0.0)
    print(f"device={device} base={cfg['base']} trainable={sum(p.numel() for p in trainable):,}",
          flush=True)

    switch = int(cfg["steps"] * cfg["curriculum_frac"])
    rollout = cfg["consume_mode"] == "rollout"
    cw = cfg["consume_weight"]
    model.train(); opt.train()
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        pol = ("y_only" if step <= switch else cfg["policy_main"]) if is_jepa else cfg["policy"]
        bt = make_batch(prior, cfg["batch_size"], pol, rng, device, return_truth=True)
        opt.zero_grad()
        lb = base_loss(model, is_jepa, bt)
        if rollout:                                    # T1b: per-step backward => O(1) memory
            lb.backward()
            eps = max(0.0, 1.0 - step / (cfg["ss_anneal_frac"] * cfg["steps"]))
            lc = rollout_consume_loss(enc, bt["z"], bt["observed_idx"], bt["split"], V, T,
                                      eps, rng, scale=cw)
            lb = lb.item()
        else:                                          # T1a: single-step, combined backward
            tt = int(rng.integers(1, T))               # slice to hand over as a latent
            lcs = consume_loss(enc, bt["z"], bt["observed_idx"], bt["split"], V, tt)
            (lb + cw * lcs).backward()
            lb, lc = lb.item(), lcs.item()
        opt.step()
        if is_jepa:
            model.ema_update()
        if step % cfg["log_every"] == 0:
            metrics.write(json.dumps(dict(step=step, base=round(lb, 4),
                          consume=round(lc, 4), sec=round(time.time() - t0, 1))) + "\n")
        if step % cfg["val_every"] == 0:
            model.eval(); opt.eval()
            vb, vc = validate(model, enc, is_jepa, cfg, V, T, device)
            model.train(); opt.train()
            metrics.write(json.dumps(dict(step=step, val_base=round(vb, 4),
                                          val_consume=round(vc, 4))) + "\n")
    opt.eval()
    torch.save(dict(model=model.state_dict(), cfg=cfg), run_dir / "ckpt.pt")  # load_any format
    print(f"saved {run_dir}/ckpt.pt", flush=True)


def selfcheck():
    """consume_loss runs, is finite, and its masks (given-latent / predict) are
    disjoint and non-empty on a temporal batch."""
    cfg = dict(temporal=(3, 6), temporal_squash=True, n_rows=(64, 64), p_missing_table=0.0)
    prior = SCMPrior(PriorConfig(**cfg), seed=1)
    rng = np.random.default_rng(1)
    enc = CellPFN(48, 4, 96, 2, 32, n_cls=4, r_scheme="rope")
    bt = make_batch(prior, 4, "any_cell", rng, split=48, return_truth=True)
    l = consume_loss(enc, bt["z"], bt["observed_idx"], bt["split"], 3, 2)
    assert torch.isfinite(l), l
    # slice t and t+1 columns are disjoint and both exist per table
    for b in range(4):
        s = bt["observed_idx"][b] // 3
        assert (s == 2).any() and (s == 3).any() and not ((s == 2) & (s == 3)).any()
    # T1b rollout path: no_grad (validate-style) is finite, and train (per-step backward)
    # runs and produces grads without a graph blow-up.
    lr = rollout_consume_loss(enc, bt["z"], bt["observed_idx"], bt["split"], 3, 6, 0.5, rng)
    assert np.isfinite(lr), lr
    enc.zero_grad()
    _ = rollout_consume_loss(enc, bt["z"], bt["observed_idx"], bt["split"], 3, 6, 0.5, rng, scale=1.0)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())
    print(f"selfcheck OK: consume={l.item():.4f} rollout_consume={lr:.4f} finite, "
          f"rollout backward produced grads", flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        selfcheck()
    else:
        main()
