"""Instrument-trust test: does a KNOWN-GOOD, INDEPENDENT architecture (nanoTabPFN) learn OUR paper
SCM prior? Same nanoTabPFN / median-split / R150-feat5-split100 setup as pc_v1prior.py -- only the
prior is swapped to prior.scm.SCMPrior (the actual paper instrument, default PriorConfig).

Why this matters: our CellPFN already learns our SCM (R256 -> val_mse 0.42), but that alone can't
rule out "SCM and our architecture are co-adapted (both quietly broken in a matching way)" -- the
exact two-variable worry that shelved the paper. If an INDEPENDENT arch also learns our SCM in-
context, the SCM is a legitimate amortizable prior, not a co-adapted artifact. Pairs with the shown
converse (our arch learns real dumps). Chance = 0.5.

Target = a NON-ROOT observed column (has parents => predictable), median-split from TRAIN rows only
(leak-free); the other 5 columns are features. p_missing_table=0 to avoid NaN nuisance (missingness
is a data-quality knob, not a mechanism property).

Run:  uv run --with numpy --with torch --with schedulefree python vendor/nanoTabPFN/pc_scmprior.py [steps]
"""
import json
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import schedulefree
import torch
from torch import nn

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("nanomodel", Path(__file__).parent / "model.py")
_nano = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_nano)
NanoTabPFNModel = _nano.NanoTabPFNModel
from prior.scm import SCMPrior, PriorConfig, FuncFamily  # noqa: E402

FEAT, ROWS, SPLIT = 5, 150, 100
CFG = PriorConfig(n_rows=(ROWS, ROWS), n_cols=(FEAT + 1, FEAT + 1), p_missing_table=0.0)


def nano_batch(prior, bs, device):
    """One nanoTabPFN batch from the SCM prior: pick a non-root observed column as target
    (median-split, train-only threshold), the other FEAT columns as features."""
    xs, ys = [], []
    while len(xs) < bs:
        sb = prior.sample_batch(bs)
        fam = sb.func_family[np.arange(bs)[:, None], sb.observed_idx]   # (bs, D) family per column
        for t in range(bs):
            nonroot = np.where(fam[t] != int(FuncFamily.ROOT))[0]
            x = sb.x[t]
            if len(nonroot) == 0 or not np.isfinite(x).all():
                continue
            tgt = nonroot[0]
            feats = [c for c in range(x.shape[1]) if c != tgt]
            y = x[:, tgt]; thr = np.median(y[:SPLIT])
            xs.append(x[:, feats]); ys.append((y > thr).astype(np.float32))
            if len(xs) == bs:
                break
    x = torch.from_numpy(np.stack(xs)).float().to(device)
    y = torch.from_numpy(np.stack(ys)).to(device)
    return x, y


@torch.no_grad()
def val_acc(model, prior, bs, device, n=8):
    model.eval()
    correct = total = 0
    for _ in range(n):
        x, y = nano_batch(prior, bs, device)
        out = model((x, y[:, :SPLIT]), train_test_split_index=SPLIT)
        pred = out.argmax(-1); tgt = y[:, SPLIT:].long()
        correct += (pred == tgt).sum().item(); total += tgt.numel()
    model.train()
    return correct / total


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    bs, lr = 32, 4e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    model = NanoTabPFNModel(96, 4, 192, 3, 2).to(device)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()
    train_prior = SCMPrior(CFG, seed=0)
    val_prior = SCMPrior(CFG, seed=10_000)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
          f"SCM(default paper prior) rows={ROWS} feat={FEAT} split={SPLIT} steps={steps}", flush=True)
    model.train(); opt.train()
    t0 = time.time()
    curve = []
    for step in range(1, steps + 1):
        x, y = nano_batch(train_prior, bs, device)
        out = model((x, y[:, :SPLIT]), train_test_split_index=SPLIT)
        loss = crit(out.reshape(-1, out.shape[-1]), y[:, SPLIT:].reshape(-1).long())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 50 == 0:
            opt.eval(); va = val_acc(model, val_prior, bs, device); opt.train()
            curve.append([step, round(va, 4), round(loss.item(), 4)])
            print(f"step {step:4d} | loss {loss.item():.4f} | val_acc {va:.4f} | {time.time()-t0:6.1f}s",
                  flush=True)
    # source-backed artifact: deterministic (fixed seeds), read by the paper build as a \num macro.
    out_json = ROOT / "eval" / "results_positive_control.json"
    rec = json.loads(out_json.read_text()) if out_json.exists() else {}
    rec["scm"] = dict(
        prior="SCMPrior default PriorConfig (paper instrument)",
        arch="nanoTabPFN (emb96 heads4 mlp192 layers3)",
        task=f"median-split binary, non-root target, R={ROWS} feat={FEAT} split={SPLIT}",
        chance=0.5, steps=steps, seed=0, val_acc_final=round(curve[-1][1], 4),
        val_acc_curve=curve)
    out_json.write_text(json.dumps(rec, indent=2))
    print(f"saved {out_json} scm val_acc_final={curve[-1][1]:.4f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
