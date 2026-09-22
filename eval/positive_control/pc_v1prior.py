"""Positive control: does a KNOWN-GOOD tabular-ICL architecture (nanoTabPFN) learn OUR revived
TabPFN-v1 prior? Isolates the confounded variable in the shelved comparison. Our CellPFN failed to
learn the v1 prior; nanoTabPFN already learns its OWN bundled prior (300k_150x5 -> breast_cancer
roc_auc 0.98). If nanoTabPFN ALSO learns our v1 extraction -> prior is fine, our architecture is the
culprit. If it fails too -> our v1 extraction/setup is suspect.

nanoTabPFN is a classifier (CE over classes); the v1 prior gives a continuous target, so we
median-split each table (threshold from the TRAIN rows only -> leak-free) into a balanced 2-class
label. Primary signal = in-distribution held-out accuracy on fresh v1 tables (chance = 0.5).

Run:  uv run --with numpy --with torch --with einops --with networkx --with schedulefree \
        python vendor/nanoTabPFN/pc_v1prior.py [steps] [rows] [feat]
"""
import sys
import time
from pathlib import Path

import numpy as np
import schedulefree
import torch
from torch import nn

import importlib.util

ROOT = Path(__file__).parents[2]              # jepa-pfn
sys.path.insert(0, str(ROOT))                 # `train` package resolves here...
# ...but nanoTabPFN's own train.py would shadow it, so load model.py by file path.
_spec = importlib.util.spec_from_file_location("nanomodel", Path(__file__).parent / "model.py")
_nano = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_nano)
NanoTabPFNModel = _nano.NanoTabPFNModel
from train.data_tabpfn_v1 import TabPFNv1Prior  # noqa: E402


def nano_batch(prior, bs, split, device):
    """One nanoTabPFN batch from the v1 prior: x=(B,R,C) features, y=(B,R) balanced 0/1 labels
    (median of TRAIN rows), split index. Oversample+filter the rare non-finite table."""
    xs, ys = [], []
    while len(xs) < bs:
        b = prior.sample_batch(bs).x                      # (bs, R, feat+1), target = last col
        ok = np.isfinite(b).all(axis=(1, 2))
        for t in b[ok]:
            x, y = t[:, :-1], t[:, -1]
            thr = np.median(y[:split])                    # train-only threshold -> leak-free
            xs.append(x); ys.append((y > thr).astype(np.float32))
            if len(xs) == bs:
                break
    x = torch.from_numpy(np.stack(xs)).float().to(device)   # (B,R,C)
    y = torch.from_numpy(np.stack(ys)).to(device)           # (B,R) float 0./1. (encoder needs float)
    return x, y


@torch.no_grad()
def val_acc(model, prior, bs, split, device, n=8):
    """Held-out in-distribution accuracy on the TEST rows of fresh v1 tables (chance 0.5)."""
    model.eval()
    correct = total = 0
    for _ in range(n):
        x, y = nano_batch(prior, bs, split, device)
        out = model((x, y[:, :split]), train_test_split_index=split)   # (B, R-split, 2)
        pred = out.argmax(-1)
        tgt = y[:, split:].long()
        correct += (pred == tgt).sum().item(); total += tgt.numel()
    model.train()
    return correct / total


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    rows = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    feat = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    split = rows * 2 // 3                                   # 100 for R=150 (matches native ~82-120)
    bs, lr = 32, 4e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    model = NanoTabPFNModel(embedding_size=96, num_attention_heads=4, mlp_hidden_size=192,
                            num_layers=3, num_outputs=2).to(device)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()
    train_prior = TabPFNv1Prior(seed=0, rows=rows, feat=feat)
    val_prior = TabPFNv1Prior(seed=10_000, rows=rows, feat=feat)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
          f"rows={rows} feat={feat} split={split} steps={steps}", flush=True)

    model.train(); opt.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = nano_batch(train_prior, bs, split, device)
        out = model((x, y[:, :split]), train_test_split_index=split)   # (B, R-split, 2)
        loss = crit(out.reshape(-1, out.shape[-1]), y[:, split:].reshape(-1).long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 50 == 0:
            opt.eval()
            va = val_acc(model, val_prior, bs, split, device)
            opt.train()
            print(f"step {step:4d} | loss {loss.item():.4f} | val_acc {va:.4f} "
                  f"| {time.time()-t0:6.1f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
