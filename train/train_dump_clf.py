"""Architecture-isolation test: OUR CellPFN backbone (identical to base_ds -- same emb/heads/
layers/rope/CLS) + a PROPER softmax classification head instead of the regression-bin head,
trained on the TabPFN-prior dump the way nanoTabPFN trains (predict the label from an in-context
(X,y) split). The backbone is byte-for-byte the base_ds backbone; only the head changes.

Decisive metric: on HELD-OUT dump tables, does our backbone+softmax head match per-table GBM /
logreg (strong learners on the SAME distribution)? If yes, the backbone learns TabPFN-prior
classification as well as the strong baselines -> the backbone is NOT the bottleneck, and our
paper's regression-bin head was the classification cap. If no, the backbone is genuinely limited.

Run: uv run --with h5py --with scikit-learn python -m train.train_dump_clf [steps=15000]
"""
import sys
import time
from pathlib import Path

import numpy as np
import schedulefree
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from model.pfn import CellPFN
from train.data_dump import DumpPrior

CFG = dict(steps=15000, batch_size=8, lr=1e-3, emb=256, heads=8, mlp=1024, layers=6,
           n_cls=4, r_scheme="rope", feat=5, seed=1, dump_path="data/tabpfn_prior_dump.h5")


def clf_batch(prior, bs, rng, device):
    """context rows [:split] show (X, standardized label); query rows [split:] hide the label.
    Returns z, input_mask, split, and the true integer query labels."""
    x = prior.sample_batch(bs).x                       # (B, R, feat+1), label = last col
    B, R, D = x.shape
    split = int(np.clip(int(R * rng.uniform(0.4, 0.8)), 16, R - 8))
    ctx = x[:, :split]
    mean, std = ctx.mean(1, keepdims=True), ctx.std(1, keepdims=True) + 1e-8
    z = np.clip((x - mean) / std, -100, 100).astype(np.float32)
    input_mask = np.zeros((B, R, D), bool)
    input_mask[:, split:, -1] = True                   # hide query labels
    y_q = x[:, split:, -1].astype(np.int64)            # true labels (0/1), un-standardized
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return t(z, torch.float32), t(input_mask, torch.bool), split, t(y_q, torch.long)


def predict(model, clf_head, z, input_mask, split):
    """Query-row label logits from the label cell's post-backbone embedding (same read as nano)."""
    h = model.encode(z, input_mask, split)             # (B,R,D,E), backbone UNCHANGED
    return clf_head(h[:, split:, -1, :])               # (B, n_query, C)


@torch.no_grad()
def evaluate(model, clf_head, prior, device, n_tables=240, seed=999):
    """Held-out dump tables: our in-context model vs per-table GBM / logreg, same split."""
    model.eval()
    rng = np.random.default_rng(seed)
    ours, gbm, lr, n = 0, 0, 0, 0
    for _ in range(n_tables // 8):
        x = prior.sample_batch(8).x
        B, R, D = x.shape
        split = int(R * 0.7)
        z, im, sp, yq = clf_batch_fixed(x, split, device)
        pred = predict(model, clf_head, z, im, sp).argmax(-1).cpu().numpy()  # (B, nq)
        yq = yq.cpu().numpy()
        for b in range(B):
            Xtr, ytr = x[b, :split, :-1], x[b, :split, -1].astype(int)
            Xte, yte = x[b, split:, :-1], x[b, split:, -1].astype(int)
            if len(np.unique(ytr)) < 2:
                continue
            ours += (pred[b] == yte).sum()
            gbm += (HistGradientBoostingClassifier(max_iter=100).fit(Xtr, ytr).predict(Xte) == yte).sum()
            lr += (LogisticRegression(max_iter=200).fit(Xtr, ytr).predict(Xte) == yte).sum()
            n += len(yte)
    model.train()
    return ours / n, gbm / n, lr / n


def clf_batch_fixed(x, split, device):
    """Same standardization/masking as clf_batch but with a fixed split (for eval)."""
    B, R, D = x.shape
    ctx = x[:, :split]
    mean, std = ctx.mean(1, keepdims=True), ctx.std(1, keepdims=True) + 1e-8
    z = np.clip((x - mean) / std, -100, 100).astype(np.float32)
    im = np.zeros((B, R, D), bool)
    im[:, split:, -1] = True
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    return t(z, torch.float32), t(im, torch.bool), split, t(x[:, split:, -1].astype(np.int64), torch.long)


def main():
    cfg = dict(CFG)
    for kv in sys.argv[1:]:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v) if k in cfg else v
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_prior = DumpPrior(cfg["dump_path"], seed=cfg["seed"], split="train", feat=cfg["feat"])
    val_prior = DumpPrior(cfg["dump_path"], seed=7, split="val", feat=cfg["feat"])
    model = CellPFN(cfg["emb"], cfg["heads"], cfg["mlp"], cfg["layers"], n_cls=cfg["n_cls"],
                    r_scheme=cfg["r_scheme"]).to(device)               # backbone == base_ds
    clf_head = nn.Linear(cfg["emb"], 2).to(device)                     # the ONLY change
    params = list(model.parameters()) + list(clf_head.parameters())
    opt = schedulefree.AdamWScheduleFree(params, lr=cfg["lr"], weight_decay=0.0)
    print(f"device={device} backbone_params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    rng = np.random.default_rng(cfg["seed"])
    model.train(); opt.train()
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        z, im, sp, yq = clf_batch(train_prior, cfg["batch_size"], rng, device)
        logits = predict(model, clf_head, z, im, sp)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 2), yq.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
        if step % 500 == 0:
            print(dict(step=step, loss=round(loss.item(), 4), sec=round(time.time() - t0, 1)),
                  flush=True)
        if step % 2500 == 0:
            opt.eval()
            o, g, l = evaluate(model, clf_head, val_prior, device)
            print(f"[eval step {step}] held-out dump acc: OURS={o:.4f}  GBM={g:.4f}  "
                  f"logreg={l:.4f}", flush=True)
            opt.train()
    opt.eval()
    o, g, l = evaluate(model, clf_head, val_prior, device, n_tables=480)
    print(f"[FINAL] held-out dump acc: OURS={o:.4f}  GBM={g:.4f}  logreg={l:.4f}", flush=True)
    torch.save(dict(model=model.state_dict(), clf_head=clf_head.state_dict(), cfg=cfg),
               Path(__file__).parents[1] / "runs" / "dump_clf_iso.pt")


if __name__ == "__main__":
    main()
