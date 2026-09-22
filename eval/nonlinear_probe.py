"""Follow-up (1): re-measure Phase 4 with nonlinear / attentive probes.

V-JEPA sec 4.3: the latent objective is unnormalized, so linear separability
is not to be expected - re-adjudicate the Phase 4 refutation with
(a) an MLP probe for cell-value recovery (C4', sigma in {1, 8}) and
(b) a V-JEPA-style attentive probe (learned cross-attention pooling + MLP)
    for column-level mechanism probing (C5': edge presence, function family).

Same holdout seeds as eval/truth_eval.py. Probes train on the first half of
batches (tables) and are scored on the second half.

Run: uv run python -m eval.nonlinear_probe
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from eval.truth_eval import c4_collect, embed_batches, encoders
from prior.scm import FuncFamily

ROOT = Path(__file__).parents[1]
CTX = 100
EPOCHS = 300
SEED = 7


def _train(net, X, y, loss_fn, epochs=EPOCHS, lr=1e-3):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(net(X), y).backward()
        opt.step()
    return net


def mlp_probe_mse(H, y):
    """(C4') 2-layer MLP probe: cell embedding -> value. Halves = train/test."""
    torch.manual_seed(SEED)
    H, y = torch.as_tensor(H), torch.as_tensor(y)
    n = len(H) // 2
    net = nn.Sequential(nn.Linear(H.shape[1], 192), nn.GELU(), nn.Linear(192, 1))
    _train(net, H[:n], y[:n, None], nn.functional.mse_loss)
    with torch.no_grad():
        return float(((net(H[n:])[:, 0] - y[n:]) ** 2).mean())


class AttnPool(nn.Module):
    """V-JEPA-style attentive pooling: learnable query token cross-attends
    the column's cell tokens; output feeds an MLP head."""

    def __init__(self, emb, out):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, emb) * 0.02)
        self.attn = nn.MultiheadAttention(emb, 1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(emb, 192), nn.GELU(), nn.Linear(192, out))

    def pool(self, tokens):  # (B_cols, ctx, E)
        return self.attn(self.q.expand(len(tokens), -1, -1), tokens, tokens)[0][:, 0]

    def forward(self, tokens):
        return self.head(self.pool(tokens))


@torch.no_grad()
def collect_columns(enc):
    """Per-column cell tokens (ctx rows) + labels + per-table pair structure."""
    toks, fams, tables = [], [], []
    for h, bt in embed_batches(enc, 1.0, CTX):
        B, _, D, E = h.shape
        for b in range(B):
            start = len(toks)
            for d in range(D):
                toks.append(h[b, :CTX, d])
                fams.append(int(bt["cell_family"][b, d]))
            obs = bt["observed_idx"][b]
            adj = bt["adjacency"][b]
            a = (adj[obs][:, obs] | adj[obs][:, obs].T)
            ii, jj = torch.triu_indices(D, D, offset=1)
            tables.append((start, list(zip(ii.tolist(), jj.tolist(),
                                           a[ii, jj].tolist()))))
    return torch.stack(toks), torch.tensor(fams), tables


def attentive_family_acc(toks, fams):
    torch.manual_seed(SEED)
    n = len(toks) // 2
    net = AttnPool(toks.shape[-1], 4)
    _train(net, toks[:n], fams[:n], nn.functional.cross_entropy)
    with torch.no_grad():
        return float((net(toks[n:]).argmax(1) == fams[n:]).float().mean())


def attentive_edge_auc(toks, tables):
    torch.manual_seed(SEED)
    emb = toks.shape[-1]
    pool = AttnPool(emb, 1)  # head unused; shares pool params
    clf = nn.Sequential(nn.Linear(3 * emb, 192), nn.GELU(), nn.Linear(192, 1))
    half = len(tables) // 2

    def pairs(table_slice):
        idx_i, idx_j, lab = [], [], []
        for start, prs in table_slice:
            for i, j, y in prs:
                idx_i.append(start + i)
                idx_j.append(start + j)
                lab.append(float(y))
        return torch.tensor(idx_i), torch.tensor(idx_j), torch.tensor(lab)

    ti, tj, ty = pairs(tables[:half])
    vi, vj, vy = pairs(tables[half:])
    opt = torch.optim.Adam(list(pool.parameters()) + list(clf.parameters()), lr=1e-3)
    for _ in range(EPOCHS):
        e = pool.pool(toks)
        f = torch.cat([e[ti] + e[tj], (e[ti] - e[tj]).abs(), e[ti] * e[tj]], 1)
        opt.zero_grad()
        nn.functional.binary_cross_entropy_with_logits(clf(f)[:, 0], ty).backward()
        opt.step()
    with torch.no_grad():
        e = pool.pool(toks)
        f = torch.cat([e[vi] + e[vj], (e[vi] - e[vj]).abs(), e[vi] * e[vj]], 1)
        return float(roc_auc_score(vy.numpy(), clf(f)[:, 0].numpy()))


def main(model_names=None):
    encs = encoders()
    if model_names:
        encs = {k: v for k, v in encs.items() if k in model_names}
    ref = json.loads((ROOT / "eval/results_p4.json").read_text())
    results = {}
    for name, enc in encs.items():
        for s in (1.0, 8.0):
            H, F_, X = c4_collect(enc, s)
            results[f"{name}|c4mlp|{s}"] = dict(
                mse_f=round(mlp_probe_mse(H, F_), 4),
                mse_x=round(mlp_probe_mse(H, X), 4),
                lin_f=ref["c4"][f"{name}|{s}"]["mse_f"] if f"{name}|{s}" in ref["c4"] else None)
            print(f"{name:12} C4' sigma={s}: {results[f'{name}|c4mlp|{s}']}", flush=True)
        toks, fams, tables = collect_columns(enc)
        results[f"{name}|c5att"] = dict(
            edge_auc=round(attentive_edge_auc(toks, tables), 4),
            fam_acc=round(attentive_family_acc(toks, fams), 4),
            lin_edge=ref["c5"].get(name, {}).get("edge_auc"),
            lin_fam=ref["c5"].get(name, {}).get("fam_acc"))
        print(f"{name:12} C5' attentive: {results[f'{name}|c5att']}", flush=True)
    out = ROOT / "eval/results_followup_probe.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **results}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
