"""Cell-token PFN with two-way attention (Phase 2 baselines).

nanoTabPFN architecture (vendor/nanoTabPFN/model.py) extended per SSOT:
- r_j: random column-identity vectors, resampled every forward (lifetime =
  one context), so any-cell masking has live cell addresses
- learned [MASK] embedding instead of nanoTabPFN's y-mean padding; also used
  for missing (NaN) inputs
- readout at every cell via a bar-distribution head (K bins on the
  standardized value scale) - one head serves y-only and any-cell policies,
  continuous and categorical columns alike

Inputs are column-standardized with context-row statistics (see
train/data.py); this module never sees raw scales.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class RotaryFeatAttn(nn.Module):
    """Multi-head self-attention over the feature (column) axis with rotary
    position on Q,K only (Issue #20, TabPFN v3 scheme). Replaces the additive
    column code: position enters the attention, not the residual stream, so the
    output embedding carries no content-free address vector."""

    def __init__(self, emb, heads):
        super().__init__()
        assert emb % heads == 0 and (emb // heads) % 2 == 0
        self.h, self.dh = heads, emb // heads
        self.qkv = nn.Linear(emb, 3 * emb)
        self.proj = nn.Linear(emb, emb)
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.dh, 2).float() / self.dh))
        self.register_buffer("inv_freq", inv)

    def _rope(self, x, pos):  # x (N,h,L,dh), pos (L,)
        ang = pos[:, None] * self.inv_freq[None, :]          # (L, dh/2)
        cos = ang.cos().repeat_interleave(2, -1)             # (L, dh)
        sin = ang.sin().repeat_interleave(2, -1)
        x2 = x.reshape(*x.shape[:-1], -1, 2)
        rot = torch.stack([-x2[..., 1], x2[..., 0]], -1).reshape_as(x)
        return x * cos + rot * sin

    def forward(self, s, key_mask=None):  # s (N, L, E); key_mask (N, L) bool = tokens no OTHER token may attend to
        N, L, E = s.shape
        q, k, v = (t.view(N, L, self.h, self.dh).transpose(1, 2)
                   for t in self.qkv(s).chunk(3, -1))
        pos = torch.arange(L, device=s.device, dtype=s.dtype)
        q, k = self._rope(q, pos), self._rope(k, pos)
        allowed = None
        if key_mask is not None:   # ctx_only (I-JEPA context encoder): hidden cells are not keys; self stays allowed so no row is all -inf
            allowed = ~key_mask[:, None, None, :] | torch.eye(L, dtype=torch.bool, device=s.device)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.proj(o.transpose(1, 2).reshape(N, L, E))


class RowAttnSSMax(nn.Module):
    """Row-axis attention with Scalable-Softmax (TabICL, _model/ssmax.py). Plain softmax
    attention spreads thin as the number of context rows grows -- past ~1000 rows it diffuses to
    a uniform average, the row embedding collapses toward the mean, and the head can only predict
    the marginal (measured: our default attn stalls at val_mse~1.0 / grad->0 at R=1024). SSMax
    scales the queries per head by s_h * log(n_keys) so attention stays sharp at any row count.
    s_h is learnable (init 1). Same context/query PFN rule as the default row attention."""

    def __init__(self, emb, heads):
        super().__init__()
        self.h, self.dh = heads, emb // heads
        self.qkv = nn.Linear(emb, 3 * emb)
        self.proj = nn.Linear(emb, emb)
        self.s = nn.Parameter(torch.ones(heads))

    def forward(self, q_in, kv_in):  # q_in (N,Lq,E), kv_in (N,Lk,E)
        N, Lq, E = q_in.shape
        Lk = kv_in.shape[1]
        qw, kw, vw = self.qkv.weight.chunk(3, 0)
        qb, kb, vb = self.qkv.bias.chunk(3, 0)
        q = F.linear(q_in, qw, qb).view(N, Lq, self.h, self.dh).transpose(1, 2)
        k = F.linear(kv_in, kw, kb).view(N, Lk, self.h, self.dh).transpose(1, 2)
        v = F.linear(kv_in, vw, vb).view(N, Lk, self.h, self.dh).transpose(1, 2)
        q = q * (self.s * math.log(max(Lk, 2))).view(1, self.h, 1, 1)  # SSMax query scaling
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(N, Lq, E))


class TwoWayBlock(nn.Module):
    """Feature-axis attention, then sample-axis attention, then cell MLP.

    Sample-axis rule (as nanoTabPFN): context rows attend to context rows;
    query rows attend only to context rows (queries blocked from each other).
    """

    def __init__(self, emb, heads, mlp, rope=False, ssmax=False):
        super().__init__()
        # rope: feature axis gets rotary position instead of an additive r code
        self.rope = rope
        self.ssmax = ssmax  # scalable-softmax row attention for large row counts (else plain)
        self.attn_feat = (RotaryFeatAttn(emb, heads) if rope
                          else nn.MultiheadAttention(emb, heads, batch_first=True))
        self.attn_row = (RowAttnSSMax(emb, heads) if ssmax
                         else nn.MultiheadAttention(emb, heads, batch_first=True))
        self.mlp = nn.Sequential(nn.Linear(emb, mlp), nn.GELU(), nn.Linear(mlp, emb))
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(emb) for _ in range(3))

    def forward(self, src, split, key_mask=None):
        B, R, D, E = src.shape
        s = src.reshape(B * R, D, E)
        if self.rope:
            s = self.attn_feat(s, None if key_mask is None else key_mask.reshape(B * R, D))
        elif key_mask is not None:   # MHA convention: True = may NOT attend; self stays allowed (no all-blocked row)
            km = key_mask.reshape(B * R, D)
            blocked = km[:, None, :] & ~torch.eye(D, dtype=torch.bool, device=km.device)
            s = self.attn_feat(s, s, s, attn_mask=blocked.repeat_interleave(self.attn_feat.num_heads, 0),
                               need_weights=False)[0]
        else:
            s = self.attn_feat(s, s, s)[0]
        s = s.reshape(B, R, D, E) + src
        s = self.norm1(s)
        t = s.transpose(1, 2).reshape(B * D, R, E)
        ctx = t[:, :split]
        if self.ssmax:
            left = self.attn_row(ctx, ctx)
            right = self.attn_row(t[:, split:], ctx)
        else:
            # need_weights=False: the weights were always discarded ([0]), and asking for them
            # forces materializing the O(split^2) attention matrix (bmm), which OOMs at large row
            # counts (TabICL's 1024-row regime). False routes to the fused SDPA kernel -- same
            # output, O(split) memory. No effect on the small-row paper runs.
            left = self.attn_row(ctx, ctx, ctx, need_weights=False)[0]
            right = self.attn_row(t[:, split:], ctx, ctx, need_weights=False)[0]
        t = torch.cat([left, right], 1) + t
        s = t.reshape(B, D, R, E).transpose(1, 2)
        s = self.norm2(s)
        return self.norm3(self.mlp(s) + s)


class CellPFN(nn.Module):
    def __init__(self, emb=96, heads=4, mlp=192, layers=3, n_bins=32, z_max=3.0,
                 n_reg_tokens=0, n_cls=0, r_scheme="resample", max_cols=16, ssmax=False, n_classes=0):
        super().__init__()
        # classification-only variant (2026-09-04): a dedicated class-logit head at the label cell, TabPFN-v2 style.
        # 0 = off (the bar head serves everything, the paper's configuration).
        self.n_classes = n_classes
        if n_classes:
            self.cls_head = nn.Sequential(nn.Linear(emb, mlp), nn.GELU(), nn.Linear(mlp, n_classes))
        # Issue #20: how column identity is encoded. "resample" (default) draws a
        # fresh additive r each forward; "fixed" reuses one bank (TabPFN v2);
        # "rope" puts position inside attn_feat with no additive code (v3).
        self.r_scheme = r_scheme
        if r_scheme == "fixed":
            # not learnable: isolate resample-vs-fixed. v2's learned projection
            # of the random code would be a second variable - skip it.
            self.register_buffer("fixed_r", torch.randn(1, 1, max_cols, emb))
        self.value_proj = nn.Linear(1, emb)
        self.mask_emb = nn.Parameter(torch.randn(emb) * 0.02)
        self.blocks = nn.ModuleList(
            TwoWayBlock(emb, heads, mlp, rope=(r_scheme == "rope"), ssmax=ssmax)
            for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(emb, mlp), nn.GELU(), nn.Linear(mlp, n_bins))
        # T-JEPA-style regularization rows (collapse fallback, SSOT R1): learnable
        # rows prepended to the context segment; stripped from the output
        self.n_reg = n_reg_tokens
        if n_reg_tokens:
            self.reg_rows = nn.Parameter(torch.randn(1, n_reg_tokens, 1, emb) * 0.02)
        # Issue #19: row-summary CLS columns - the TRANSPOSE of reg_rows. Feature
        # axis D -> D+n_cls; attn_feat makes them a row summary, attn_row carries
        # them across rows under the same PFN rule. No value, no r_j: they have
        # no cell to encode and no address to carry.
        self.n_cls = n_cls
        if n_cls:
            self.cls_cols = nn.Parameter(torch.randn(1, 1, n_cls, emb) * 0.02)
        edges = torch.linspace(-z_max, z_max, n_bins - 1)
        self.register_buffer("bin_edges", edges)
        centers = torch.cat([edges[:1] - 0.1, (edges[1:] + edges[:-1]) / 2, edges[-1:] + 0.1])
        self.register_buffer("bin_centers", centers)

    def sample_r(self, B, D, device, dtype=torch.float32):
        """Column-identity vectors; lifetime = one forward. JEPA passes the SAME
        r to online and target encoders (else cell addresses diverge). For
        "fixed" the bank is reused; for "rope" there is no additive code and
        encode() ignores this (position lives inside attn_feat)."""
        E = self.value_proj.out_features
        if self.r_scheme == "fixed":
            return self.fixed_r[:, :, :D].to(device=device, dtype=dtype).expand(B, -1, -1, -1)
        if self.r_scheme == "rope":
            return torch.zeros(1, 1, 1, E, device=device, dtype=dtype)  # unused
        return torch.randn(B, 1, D, E, device=device, dtype=dtype)

    def encode(self, z, input_mask, split, r=None, keep_cls=False,
               inject_mask=None, inject_emb=None, ctx_only=False, ckpt=False):
        """z (B,R,D) standardized values (anything under input_mask is ignored),
        input_mask (B,R,D) bool cells hidden from the model, split = #context rows.
        Returns cell embeddings (B,R,D,E), or (B,R,n_cls+D,E) with keep_cls -
        CLS columns first, so every existing caller is unaffected.

        ctx_only (I-JEPA context encoder): hidden cells are excluded as keys on the feature
        axis in every block, so an observed cell's embedding is a function of observed cells
        only; the outputs AT hidden cells are then meaningless and the caller (JEPA) replaces
        them with its predictor mask token. The row axis needs no change: target cells live in
        query rows, and query rows are never keys.

        Issue #22 rollout: inject_mask (B,R,D) + inject_emb (B,R,D,E) overwrite the
        input embedding at chosen cells with a given latent (instead of mask_emb) -
        the one path that lets a predicted/true latent re-enter as input.

        ckpt: gradient-checkpoint every block (non-reentrant; no-op without grad). For a second grad pass over the
        full table (JEPA ckpt_pass2: sigreg / dae targets) whose activations do not fit next to the first pass."""
        h = self.value_proj(torch.where(input_mask, torch.zeros_like(z), z).unsqueeze(-1))
        h = torch.where(input_mask.unsqueeze(-1), self.mask_emb.to(h.dtype), h)
        B, _, D, E = h.shape
        if self.r_scheme != "rope":  # rope injects position inside attn_feat
            if r is None:
                r = self.sample_r(B, D, h.device, h.dtype)
            h = h + r
        if inject_mask is not None:  # rollout: given latents win over value/mask_emb
            h = torch.where(inject_mask.unsqueeze(-1), inject_emb.to(h.dtype), h)
        if self.n_cls:
            h = torch.cat([self.cls_cols.expand(B, h.shape[1], -1, -1), h], 2)
        if self.n_reg:
            h = torch.cat([self.reg_rows.expand(B, -1, h.shape[2], -1), h], 1)
            split = split + self.n_reg
        key_mask = None
        if ctx_only:
            if self.n_reg:
                raise NotImplementedError("ctx_only with reg rows")
            key_mask = input_mask
            if self.n_cls:   # CLS columns carry no value: always visible
                key_mask = torch.cat([key_mask.new_zeros(B, key_mask.shape[1], self.n_cls), key_mask], 2)
        for blk in self.blocks:
            h = checkpoint(blk, h, split, key_mask, use_reentrant=False) if ckpt and torch.is_grad_enabled() else blk(h, split, key_mask)
        if self.n_reg:
            h = h[:, self.n_reg:]
        return h if keep_cls else h[:, :, self.n_cls:]

    def forward(self, z, input_mask, split, r=None):
        return self.head(self.encode(z, input_mask, split, r))

    def to_bins(self, z):
        return torch.bucketize(z.clamp(-self.bin_edges[-1] + 1e-4,
                                       self.bin_edges[-1] - 1e-4), self.bin_edges)

    def loss(self, logits, z_true, target_mask):
        """CE over bins at target cells (data-space objective, both baselines)."""
        lg = logits[target_mask]
        return F.cross_entropy(lg, self.to_bins(z_true[target_mask]))

    def cls_logits(self, h, y_ids, target_mask, n_classes):
        """Class logits at target cells with the table's unused classes masked out; returns (logits, labels)."""
        logits = self.cls_head(h[target_mask])
        k = n_classes.view(-1, 1, 1).expand_as(target_mask)[target_mask]
        invalid = torch.arange(logits.shape[-1], device=logits.device)[None] >= k[:, None]
        return logits.masked_fill(invalid, float("-inf")), y_ids[target_mask]

    def loss_cls(self, h, y_ids, target_mask, n_classes):
        logits, y = self.cls_logits(h, y_ids, target_mask, n_classes)
        if len(y) == 0:                        # a batch of tables without a label column: no signal, no NaN
            return h.sum() * 0.0
        return F.cross_entropy(logits, y)

    def point_pred(self, logits):
        return F.softmax(logits, -1) @ self.bin_centers
