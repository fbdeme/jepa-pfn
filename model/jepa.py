"""JEPA objective on the cell-token PFN (Phase 3, SSOT section 3).

Two objective families, selected by `mode`:

- "ema" (Phase 3 default): target encoder = EMA copy with identical attention
  rules, sees the unmasked original (stop-gradient). Targets are normalized
  with a parameter-free LayerNorm before the L2 loss.
- "sigreg" (Issue #8, LeJEPA arXiv 2511.08544 / LeWM arXiv 2603.19312): EMA-free.
  The target is the online encoder's own unmasked output through a projector, and
  collapse is prevented by an explicit distributional constraint (SIGReg) on that
  field instead of the EMA/stop-grad/LayerNorm heuristics. LeWM propagates gradients
  through all components - no stop-gradient (2603.19312 L296-300): sigreg_detach=False
  is that wiring; sigreg_detach=True (this port's historical default) detaches the target.
- "dae" (docs/prior_v2_plan.md 5.2, 2026-09-11): EMA-free. A second CellPFN (y-encoder)
  sees the full table under Gaussian noise (dae_sigma) and reconstructs the clean bins
  with its own bar head (that loss trains the y-encoder only); its hidden-cell embedding,
  detached + LayerNorm, is the latent target. The reconstruction loss lower-bounds the
  target's information at the value, so the target cannot decorrelate into noise.

A shallow MLP predictor maps online embeddings at masked positions to the
target embeddings there.

Correctness invariant: both encoders receive the SAME r_j column vectors
within a forward - column identity must match between prediction and target.

An auxiliary bar-distribution head (lambda_ppd > 0) is wired for Phase 5;
Phase 3 trains pure-latent (lambda_ppd = 0).
"""

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.pfn import CellPFN, TwoWayBlock


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (LeJEPA, arXiv 2511.08544).

    Ported from ../EEG-WM-JEPA/src/model/lewm_modules.py:20-51 (our own
    verified implementation, itself from github.com/lucas-maes/le-wm).

    Random 1-D slices of the embeddings are tested against N(0,1) by the
    Epps-Pulley characteristic-function statistic, averaged over slices:
    N * integral |phi_hat(t) - exp(-t^2/2)|^2 w(t) dt, with w = exp(-t^2/2).
    The N scaling makes the value O(1) under the null at any sample size.
    Input (T, N, E); the empirical CF is estimated over the N axis. We pass
    (1, n_cells, E) - every cell embedding in the batch is one sample.

    Deviations from the paper's Algorithm 1, both checked against its
    ablations: the grid is 17 trapezoid knots on [0,3] doubled by symmetry
    (= [-3,3]; the paper's own codebase does this, and [-3,3] ~ [-5,5] to
    within noise), and num_proj is 128 rather than the recommended 1024 -
    slice count is the least sensitive knob (resampled directions at |A|=16
    beat thousands of fixed ones), and the (N, num_proj, knots) intermediate
    would be ~1 GB at 1024 slices for our ~19k cells/batch.
    """

    def __init__(self, knots=17, num_proj=128):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class JEPA(nn.Module):
    def __init__(self, emb=96, heads=4, mlp=192, layers=3, n_bins=32,
                 n_reg_tokens=0, ema=0.996, lambda_ppd=0.0, predictor="mlp",
                 mode="ema", lambda_sig=0.05, sig_proj=128, r_invariant=False,
                 n_cls=0, cls_target=False, r_scheme="resample", shuffle_target=False, n_classes=0,
                 ctx_only=False, fresh_target_r=False, diff_target=False, head_on_pred=False, lambda_jepa=1.0,
                 dae_sigma=0.5, sigreg_detach=True, sigreg_additive=False, ckpt_pass2=False):
        super().__init__()
        self.online = CellPFN(emb, heads, mlp, layers, n_bins,
                              n_reg_tokens=n_reg_tokens, n_cls=n_cls, r_scheme=r_scheme, n_classes=n_classes)
        self.mode = mode
        # arbitrary-auxiliary control: permute the latent target across masked cells so the auxiliary
        # keeps dual's machinery/capacity but carries no cell-aligned (shared-latent) signal.
        self.shuffle_target = shuffle_target
        if mode == "ema":
            self.target = copy.deepcopy(self.online)
            for p in self.target.parameters():
                p.requires_grad_(False)
        elif mode == "sigreg":
            self.target = None
            # SIGReg needs a space where N(0,I) is reachable: the encoder's own
            # output ends in LayerNorm (pfn.TwoWayBlock.norm3), i.e. lives on a
            # sphere. LeWM/EEG-WM-JEPA put the projector here for this reason.
            self.projector = nn.Sequential(nn.Linear(emb, mlp), nn.GELU(),
                                           nn.Linear(mlp, emb))
            self.sigreg = SIGReg(num_proj=sig_proj)
        elif mode == "dae":
            self.target = None
            # y-encoder: a second CellPFN of the online's shape (own weights, own bar head), trained in forward() by its
            # reconstruction loss only. Not `target`: ema_update / encoder() / the audits read `target` as the EMA copy.
            self.yenc = CellPFN(emb, heads, mlp, layers, n_bins, n_reg_tokens=n_reg_tokens, n_cls=n_cls, r_scheme=r_scheme)
        else:
            raise ValueError(mode)
        # capacity sweep (I-JEPA Table 12: deeper predictor -> better targets;
        # a weak predictor pressures targets simpler = rank-drift suspect)
        if predictor == "mlp":
            self.pred_blocks = None
            self.predictor = nn.Sequential(nn.Linear(emb, mlp), nn.GELU(),
                                           nn.Linear(mlp, emb))
        elif predictor.startswith("twoway-"):
            n = int(predictor.split("-")[1])
            self.pred_blocks = nn.ModuleList(
                TwoWayBlock(emb, heads, mlp, rope=(r_scheme == "rope")) for _ in range(n))
            self.pred_out = nn.Linear(emb, emb)
        else:
            raise ValueError(predictor)
        self.ema = ema
        self.lambda_ppd = lambda_ppd
        self.lambda_sig = lambda_sig
        self.dae_sigma = dae_sigma
        self.sigreg_detach = sigreg_detach
        self.sigreg_additive = sigreg_additive
        # 2026-09-11 peak smoke: the sigreg / dae second grad pass over the full table OOMs the 96 GB box next to the
        # first pass (97 GB at 64 x 1024 x 4). ckpt_pass2 gradient-checkpoints that pass (pfn.encode ckpt + the SIGReg
        # statistic, whose (N, M, knots) intermediates are recomputed in backward); the first pass is untouched.
        self.ckpt_pass2 = ckpt_pass2
        # Issue #15: the latent target carries the column address r (36% of its
        # variance, eval/address_leak.py), which the data-space target does not.
        # Projecting r_j out of both sides makes the address unscoreable.
        # NOTE: r reaches the output through 3 non-linear blocks, so this only
        # removes the component still aligned with r_j - re-measure the residual
        # share with eval/address_leak.py rather than assuming it is zero.
        self.r_invariant = r_invariant
        # Issue #19: move the latent target off the cell and onto the row summary,
        # where there is something to throw away (a cell latent is a near-invertible
        # code of its value, so predicting it ~ predicting the value).
        self.cls_target = cls_target
        if cls_target and not n_cls:
            raise ValueError("cls_target needs n_cls > 0")
        if cls_target and r_invariant:
            raise ValueError("CLS columns carry no r_j - r_invariant is meaningless")
        if r_invariant and r_scheme != "resample":
            raise ValueError("r_invariant projects out an additive r; fixed/rope have none")
        # I-JEPA split (2026-09-07): the online encoder sees observed cells only (pfn.encode ctx_only), and the
        # PREDICTOR fills the hidden cells from a learned mask token (column position via rope inside its
        # two-way blocks). Without this, the encoder itself in-fills from mask_emb and the predictor is a
        # read-out - the latent arm then differs from data-space training only in its target.
        self.ctx_only = ctx_only
        if ctx_only:
            if self.pred_blocks is None:
                raise ValueError("ctx_only needs a twoway predictor (an MLP at a mask token sees nothing)")
            if cls_target:
                raise ValueError("ctx_only scores hidden cells, not CLS")
            self.pred_mask = nn.Parameter(torch.randn(emb) * 0.02)
        # Address decorrelation (2026-09-08): the target encoder draws its OWN column codes, so the address
        # component of the target is not reproducible from the online side (Issue #15 shortcut closed at the
        # source instead of projected out). TabPFN-v2-style: codes are redrawn per forward anyway; here also per
        # branch. Only meaningful with an additive resampled code.
        self.fresh_target_r = fresh_target_r
        # Difference target (2026-09-08): target = EMA(full table) - EMA(the predictor's own view: hidden cells masked),
        # read at the hidden cells. Everything the visible cells determine (address, column/row fingerprints) sits in
        # both terms and cancels; what is left is what revealing the hidden values changes in the latent. EMA only.
        self.diff_target = diff_target
        if diff_target and mode != "ema":
            raise ValueError("diff_target needs mode=ema (a detached online target would be the same encoder twice)")
        # Value head on the PREDICTOR's output (2026-09-08, "table MAE"): the same x-encoder / predictor split as the
        # I-JEPA arm, but the hidden cell is scored in data space (bar CE) instead of against a latent target.
        # lambda_jepa 0 + lambda_ppd > 0 = the data-space control of that split; both > 0 = dual on the split.
        self.head_on_pred = head_on_pred
        self.lambda_jepa = lambda_jepa
        if head_on_pred and (self.pred_blocks is None or lambda_ppd <= 0):
            raise ValueError("head_on_pred needs a twoway predictor and lambda_ppd > 0")
        if fresh_target_r:
            if r_scheme != "resample":
                raise ValueError("fresh_target_r needs r_scheme=resample (fixed/rope have no per-forward draw)")
            if r_invariant:
                raise ValueError("fresh_target_r and r_invariant are two answers to the same shortcut; pick one")

    @staticmethod
    def _drop_r(pred, tgt, r, target_mask, R):
        u = F.normalize(r.expand(-1, R, -1, -1)[target_mask], dim=-1)
        rm = lambda v: v - (v * u).sum(-1, keepdim=True) * u
        return rm(pred), rm(tgt)

    @staticmethod
    def _cls_mask(target_mask, n_cls):
        """Score the CLS of every row that lost a cell: "what do the hidden cells
        contribute to this row's summary?". The conditioning stays inside the row,
        so this is not the trivial task - masking whole rows would be, since rows
        are i.i.d. and the optimum is then the average row."""
        B, R, D = target_mask.shape
        tm = target_mask.new_zeros(B, R, n_cls + D)
        tm[:, :, :n_cls] = target_mask.any(-1, keepdim=True)
        return tm

    def _encode_online(self, z, input_mask, split, r, keep_cls):
        """(predictor input, encoder field). With ctx_only the predictor input has the hidden cells swapped for the mask
        token, while the encoder field keeps the encoder's own in-fill there - what a head_on_pred=False value head reads
        (headenc, docs/prior_v2_plan.md 5.2: PFN path, hidden cells invisible to each other). Without ctx_only both are h."""
        h = self.online.encode(z, input_mask, split, r=r, keep_cls=keep_cls, ctx_only=self.ctx_only)
        if self.ctx_only:   # hidden cells enter the predictor as the mask token, not as the encoder's output there
            return torch.where(input_mask.unsqueeze(-1), self.pred_mask.to(h.dtype), h), h
        return h, h

    def _predict(self, h, target_mask, split, full=False):
        """Predictor output at the target cells, or the whole field (B,R,D,E) with full=True."""
        if self.pred_blocks is None:
            out = self.predictor(h)
        else:
            for blk in self.pred_blocks:
                h = blk(h, split)
            out = self.pred_out(h)
        return out if full else out[target_mask]

    @torch.no_grad()
    def value_logits(self, z, input_mask, split):
        """Bar-head logits at every cell (B,R,D,n_bins): from the predictor's field when head_on_pred, else from the
        online encoder (the dual arm's head). One entrance for validate / read-outs."""
        B, _, D = z.shape
        r = self.online.sample_r(B, D, z.device)
        k = self.online.n_cls
        h, h_enc = self._encode_online(z, input_mask, split, r, False)
        if self.head_on_pred:
            return self.online.head(self._predict(h, None, split, full=True))
        return self.online.head(h_enc)

    def forward(self, z, input_mask, target_mask, split, y_ids=None, n_classes=None):
        B, _, D = z.shape
        r = self.online.sample_r(B, D, z.device)
        k = self.online.n_cls
        keep = self.cls_target
        h, h_enc = self._encode_online(z, input_mask, split, r, keep)
        tm = self._cls_mask(target_mask, k) if keep else target_mask
        if not tm.any():                       # no target cell in this batch (y_target on label-free tables): zero loss, no NaN
            zero = h.sum() * 0.0
            return zero, zero.detach(), h.new_zeros(0, h.shape[-1]), h.new_zeros(0, h.shape[-1])
        pred_field = self._predict(h, tm, split, full=True) if self.head_on_pred else None
        pred = pred_field[tm] if self.head_on_pred else self._predict(h, tm, split)
        no_mask = torch.zeros_like(input_mask)
        r_t = self.online.sample_r(B, D, z.device) if self.fresh_target_r else r
        if self.mode == "ema":
            with torch.no_grad():
                z_tgt = self.target_latent(z, input_mask, split, r_t, keep)[tm]
                z_tgt = F.layer_norm(z_tgt, z_tgt.shape[-1:])
                if self.shuffle_target:   # break pred<->target cell alignment (arbitrary auxiliary)
                    z_tgt = z_tgt[torch.randperm(z_tgt.shape[0], device=z_tgt.device)]
            if self.r_invariant:
                pred, z_tgt = self._drop_r(pred, z_tgt, r, target_mask, z.shape[1])
            loss_jepa = F.mse_loss(pred, z_tgt)
            loss = self.lambda_jepa * loss_jepa
        elif self.mode == "sigreg":
            # EMA-free: the same encoder on the full view through the projector, no target LayerNorm - SIGReg on that
            # projected field is what keeps it from collapsing. LeWM (arXiv 2603.19312, L296-300) keeps the asymmetric
            # predictor and propagates gradients through every component, no stop-gradient: with sigreg_detach=False the
            # MSE pulls target and prediction toward each other and SIGReg forbids the constant solution (the masked-
            # prediction ablation of arXiv 2608.24093 ranks a detached target worst). sigreg_detach=True is this port's
            # historical default (arms before 2026-09-11). Loss: LeJEPA Eq. 9 convex combination, or LeWM's additive
            # pred + lambda * reg with sigreg_additive. Census: docs/sigreg_census/verdict.md.
            full = self.projector(self.online.encode(z, no_mask, split, r=r_t,
                                                     keep_cls=keep, ckpt=self.ckpt_pass2))
            z_tgt = full[tm].detach() if self.sigreg_detach else full[tm]
            if self.r_invariant:
                pred, z_tgt = self._drop_r(pred, z_tgt, r, target_mask, z.shape[1])
            loss_jepa = F.mse_loss(pred, z_tgt)
            flat = full.reshape(1, -1, full.shape[-1])
            sig = checkpoint(self.sigreg, flat, use_reentrant=False) if self.ckpt_pass2 else self.sigreg(flat)
            loss = loss_jepa + self.lambda_sig * sig if self.sigreg_additive else (1 - self.lambda_sig) * loss_jepa + self.lambda_sig * sig
        else:
            # dae: the y-encoder sees the full table under fresh Gaussian noise; its hidden-cell embedding, detached +
            # LayerNorm (as the EMA path), is the target. Reconstruction = clean bins at every cell that has a value
            # (MCAR-missing cells are hidden and never targets) from the noised table: bin width .2 at z_max 3 / 32 bins,
            # so sigma .5 cannot be undone from the cell alone and row/column context is needed. Trains the y-encoder only.
            hy = self.yenc.encode(z + self.dae_sigma * torch.randn_like(z), no_mask, split, r=r_t, keep_cls=keep, ckpt=self.ckpt_pass2)
            z_tgt = F.layer_norm(hy[tm].detach(), hy.shape[-1:])
            if self.r_invariant:
                pred, z_tgt = self._drop_r(pred, z_tgt, r, target_mask, z.shape[1])
            loss_jepa = F.mse_loss(pred, z_tgt)
            valid = ~input_mask | target_mask
            hyc = hy[:, :, k:] if keep else hy
            recon = F.cross_entropy(self.yenc.head(hyc)[valid], self.yenc.to_bins(z[valid]))
            loss = self.lambda_jepa * loss_jepa + recon
        if self.lambda_ppd > 0:
            src = pred_field if self.head_on_pred else h_enc
            hc = src[:, :, k:] if keep else src
            if y_ids is not None:   # classification-only variant: class-CE at the label cell instead of the bar CE
                loss = loss + self.lambda_ppd * self.online.loss_cls(hc, y_ids, target_mask, n_classes)
            else:
                logits = self.online.head(hc)[target_mask]
                loss = loss + self.lambda_ppd * F.cross_entropy(
                    logits, self.online.to_bins(z[target_mask]))
        return loss, loss_jepa.detach(), pred.detach(), z_tgt.detach()

    @torch.no_grad()
    def target_latent(self, z, input_mask, split, r, keep_cls=False):
        """The EMA target field (B,R,D,E) before LayerNorm: the full-table encoding, minus the masked-view encoding
        when diff_target (same weights, same r, same rows; the masked view in-fills with mask_emb as any encoder call)."""
        full = self.target.encode(z, torch.zeros_like(input_mask), split, r=r, keep_cls=keep_cls)
        if self.diff_target:
            full = full - self.target.encode(z, input_mask, split, r=r, keep_cls=keep_cls)
        return full

    @torch.no_grad()
    def ema_update(self):
        if self.target is None:  # sigreg mode has no EMA copy
            return
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.mul_(self.ema).add_(po, alpha=1 - self.ema)

    def encoder(self):
        """The representation under evaluation (eval/truth_eval.py). The EMA
        arm's is its target copy; the SIGReg arm has only the online encoder."""
        return self.online if self.target is None else self.target

    @torch.no_grad()
    def predict_latent(self, z, input_mask, target_mask, split):
        """Predicted latents at masked cells (probe features for eval)."""
        B, _, D = z.shape
        r = self.online.sample_r(B, D, z.device)
        h, _ = self._encode_online(z, input_mask, split, r, False)
        return self._predict(h, target_mask, split)


class ValueModel(nn.Module):
    """CellPFN-shaped read-out of a JEPA arm on the I-JEPA split (ctx_only): head_on_pred arms read x-encoder -> predictor
    -> bar head, headenc arms (ctx_only, head on the encoder) read x-encoder (ctx_only) -> bar head - the field forward()
    scores in both cases. This is what the real-data suite (realdata_bench / suite_bench) and the fixed-R read-out
    (rows_control / real_readout) call at inference; the y-encoder is training-only. encode() returns the field the head
    reads so head(encode(...)) is the same path as forward()."""

    def __init__(self, jepa):
        super().__init__()
        if not (jepa.head_on_pred or jepa.ctx_only):
            raise ValueError("ValueModel is for head_on_pred / ctx_only arms; other arms read their online encoder directly")
        self.jepa = jepa
        self.n_classes, self.n_cls = 0, 0          # the field carries no CLS columns (encode keep_cls=False)

    @property
    def head(self): return self.jepa.online.head
    @property
    def bin_centers(self): return self.jepa.online.bin_centers
    @property
    def bin_edges(self): return self.jepa.online.bin_edges
    def point_pred(self, logits): return self.jepa.online.point_pred(logits)
    def to_bins(self, z): return self.jepa.online.to_bins(z)

    @torch.no_grad()
    def encode(self, z, input_mask, split, r=None, keep_cls=False):
        B, _, D = z.shape
        r = self.jepa.online.sample_r(B, D, z.device) if r is None else r
        h, h_enc = self.jepa._encode_online(z, input_mask, split, r, False)
        return self.jepa._predict(h, None, split, full=True) if self.jepa.head_on_pred else h_enc

    @torch.no_grad()
    def forward(self, z, input_mask, split, r=None):
        return self.head(self.encode(z, input_mask, split, r))


@torch.no_grad()
def within_table_var_frac(v, tids):
    """Share of the total variance of v (n, E) that lies WITHIN tables (1 - between-table share), tids (n,) = table id per row.
    A target that is a per-table constant scores ~0 while its batch-level dim_std can still look healthy
    (eval/jepa_target_audit.py found exactly that mode); computed on all rows, no subsampling."""
    v = v.float().cpu()
    c = v - v.mean(0)
    if c.pow(2).mean() < 1e-8:      # batch-collapsed already (per-dim std < 1e-4): the ratio is float noise, undefined
        return None
    tot = c.pow(2).sum()
    _, inv = torch.unique(tids.cpu(), return_inverse=True)
    n_t = int(inv.max()) + 1
    cnt = torch.zeros(n_t).index_add_(0, inv, torch.ones(len(inv)))
    means = torch.zeros(n_t, v.shape[1]).index_add_(0, inv, c) / cnt[:, None]
    between = (means.pow(2) * cnt[:, None]).sum()
    return round(float(1 - between / tot), 4)


@torch.no_grad()
def collapse_stats(v, max_n=512, tids=None):
    """SSOT R1 monitoring on a (n, E) batch of representations:
    per-dim std (mean/min), effective rank (entropy of singular values),
    mean pairwise cosine similarity; with tids (table id per row) also within_table_var_frac."""
    # .cpu(): monitoring only, <=512 rows - and concurrent runs on one GPU
    # exhaust cusolver handles (CUSOLVER_STATUS_INTERNAL_ERROR on svdvals)
    if len(v) < 2:
        out = dict(dim_std=0.0, dim_std_min=0.0, erank=0.0, cos=0.0)
        if tids is not None:
            out["within_table_var_frac"] = None
        return out
    extra = {} if tids is None else dict(within_table_var_frac=within_table_var_frac(v, tids))
    v = v[torch.randperm(len(v))[:max_n]].float().cpu()
    c = v - v.mean(0)
    std = c.std(0)
    s = torch.linalg.svdvals(c)
    p = s / (s.sum() + 1e-12)
    erank = torch.exp(-(p * (p + 1e-12).log()).sum())
    vn = F.normalize(v, dim=1)
    n = len(vn)
    cos = (vn @ vn.T).sum().sub(n).div(n * (n - 1) + 1e-12)
    return dict(dim_std=round(std.mean().item(), 4),
                dim_std_min=round(std.min().item(), 4),
                erank=round(erank.item(), 1),
                cos=round(cos.item(), 4), **extra)
