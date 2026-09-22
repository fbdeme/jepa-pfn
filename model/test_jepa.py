"""Self-checks for the JEPA arms, SIGReg port in particular (Issue #8).

Run: uv run python -m model.test_jepa
"""

import torch

from model.jepa import JEPA, SIGReg, collapse_stats, within_table_var_frac


def _batch(B=4, R=12, D=5, split=8):
    z = torch.randn(B, R, D)
    input_mask = torch.zeros(B, R, D, dtype=torch.bool)
    input_mask[:, split:, -1] = True  # hide the last column of the query rows
    return z, input_mask, input_mask.clone(), split


def test_sigreg_separates_gaussian_from_collapse():
    """The statistic must be near its null floor for N(0,I) and large for
    degenerate embeddings - otherwise the port is not testing what we think."""
    torch.manual_seed(0)
    sig = SIGReg(num_proj=256)
    n, e = 4096, 96
    gauss = sig(torch.randn(1, n, e)).item()
    collapsed = sig(torch.randn(1, 1, e).expand(1, n, e)).item()   # all cells equal
    rank1 = sig(torch.randn(1, n, 1) * torch.randn(1, 1, e)).item()  # 1-D subspace
    scaled = sig(torch.randn(1, n, e) * 3.0).item()  # right shape, wrong scale
    assert gauss < 5.0, gauss  # O(1) under the null despite n=4096
    assert collapsed > 100 * gauss, (collapsed, gauss)
    assert rank1 > 10 * gauss, (rank1, gauss)
    assert scaled > 10 * gauss, (scaled, gauss)
    print(f"  sigreg: gauss={gauss:.2f} collapsed={collapsed:.0f} "
          f"rank1={rank1:.0f} scaled={scaled:.0f}")


def test_sigreg_mode_trains_without_ema():
    torch.manual_seed(0)
    m = JEPA(mode="sigreg", predictor="twoway-2")
    assert m.target is None
    assert m.encoder() is m.online
    loss, lj, pred, tgt = m(*_batch())
    loss.backward()
    grads = {n for n, p in m.named_parameters() if p.grad is not None
             and p.grad.abs().sum() > 0}
    # SIGReg must reach the encoder through the projector; the MSE target is
    # detached, so without SIGReg the unmasked branch would contribute nothing.
    assert any(n.startswith("projector") for n in grads)
    assert any(n.startswith("online.blocks") for n in grads)
    m.ema_update()  # no-op, must not raise
    print(f"  sigreg mode: loss={loss.item():.3f} pred_loss={lj.item():.3f} "
          f"n_grad_tensors={len(grads)}")


def test_ema_mode_unchanged():
    torch.manual_seed(0)
    m = JEPA(mode="ema")
    before = m.target.value_proj.weight.clone()
    loss, lj, pred, tgt = m(*_batch())
    assert torch.allclose(loss, lj)  # no sigreg term in this arm
    loss.backward()
    for p in m.online.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    m.ema_update()
    assert not torch.allclose(m.target.value_proj.weight, before)
    assert m.encoder() is m.target
    print(f"  ema mode: loss={loss.item():.3f}, target tracks online")


def test_r_invariant_kills_the_address_component():
    """Issue #15: with r_invariant, nothing on either side may still lie along
    r_j, so a target made purely of addresses becomes unscoreable (loss = 0)."""
    torch.manual_seed(0)
    m = JEPA(mode="ema", r_invariant=True)
    B, R, D, E = 4, 12, 5, 96
    r = torch.randn(B, 1, D, E)
    tm = torch.zeros(B, R, D, dtype=torch.bool)
    tm[:, 8:, :] = True
    u = torch.nn.functional.normalize(r.expand(-1, R, -1, -1)[tm], dim=-1)
    # pure-address prediction vs pure-address target, different magnitudes:
    # only the r direction distinguishes them, and it is projected away
    p, t = m._drop_r(3.0 * u, -2.0 * u, r, tm, R)
    assert p.abs().max() < 1e-5, p.abs().max()
    assert t.abs().max() < 1e-5, t.abs().max()
    # content orthogonal to r must survive untouched
    c = torch.randn_like(u)
    c = c - (c * u).sum(-1, keepdim=True) * u
    kept, _ = m._drop_r(c, c, r, tm, R)
    assert torch.allclose(kept, c, atol=1e-5)
    print("  r_invariant: address component removed, content preserved")


def test_cls_columns_are_additive_and_targetable():
    """Issue #19: CLS columns extend the feature axis without disturbing the cell
    path (every existing caller and eval must see the same (B,R,D,E)), and the
    CLS latent target must score one vector per CLS per row that lost a cell."""
    torch.manual_seed(0)
    k = 2
    m = JEPA(n_cls=k, cls_target=True, predictor="twoway-2")
    z, im, tm, split = _batch()
    B, R, D = z.shape

    enc = m.online
    h_cells = enc.encode(z, im, split, r=enc.sample_r(B, D, z.device))
    assert h_cells.shape == (B, R, D, 96), h_cells.shape  # cell path unchanged

    loss, lj, pred, tgt = m(z, im, tm, split)
    n_rows = int(tm.any(-1).sum())
    assert pred.shape == (n_rows * k, 96), (pred.shape, n_rows)
    assert tgt.shape == pred.shape
    # the summary must actually respond to the row's cells, or the target is a
    # constant and the loss is trivially zero
    assert tgt.std(0).mean() > 1e-3, tgt.std(0).mean()

    loss.backward()
    assert m.online.cls_cols.grad.abs().sum() > 0
    for bad in (dict(cls_target=True), dict(n_cls=k, cls_target=True,
                                            r_invariant=True)):
        try:
            JEPA(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    print(f"  cls: cell path {tuple(h_cells.shape)} intact, "
          f"{n_rows} masked rows x k={k} targets, loss={loss.item():.3f}")


def test_address_schemes_train_and_stay_permutation_equivariant():
    """Issue #20: fixed/rope arms must train; rope must NOT inject an additive
    code (a column-permuted table, un-permuted, comes back exactly - because
    position lives only in attn_feat's Q/K, not the residual stream)."""
    from model.pfn import CellPFN
    z, im, tm, split = _batch()
    B, R, D = z.shape
    for scheme in ("fixed", "rope"):
        m = JEPA(mode="ema", predictor="twoway-2", r_scheme=scheme)
        loss, lj, pred, tgt = m(z, im, tm, split)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in m.online.blocks.parameters()), scheme
        # fixed bank is a buffer, never a grad-carrying parameter
        assert not any(n == "online.fixed_r"
                       for n, p in m.named_parameters()), scheme
    # rope's defining property: the additive r is not in the residual stream at
    # all (position lives inside attn_feat), so encode IGNORES r - two different
    # r draws give byte-identical output.
    encr = CellPFN(r_scheme="rope").eval()
    with torch.no_grad():
        a = encr.encode(z, im, split, r=torch.randn(B, 1, D, 96))
        b = encr.encode(z, im, split, r=torch.randn(B, 1, D, 96) * 5)
    assert torch.allclose(a, b), (a - b).abs().max()
    # resample DOES ride r: a different draw moves the output (that is the tax)
    enc0 = CellPFN(r_scheme="resample").eval()
    with torch.no_grad():
        c = enc0.encode(z, im, split, r=torch.randn(B, 1, D, 96))
        d = enc0.encode(z, im, split, r=torch.randn(B, 1, D, 96) * 5)
    assert not torch.allclose(c, d)
    try:
        JEPA(r_invariant=True, r_scheme="fixed")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for r_invariant + fixed")
    print("  address schemes: fixed/rope train, rope ignores additive r, "
          "resample rides it")


if __name__ == "__main__":
    for fn in [test_sigreg_separates_gaussian_from_collapse,
               test_sigreg_mode_trains_without_ema,
               test_ema_mode_unchanged,
               test_r_invariant_kills_the_address_component,
               test_cls_columns_are_additive_and_targetable,
               test_address_schemes_train_and_stay_permutation_equivariant]:
        print(fn.__name__)
        fn()
    print("ok")


def test_within_table_var_frac():
    """C1: per-table-constant -> 0, iid rows -> ~1, batch-collapsed -> None (undefined), and the key
    appears in collapse_stats only when table ids are passed."""
    torch.manual_seed(0)
    tids = torch.arange(8).repeat_interleave(16)                       # 8 tables x 16 rows
    per_table = torch.randn(8, 32).repeat_interleave(16, 0)
    assert within_table_var_frac(per_table, tids) == 0.0
    iid = torch.randn(128, 32)
    assert within_table_var_frac(iid, tids) > 0.9                      # between share ~ (T-1)/n
    mixed = per_table + 0.1 * torch.randn(128, 32)                     # mostly per-table: small but > 0
    assert 0.0 < within_table_var_frac(mixed, tids) < 0.1
    assert within_table_var_frac(torch.ones(128, 32), tids) is None
    assert within_table_var_frac(torch.ones(128, 32) + 1e-6 * torch.randn(128, 32), tids) is None
    s = collapse_stats(iid, tids=tids)
    assert s["within_table_var_frac"] == within_table_var_frac(iid, tids)
    assert "within_table_var_frac" not in collapse_stats(iid)
    assert collapse_stats(iid[:1], tids=tids[:1])["within_table_var_frac"] is None
    print(f"  within_table_var_frac: const={within_table_var_frac(per_table, tids)} iid={within_table_var_frac(iid, tids)} mixed={within_table_var_frac(mixed, tids)}")


def test_ctx_only_hides_the_mask_token_from_observed_cells():
    """I-JEPA split (ctx_only): an observed cell's embedding must not depend on the encoder's mask token (the only
    thing a hidden cell contributes), the default in-filling encoder must (so the check can fire), the predictor's
    own token must carry the loss, an all-hidden query row must not NaN, and the ctor guards must fire."""
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True)   # n_cls: the CLS pad path is exercised
    obs = lambda ctx: m.online.encode(z, im, split, ctx_only=ctx)[~im]
    a0, b0 = obs(True), obs(False)
    with torch.no_grad():
        m.online.mask_emb.add_(1.0)
    assert torch.allclose(a0, obs(True), atol=1e-6)        # ctx_only: mask token cannot reach observed cells
    assert not torch.allclose(b0, obs(False), atol=1e-3)   # in-filling encoder: it can (the assertion above is not vacuous)
    loss, lj, pred, tgt = m(z, im, tm, split)
    loss.backward()
    g = m.online.mask_emb.grad
    assert g is None or g.abs().max() == 0                 # dead weight under ctx_only
    assert m.pred_mask.grad is not None and m.pred_mask.grad.abs().max() > 0
    assert torch.isfinite(loss) and pred.shape == tgt.shape == (int(tm.sum()), 96)
    im2 = im.clone(); im2[0, split:, :] = True             # a query row with every cell hidden: self-key keeps softmax finite
    assert torch.isfinite(m(z, im2, im2.clone(), split)[0])
    for bad in (dict(predictor="mlp", r_scheme="rope"),
                dict(predictor="twoway-2", r_scheme="rope", n_cls=2, cls_target=True)):
        try:
            JEPA(ctx_only=True, **bad)
            raise AssertionError(f"guard did not fire: {bad}")
        except ValueError:
            pass
    print(f"  ctx_only: observed cells blind to mask_emb, loss={loss.item():.3f}")


def test_fresh_target_r_and_ctx_only_on_the_additive_scheme():
    """Experiment A (2026-09-08): TabPFN-v2-style additive resampled codes, no CLS, I-JEPA split, and the target
    encoder drawing its OWN codes. Checks: the MHA key-mask path hides the mask token from observed cells (and can
    fire), the target branch draws a second r exactly when asked, an all-hidden row stays finite, guards fire."""
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="resample", n_cls=0, ctx_only=True, fresh_target_r=True)
    r = m.online.sample_r(z.shape[0], z.shape[2], z.device)
    obs = lambda ctx: m.online.encode(z, im, split, r=r, ctx_only=ctx)[~im]
    a0, b0 = obs(True), obs(False)
    with torch.no_grad():
        m.online.mask_emb.add_(1.0)
    assert torch.allclose(a0, obs(True), atol=1e-6) and not torch.allclose(b0, obs(False), atol=1e-3)
    calls = []
    orig = m.online.sample_r
    m.online.sample_r = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    loss = m(z, im, tm, split)[0]
    assert len(calls) == 2 and torch.isfinite(loss)            # online r + the target's own r
    m.fresh_target_r = False
    calls.clear()
    m(z, im, tm, split)
    assert len(calls) == 1                                     # shared r: one draw
    m.online.sample_r = orig
    im2 = im.clone(); im2[0, split:, :] = True
    assert torch.isfinite(m(z, im2, im2.clone(), split)[0])
    for bad in (dict(r_scheme="rope", fresh_target_r=True), dict(r_scheme="fixed", fresh_target_r=True),
                dict(r_scheme="resample", fresh_target_r=True, r_invariant=True)):
        try:
            JEPA(predictor="twoway-2", **bad)
            raise AssertionError(f"guard did not fire: {bad}")
        except ValueError:
            pass
    print("  fresh_target_r + ctx_only (resample, MHA key mask) ok")


def test_diff_target_is_full_minus_masked_view():
    """diff_target: the latent target at a hidden cell is LN(EMA(full) - EMA(masked view)); the arithmetic is checked
    against the two encoder calls, the loss stays finite, and the guard fires outside EMA mode."""
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, diff_target=True)
    r = m.online.sample_r(z.shape[0], z.shape[2], z.device)
    want = m.target.encode(z, torch.zeros_like(im), split, r=r) - m.target.encode(z, im, split, r=r)
    got = m.target_latent(z, im, split, r)
    assert torch.allclose(want, got, atol=1e-6)
    loss, lj, pred, tgt = m(z, im, tm, split)
    assert torch.isfinite(loss) and tgt.shape[0] == int(tm.sum())
    assert not torch.allclose(tgt, torch.nn.functional.layer_norm(m.target.encode(z, torch.zeros_like(im), split, r=r)[tm], (96,)), atol=1e-3)
    try:
        JEPA(mode="sigreg", predictor="twoway-2", r_scheme="rope", diff_target=True)
        raise AssertionError("guard did not fire")
    except ValueError:
        pass
    print("  diff_target ok")


def test_head_on_pred_is_a_data_space_loss_through_the_predictor():
    """Table-MAE arm: ctx_only x-encoder + predictor, bar head on the predictor field, no latent term
    (lambda_jepa 0). The loss must equal the bar CE at the hidden cells, the gradient must reach the predictor and
    the encoder, value_logits must come from the predictor path, and the guards must fire."""
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, lambda_ppd=1.0, lambda_jepa=0.0, head_on_pred=True)
    loss, lj, pred, tgt = m(z, im, tm, split)
    r = m.online.sample_r(z.shape[0], z.shape[2], z.device)
    # rope: r is unused, so the forward is reproducible with any r -> recompute the CE by hand
    field = m._predict(m._encode_online(z, im, split, r, False)[0], None, split, full=True)
    ce = torch.nn.functional.cross_entropy(m.online.head(field)[tm], m.online.to_bins(z[tm]))
    assert torch.allclose(loss, ce, atol=1e-5) and lj.item() > 0          # loss is the CE alone; the jepa term is only logged
    loss.backward()
    assert m.pred_blocks[0].mlp[0].weight.grad.abs().max() > 0 and m.online.value_proj.weight.grad.abs().max() > 0
    lg = m.value_logits(z, im, split)
    assert lg.shape == (*z.shape, 32) and torch.allclose(lg[tm], m.online.head(field)[tm], atol=1e-5)
    for bad in (dict(predictor="mlp", lambda_ppd=1.0), dict(predictor="twoway-2", lambda_ppd=0.0)):
        try:
            JEPA(r_scheme="rope", ctx_only=(bad["predictor"] != "mlp"), head_on_pred=True, **bad)
            raise AssertionError(f"guard did not fire: {bad}")
        except ValueError:
            pass
    print("  head_on_pred (table MAE on the I-JEPA split) ok")


def test_dual_on_the_split_has_both_terms():
    """x-encoder (ctx_only) -> predictor -> [latent loss vs the y-encoder's diff target] + [bar head -> values]."""
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, diff_target=True, head_on_pred=True,
             lambda_ppd=1.0, lambda_jepa=1.0)
    loss, lj, pred, tgt = m(z, im, tm, split)
    r = m.online.sample_r(z.shape[0], z.shape[2], z.device)
    field = m._predict(m._encode_online(z, im, split, r, False)[0], None, split, full=True)
    ce = torch.nn.functional.cross_entropy(m.online.head(field)[tm], m.online.to_bins(z[tm]))
    assert torch.allclose(loss, lj + ce, atol=1e-5)            # both terms, unit weights
    assert torch.allclose(pred, field[tm], atol=1e-6)          # the latent loss reads the same field the head reads
    loss.backward(); assert m.pred_mask.grad.abs().max() > 0
    print("  dual on the split (diff target + value head on the predictor) ok")


def test_value_model_reads_the_head_through_the_predictor():
    """ValueModel (inference wrapper of a head_on_pred arm): forward == value_logits == head(encode()), CellPFN-shaped
    attributes present, refused for arms whose head is not on the predictor."""
    from model.jepa import ValueModel
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, diff_target=True, head_on_pred=True, lambda_ppd=1.0).eval()
    v = ValueModel(m)
    a, b, c = v(z, im, split), m.value_logits(z, im, split), v.head(v.encode(z, im, split))
    assert a.shape == (*z.shape, 32) and torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)
    assert v.point_pred(a).shape == z.shape and v.n_classes == 0 and v.n_cls == 0 and v.bin_centers.shape == (32,)
    try:
        ValueModel(JEPA(predictor="twoway-2", r_scheme="rope", lambda_ppd=0.1)); raise AssertionError("guard did not fire")
    except ValueError:
        pass
    print("  ValueModel ok")


def test_headenc_value_head_reads_the_encoder_infill():
    """headenc arm (docs/prior_v2_plan.md 5.2): ctx_only x-encoder, value head on the ENCODER's in-fill at the hidden
    cells - not on the predictor and not on the mask token. A hidden cell's own value must not reach the head (it is
    masked), an observed cell's value must, two hidden cells must not share one logit vector (the pred_mask constant
    would), value_logits / ValueModel / the forward's CE read the same field, and the predictor input still carries the
    mask token while the latent term still reaches it."""
    from model.jepa import ValueModel
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, diff_target=True, lambda_ppd=1.0, lambda_jepa=1.0).eval()
    lg = m.value_logits(z, im, split)
    z2 = z.clone(); z2[im] += 3.0                                             # hidden values only
    assert torch.allclose(m.value_logits(z2, im, split), lg, atol=1e-6)
    z3 = z.clone(); z3[:, 0, 0] += 3.0                                        # one observed context cell
    assert not torch.allclose(m.value_logits(z3, im, split)[tm], lg[tm], atol=1e-4)
    hid = lg[tm]
    assert (hid - hid[0]).abs().max() > 1e-4                                  # not one constant vector
    r = m.online.sample_r(z.shape[0], z.shape[2], z.device)
    h_in, h_enc = m._encode_online(z, im, split, r, False)
    assert torch.allclose(h_in[im], m.pred_mask.expand(int(im.sum()), -1)) and not torch.allclose(h_enc[im], h_in[im])
    assert torch.allclose(m.online.head(h_enc), lg, atol=1e-6) and torch.allclose(ValueModel(m)(z, im, split), lg, atol=1e-6)
    loss, lj, pred, tgt = m(z, im, tm, split)
    ce = torch.nn.functional.cross_entropy(m.online.head(h_enc)[tm], m.online.to_bins(z[tm]))
    assert torch.allclose(loss, lj + ce, atol=1e-5)
    loss.backward(); assert m.online.value_proj.weight.grad.abs().max() > 0 and m.pred_mask.grad.abs().max() > 0
    print("  headenc (value head on the ctx_only encoder's in-fill) ok")


def test_dae_mode_targets_come_from_a_denoising_y_encoder():
    """mode=dae: no EMA copy (encoder() is the online, ema_update is a no-op), the target is detached, the reconstruction CE
    is the y-encoder's only teacher (with lambda_jepa 0 the loss is that CE alone: it moves no online/predictor weight and
    it falls under training), the latent term adds nothing to the y-encoder's gradient, diff_target is refused."""
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, mode="dae", dae_sigma=0.5, lambda_ppd=0.0, lambda_jepa=0.0)
    assert m.target is None and m.encoder() is m.online; m.ema_update()
    torch.manual_seed(1); loss, lj, pred, tgt = m(z, im, tm, split)
    assert not tgt.requires_grad and lj.item() > 0 and loss.item() > 0
    loss.backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in list(m.online.parameters()) + list(m.pred_blocks.parameters()))
    g0 = m.yenc.value_proj.weight.grad.clone(); assert g0.abs().max() > 0
    m.zero_grad(); m.lambda_jepa = 1.0
    torch.manual_seed(1); loss1, lj1, _, _ = m(z, im, tm, split)                # same noise -> same reconstruction term
    assert torch.allclose(loss1 - lj1, loss, atol=1e-5)
    loss1.backward(); assert torch.allclose(m.yenc.value_proj.weight.grad, g0, atol=1e-6)
    m.lambda_jepa = 0.0
    opt = torch.optim.Adam(m.yenc.parameters(), lr=1e-3)
    for _ in range(40):
        opt.zero_grad(); l = m(z, im, tm, split)[0]; l.backward(); opt.step()
    torch.manual_seed(1); assert m(z, im, tm, split)[0].item() < loss.item()
    try:
        JEPA(predictor="twoway-2", r_scheme="rope", ctx_only=True, mode="dae", diff_target=True); raise AssertionError("guard did not fire")
    except ValueError:
        pass
    print("  dae mode (denoising y-encoder target) ok")


def test_sigreg_no_detach_and_additive_loss():
    """LeWM wiring (docs/sigreg_census/verdict.md 5, 7): sigreg_detach=False lets the MSE reach the target side (the
    projector gets a gradient even at lambda_sig 0; detached, it gets none), sigreg_additive gives mse + lambda * SIGReg
    (checked against the convex form under the same random slices), and the defaults keep the convex / detached form."""
    z, im, tm, split = _batch()
    torch.manual_seed(0)
    m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, mode="sigreg", lambda_sig=0.0)
    assert m.sigreg_detach and not m.sigreg_additive
    m(z, im, tm, split)[0].backward()
    assert m.projector[0].weight.grad.abs().max() == 0
    m.zero_grad(); m.sigreg_detach = False
    m(z, im, tm, split)[0].backward()
    assert m.projector[0].weight.grad.abs().max() > 0
    m.lambda_sig = 0.5
    torch.manual_seed(1); lc, lj, _, _ = m(z, im, tm, split)                   # convex: (1 - l) mse + l sig
    torch.manual_seed(1); m.sigreg_additive = True; la, lj2, _, _ = m(z, im, tm, split)   # additive: mse + l sig
    sig = (lc - 0.5 * lj) / 0.5
    assert torch.allclose(lj, lj2) and sig.item() > 0 and torch.allclose(la, lj + 0.5 * sig, atol=1e-5)
    print("  sigreg without detach + additive loss ok")


def test_ckpt_pass2_is_numerically_a_no_op():
    """ckpt_pass2 (gradient checkpointing of the sigreg / dae second pass, 2026-09-11 peak smoke fallback) must leave
    loss and gradients unchanged: same seed -> same SIGReg slices (preserve_rng_state) and same dae noise."""
    z, im, tm, split = _batch()
    for kw in (dict(mode="sigreg", sigreg_detach=False, sigreg_additive=True, lambda_sig=0.05),
               dict(mode="dae", lambda_ppd=1.0)):
        torch.manual_seed(0)
        m = JEPA(predictor="twoway-2", r_scheme="rope", n_cls=4, ctx_only=True, **kw)
        out = []
        for ck in (False, True):
            m.ckpt_pass2 = ck; m.zero_grad()
            torch.manual_seed(1); loss = m(z, im, tm, split)[0]; loss.backward()
            out.append((loss.item(), [p.grad.clone() for p in m.parameters() if p.grad is not None]))
        assert abs(out[0][0] - out[1][0]) < 1e-6 and len(out[0][1]) == len(out[1][1]) > 0
        assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(out[0][1], out[1][1]))
    print("  ckpt_pass2 numerically a no-op ok")
