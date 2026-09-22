"""On-the-fly TabPFN-v1 prior (the original SCM/MLP "differentiable prior", vendored from
tabpfn==0.1.11 under vendor/tabpfn_v1). Its hyperprior was tuned by the TabPFN authors to
transfer to real tables, and unlike TabICL's it is SMALL-TABLE native -- signal at a few hundred
rows -- so our model can actually learn it (our attention diffuses past ~1000 rows; measured).

TabPFNv1Prior.sample_batch mimics SCMPrior/DumpPrior's `.x` contract: (B, R, feat+1) with the
target appended as the last column (use the `y_last` policy). Hyperparameters are RESAMPLED per
batch from ranges inspired by model_configs.get_diff_causal, so the model sees varied mechanisms
(depth/width/causes/activation/noise) rather than one fixed architecture. Fast (~13 ms/batch on
CPU) so we generate fresh tables every step -- no finite-pool overfitting.
"""
import sys
import types
from pathlib import Path

import numpy as np
import torch

_V1 = Path(__file__).parents[1] / "vendor/tabpfn_v1"
if "tabpfn" not in sys.modules:
    sys.path.insert(0, str(_V1))
    _pkg = types.ModuleType("tabpfn")
    _pkg.__path__ = [str(_V1)]
    sys.modules["tabpfn"] = _pkg
import tabpfn.utils  # noqa: E402,F401
import tabpfn.priors.mlp as _mlp  # noqa: E402

# Smooth nonlinear activations only. Identity makes the MLP linear and ReLU tends flatter, both
# collapsing the GBM-over-linear headroom (Identity in -> -0.01, wide mix -> +0.03); Tanh/GELU
# near the measured sweet spot keep it high (+0.14). We want nonlinear mechanisms to learn.
_ACTS = [torch.nn.Tanh, torch.nn.GELU]


class _Batch:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x


class TabPFNv1Prior:
    """Fresh TabPFN-v1 SCM/MLP tables each call; target appended as the last column."""

    def __init__(self, seed=0, rows=512, feat=10):
        self.rng = np.random.default_rng(seed)
        self.rows = rows
        self.feat = feat

    def _hp(self):
        r = self.rng
        act = _ACTS[r.integers(len(_ACTS))]
        return dict(
            num_causes=int(r.integers(6, 11)),
            prior_mlp_hidden_dim=int(r.integers(48, 81)),
            num_layers=int(r.integers(3, 5)),
            prior_mlp_activations=lambda a=act: a(),
            noise_std=float(r.uniform(0.02, 0.1)),
            prior_mlp_dropout_prob=float(r.uniform(0.0, 0.1)),
            y_is_effect=True, pre_sample_causes=True, pre_sample_weights=False,
            block_wise_dropout=True, init_std=1.0, sort_features=False, in_clique=False,
            sampling="normal", is_causal=bool(r.random() < 0.8),
            prior_mlp_scale_weights_sqrt=True, random_feature_rotation=True,
            mix_activations=False, multiclass_type="rank", num_classes=2,
            new_mlp_per_example=True, verbose=False,
        )

    def sample_batch(self, batch_size):
        with torch.no_grad():
            x, y, _ = _mlp.get_batch(batch_size=batch_size, seq_len=self.rows,
                                     num_features=self.feat, hyperparameters=self._hp(),
                                     device="cpu")
        x = x.numpy().transpose(1, 0, 2)              # (B, R, feat)
        y = y.numpy().T[:, :, None]                   # (B, R, 1)
        xf = np.concatenate([x, y], -1).astype(np.float32)   # target = last col
        # rare non-finite rows -> NaN so make_batch masks them (never a training target)
        xf[~np.isfinite(xf)] = np.nan
        return _Batch(xf)


if __name__ == "__main__":  # smoke + headroom self-check
    p = TabPFNv1Prior(seed=0, rows=512, feat=10)
    b = p.sample_batch(8)
    print("x", b.x.shape, "finite frac", float(np.isfinite(b.x).mean()))
