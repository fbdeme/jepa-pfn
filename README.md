# A JEPA Recipe for Tabular Foundation Models

Code, configs, evaluation results and released checkpoints for the paper
*A JEPA Recipe for Tabular Foundation Models* (Mingyu Jeon, Suwan Cho, Jae Young Suh; Modulabs, 2026).
arXiv: *to be added once the preprint is announced.* Under review at TMLR.

> Tabular foundation models learn to predict cell values in context, whereas world-model self-supervision asks for prediction in representation space. On a tabular foundation-model prior, the latent term of a joint-embedding predictive architecture (JEPA) collapsed in our earlier runs and took the encoder with it to a constant map. We report a recipe under which the latent term survives to convergence beside the value objective: the value head reads the encoder field rather than the predictor, and the target is an exponential moving average (EMA) difference. To bound its cost against the value-only arm, both arms train until a plateau rule stops them, with no fixed step budget. A fixed horizon had confounded a slowdown with a ceiling, since the value-only arm was still improving well past the usual budget. At convergence, in one run per arm, the JEPA arm trails the value-only arm across 147 real datasets, 32:70 wins to losses on classification (29:63 with one entry per dataset name) and 8:24 on regression, the margin small on classification and wider on regression, and the count leans the same way in each stratum and each benchmark. The JEPA arm (jepa) needs 1.42 times as many steps as the value-only arm (ds), and 1.66 times its wall-clock, to reach its plateau.

## What is here

| Path | Contents |
|---|---|
| `model/` | `pfn.py`: the cell-level in-context transformer (`CellPFN`, two-way feature/row attention blocks, derived from nanoTabPFN); `jepa.py`: the JEPA wrapper (`JEPA` with EMA target and predictor, the `SIGReg` variant), the value-only `ValueModel`, collapse statistics |
| `train/` | `train.py` (value-only arm), `train_jepa.py` (JEPA arm), plateau stop rule (`plateau.py`), data pipeline, tests |
| `prior/` | Training priors: the TabICL graph-SCM prior wrapper (`tabicl.py`), SCM / factor / real-data-matched priors |
| `third_party/tabicl_prior/` | Vendored `tabicl/prior` generator from [soda-inria/tabicl](https://github.com/soda-inria/tabicl) (BSD-3-Clause, see its `LICENSE` and `VENDORED.md`) |
| `configs/` | Every training config used in the project; the two paper arms are listed below |
| `eval/` | Benchmark and analysis scripts plus the result JSONs the paper's numbers are generated from (`suite_bench.py` = OpenML-CC18 + Grinsztajn-2022 + TabArena, classification and regression) |
| `scripts/` | Config generators and the prior-dump utilities |

Not included: the paper LaTeX source (the arXiv version is the reference), internal notes, and the
compute-box scripts.

## Released checkpoints

Both arms were trained on one GPU until the plateau rule stopped them (no fixed step budget;
`stop_delta` 0.002 over 80 consecutive validations). Final checkpoints, configs and `metrics.jsonl`
are on the Hugging Face Hub at [`fbdeme/jepa-pfn`](https://huggingface.co/fbdeme/jepa-pfn).

| Arm | What | Config | Trainer | Stop step | Wall-clock | Final val MSE | `ckpt.pt` |
|---|---|---|---|---|---|---|---|
| `ds` | value-only PFN arm | `configs/PFN_tabicl2_ds_conv_s0.yaml` | `python -m train.train` | 178,500 | 24.8 h | 0.4179 | 27 MB |
| `jepa` | JEPA arm (latent term beside the value objective) | `configs/TabularJEPA_v3_tabicl2_headenc_conv_s0.yaml` | `python -m train.train_jepa` | 253,750 | 41.3 h | 0.4360 | 70 MB |

```bash
uv run hf download fbdeme/jepa-pfn --local-dir runs/     # -> runs/<run_name>/ckpt.pt, config.yaml, metrics.jsonl
```

## Setup

```bash
uv sync
```

Python 3.12 + PyTorch. `pyproject.toml` pins a CPU torch wheel because the development machine has no
GPU; on a GPU box delete the `[tool.uv.sources]` block (and the `pytorch-cpu` index) before `uv sync`.
The TabICL prior is sampled on the fly by CPU worker processes; no dataset download is needed for training.

## Reproduce

Train the two arms (each config runs until the plateau rule fires; the JEPA arm peaked at 79 GB of GPU
memory in our run, the value-only arm is lighter):

```bash
uv run python -m train.train       configs/PFN_tabicl2_ds_conv_s0.yaml
uv run python -m train.train_jepa  configs/TabularJEPA_v3_tabicl2_headenc_conv_s0.yaml
```

Evaluate a checkpoint on the real-data suite (147 datasets; OpenML downloads on first use). The paper's
result files are `eval/results_suite_<run_name>.json`:

```bash
PFN_RUNS=<run_name> SUITE_TAG=<run_name> K_FEAT=64 CTX=1024 DEVICE=cuda \
  uv run --with openml python -m eval.suite_bench
```

Tests:

```bash
uv run --with pytest pytest model train prior
```

## License

Code and released weights: [Apache-2.0](LICENSE). The model architecture is derived from
[nanoTabPFN](https://github.com/automl/nanoTabPFN) (Apache-2.0). `third_party/tabicl_prior/` is
BSD-3-Clause, Copyright (c) 2025 Soda team @ Inria.

## Citation

See [`CITATION.cff`](CITATION.cff); the arXiv identifier will be added there once the preprint is announced.
