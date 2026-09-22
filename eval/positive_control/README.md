# Positive control — instrument-trust producers

Archival copies of the three scripts that produced the paper's **positive-control** numbers
(§threat:instrument). They live here so the release is self-documenting; they were authored to
run *inside* a nanoTabPFN checkout (they import its `model.py` as a sibling), so **to reproduce,
drop them into `vendor/nanoTabPFN/` and run from there** (see commands below). `vendor/` is an
external dependency and is gitignored, which is why these copies exist.

## Why this control exists

Our own architecture (CellPFN) learns our SCM prior, but that alone cannot rule out
"the SCM prior and our architecture are co-adapted (both quietly broken in a matching way)" —
the two-variable worry that once shelved the paper. So we feed the **same prior** to a
**known-good, independent** architecture (nanoTabPFN, Pfefferle et al. 2025). If an independent
arch also learns our SCM in-context, the SCM is a legitimate amortizable prior, not a
co-adapted artifact. Chance = 0.5 throughout (median-split binary task).

## The three scripts

| Script | Prior fed to nanoTabPFN | Result | Role |
|---|---|---|---|
| `pc_scmprior.py` | **our paper SCM** (`prior.scm`, default `PriorConfig`) | val_acc **0.882** | **headline** — the instrument-validity number |
| `pc_native_ctrl.py` | nanoTabPFN's own bundled dump | val_acc ~0.98 | harness sanity (native prior learns) |
| `pc_v1prior.py` | our revived TabPFN-v1 extraction | flat 0.50 | context (v1 extraction does not amortize) |

**Headline wiring (source-backed):** `pc_scmprior.py` writes
`eval/results_positive_control.json` → `["scm"]["val_acc_final"]` = 0.8816 →
`build_tables.py` macro `\numPCScmAcc` (d3 → **.882**) → cited in
`paper/sections/discussion.tex` §threat:instrument and its App. B learning curve (Table 9).
The JSON is committed; these scripts are how it was produced.

## Reproduce

```bash
# 1. get a nanoTabPFN checkout at vendor/nanoTabPFN/ (Pfefferle et al. 2025; automl/TFM-Playground)
# 2. copy these three files into it
cp eval/positive_control/pc_*.py vendor/nanoTabPFN/

# headline .882 (deterministic: train seed 0, val seed 10000) — CPU is fine, ~minutes
uv run --with numpy --with torch --with schedulefree \
  python vendor/nanoTabPFN/pc_scmprior.py 800

# native harness sanity — needs the bundled dump (figshare):
#   curl -L -o vendor/nanoTabPFN/300k_150x5_2.h5 \
#     "https://ndownloader.figshare.com/files/58932628?private_link=63fc1ada93e42e388e63"
uv run --with numpy --with torch --with schedulefree python vendor/nanoTabPFN/pc_native_ctrl.py

# our v1 extraction (flat 0.50)
uv run --with numpy --with torch --with einops --with networkx --with schedulefree \
  python vendor/nanoTabPFN/pc_v1prior.py 550
```

`pc_scmprior.py` and `pc_v1prior.py` also import from this repo (`prior.scm`,
`train.data_tabpfn_v1`); they add the repo root to `sys.path` via `Path(__file__).parents[2]`,
which resolves to the repo root from either `vendor/nanoTabPFN/` or `eval/positive_control/`.
`pc_native_ctrl.py` additionally needs nanoTabPFN's own `train.py` (`PriorDumpDataLoader`) and
the `.h5` dump, so it only runs from within the checkout.
