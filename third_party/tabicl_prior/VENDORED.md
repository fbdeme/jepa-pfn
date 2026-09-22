# tabicl_prior — vendored copy of `tabicl/prior` (soda-inria/tabicl, BSD-3-Clause)

- Upstream: https://github.com/soda-inria/tabicl, path `src/tabicl/prior/`
- Commit: 8f1aa2098c894ab91dba15209cf2002ba4be6c6c (2026-08-21)
- Why vendored: `scripts/remote.sh sync` excludes `vendor/`, and the training box must generate the prior on its own CPU
  (`prior/tabicl.py`, docs/prior_v2_plan.md). Unmodified except for import paths if noted below.
- Modifications: (1) every absolute `tabicl.prior.` import rewritten to package-relative (sed), `graph_lib/eval/` plotting scripts dropped; (2) `graph_lib/_dataset.py` `list(set(...))` -> `sorted(set(...))` for feature groups, because set-of-str iteration order follows PYTHONHASHSEED and made the table stream differ between processes (prior/test_tabicl.py). No other change.
- Deps added to pyproject for it: xgboost, psutil, threadpoolctl, scipy.
