# Agent episodes on added workflow families

This independent Flash cohort covers four archived heart-beat detection records
and four graph link-prediction sources. Each agent receives public workflow code,
two free anchors, and 32 purchased measurements. The prompts, tools, and
session limits match the six-option agent interface used by the core study.

Six of eight attempts produce valid complete prediction tables. The last two
sources use all 32 measurements before the provider returns HTTP 402 for
insufficient balance; their original attempts and tool receipts remain in
`artifacts/`. The planned eight-attempt analysis assigns zero delivered recovery
to those two missing tables. It also reconstructs their saved observations
with a shared Gaussian process.

Across all eight sources, family-macro delivered effect recovery is **0.227**,
and shared GP on the same observations reaches **0.460**. Among the six valid
submissions, shared GP improves every source; valid-only family-macro recovery
changes from **0.303** to **0.455**. Same-source, same-budget D2, grid-GP, and
effect-GP controls are included in `source_comparison.csv`.

`plan.json` records the frozen cohort. `model_results.json`, `analysis.json`,
and `source_comparison.csv` contain the numerical evidence. `artifacts/`
preserves public tasks, prompts, ordered observations, calculation receipts,
and final submissions where present. Independent accounting and release
checks are in `independent_audit.json` and `export_audit.json`.

After installing the package, run `python scripts/audit_added.py` to regrade
all saved submissions and shared-GP reconstructions using the released
references. The technical smoke belongs to a separate namespace and is
excluded from these eight attempts.
