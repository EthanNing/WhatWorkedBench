# Recorded evidence

`controls/` contains all 4,206 numerical-control records. `control_summary.json` summarizes 976 condition/method/budget cells. `agent_index.json` records all 108 core attempts, including 94 valid prediction artifacts and observation order. `agent_results.json` preserves core numerical predictions and measured observations. `core_summary.json` records cohort-aware aggregates and shared-inference comparisons.

`structure_diagnostic.json` contains the separate, complete eight-episode, four-source recognition/application diagnostic. Its modified first-call instruction defines a separate cohort.

`structure/` includes the eight submitted prediction tables, their purchased
observations, the public code-equivalence rules, and an independently checked
post-hoc projection. Run `python scripts/audit_structure.py` to reproduce the
change in mean effect recovery from 0.338 to 0.507 using fixed observations.

`added_workflows/` retains the supplemental run artifacts, collection records,
and independent release audit. The folder has its own audit script at
`scripts/audit_added.py`.

`conditional_effects.csv` and `mean_pair_interactions.csv` contain 3,392 conditional component effects and 342 mean pair interactions. The numerical-control records preserve acquisition observations and final predictions. Each cohort retains its original model and condition identifiers.

Use `python scripts/audit_results.py` to recompute every control score from the evaluator reference and verify core record counts. It runs offline and spends no model credits.
