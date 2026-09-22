# Recorded evidence

`controls/` contains all 4,206 numerical-control records. `control_summary.json` summarizes 976 condition/method/budget cells. `agent_index.json` records all 108 core attempts, including 94 valid prediction artifacts and observation order. `agent_results.json` preserves core numerical predictions and measured observations. `core_summary.json` records cohort-aware aggregates and shared-inference comparisons.

`structure_diagnostic.json` contains the separate eight-episode, four-source recognition/application diagnostic. Its modified first-call instruction defines a separate cohort. Across the two studies there are 116 attempts and 102 valid artifacts.

`conditional_effects.csv` and `mean_pair_interactions.csv` contain 3,392 conditional component effects and 342 mean pair interactions. The numerical-control records preserve acquisition observations and final predictions. Each cohort retains its original model and condition identifiers.

Use `python scripts/audit_results.py` to recompute every control score from the evaluator reference and verify core record counts. It runs offline and spends no model credits.
