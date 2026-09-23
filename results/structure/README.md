# Structure recognition and final prediction

Eight Flash episodes on four six-option workflows test rule reporting and use
under a fixed 20-measurement budget. Four agents receive workflow code, and
four receive the same code plus verified inactive-option rules. Each source has
one run per information view. All eight submit complete prediction tables.

`model_results.json` preserves the submitted tables and purchased observations.
`rules.json` lists the public-code relations, `analysis.json` records paired
outcomes, and `projection_independent_audit.json` checks an offline rescoring.

The post-hoc projection groups masks by code-derived equivalence, propagates
observed utilities within measured classes, and averages submitted predictions
within remaining classes. It uses exactly the agent's purchased observations.
Mean conditional-effect recovery changes from 0.338 for the submitted tables
to 0.507 after projection. Raw effect error decreases in all eight episodes.
The original agent results remain available in the same records.

Recompute this analysis with `python scripts/audit_structure.py` after
installing the package from the repository root.
