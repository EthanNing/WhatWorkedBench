# Added-workflow agent submissions

The B=32 Flash cohort has six valid submitted prediction tables, spanning three
beat-detection sources and three graph-link-prediction sources. Shared GP
reconstruction uses each agent's acquired observations. Effect recovery
improves for every valid submission.

| Family | Valid submissions | Agent table | Same-observation GP |
| --- | ---: | ---: | ---: |
| Beat detection | 3 | 0.412 | 0.573 |
| Graph link prediction | 3 | 0.194 | 0.338 |
| Equal-family mean | 6 | 0.303 | 0.455 |

The comparison includes the six valid submissions. Two additional attempted
episodes ended before submission after an upstream service interruption; their
raw records remain here for audit. The source-level scores and validity flags
are in `source_comparison.csv`. Filter to `valid=True` and average the three
sources in each family to reproduce this table.

The frozen plan, ordered observations, calculation receipts, submitted tables,
and independent audit are available here. Run `python scripts/audit_added.py`
to regrade the saved artifacts against the released references.
