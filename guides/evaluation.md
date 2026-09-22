# Evaluation protocol

## Task and budget

A task specifies native workflow code, four or six binary component changes, fixed source data, and a utility in [0,1]. A mask lists options in the public `factor_order`. The response surface is the complete mapping from legal masks to scores. Four options yield 16 configurations; six yield 64.

Each episode begins with the all-zero and all-one anchors. A budget B buys B additional distinct measurements. Repeated queries are free. A batch exceeding the remaining budget fails atomically. The agent submits a complete finite prediction table; observed entries are replaced with their measured values before evaluation.

## Metrics

A conditional component effect is the score change from toggling one option while holding every other option fixed. Four-option tasks contain 32 such contrasts; six-option tasks contain 192. Pair interactions are mean second differences over the remaining options, giving 6 or 15 pair summaries.

| Metric | Definition | Direction |
| --- | --- | --- |
| Effect MAE | Mean absolute error over every conditional effect | Lower |
| Effect recovery | max(0, 1 − effect MAE / mean absolute true effect) | Higher |
| Interaction MAE | Mean error in mean pair interactions | Lower |
| Grid MAE | Mean absolute utility error across the complete table | Lower |
| Selection regret | Best true utility minus utility of the predicted best mask | Lower |
| Exact optimal choice | Selected utility matches the true maximum | Higher |
| Strict reconstruction | Every effect error is at most 10% of the true utility range | Higher |

The evaluator handles zero-amplitude surfaces explicitly. Selection ties prefer fewer enabled options, then lexicographic mask order. See `whatworkedbench/episode.py` for exact numerical tolerances.

## Comparisons

Report task conditions and source identities separately. The six paired six-option tasks share sources with the four-option catalog. Random-design results average trials within source. Family-macro aggregates average within family and then weight families equally. Keep budgets, information views, computation tools, and deadlines fixed.

Shared inference fits a new estimator to exactly the agent's saved observations. It measures how effectively the acquired evidence supports reconstruction. Code-aware controls derive equivalence classes from visible parameter guards. The separate eight-episode diagnostic records explicit structure recognition and subsequent use under its own prompt.

## Agent integration

The evaluator runs in a trusted process with reference files and writable episode state. Agent environments receive `benchmark/public/tasks/<task>.json`, allowed helpers, and tool receipts only. Your adapter maps measurement requests to `Episode.action("observe", masks)` and final tables to `Episode.action("submit", predictions)`. Keep native inputs, grading results, references, and host state inaccessible during the episode. Record all attempts and declare your isolation mechanism.
