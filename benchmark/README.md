# Budgeted task interface

The manifest enumerates 36 task conditions on 30 sources in eight workflow families. There are 22 four-option conditions and 14 six-option conditions, including six paired variants of existing sources. The catalog contains 1,248 configuration records, with 96 repeated original-subcube outcomes in the paired variants.

`public/tasks/` contains workflow code, factor definitions, and free anchors. `public/wwb_helpers.py` provides ridge fitting and design helpers. `evaluator/references/` contains complete outcomes and `sources/` records source attribution and preparation metadata. `prompts/` preserves the original four- and six-factor instructions.

Run the trusted evaluator from the repository root using Python 3.11+ on Linux.

```bash
python benchmark/evaluator/episode.py start --episode output/example --reference benchmark/evaluator/references/scifact.json --budget 8
python benchmark/evaluator/episode.py observe --episode output/example --masks 0001 0010
python benchmark/evaluator/episode.py status --episode output/example
python benchmark/evaluator/episode.py submit --episode output/example --prediction predictions.json
python benchmark/evaluator/episode.py score --episode output/example
```

A prediction is a JSON object mapping every legal mask to a finite utility in [0,1]. Two anchors are free. Each additional distinct measurement costs one credit. Over-budget batches are rejected atomically. Submission closes the episode, and measured values replace corresponding predictions at grading.

Run the evaluator and store references on a trusted host outside the agent's filesystem. Expose only the public task and permitted tool receipts to the evaluated agent. The CLI supplies budget accounting; your integration supplies process and filesystem isolation.

Effects are conditional score differences within the fixed native workflow and evaluation cohort. See [evaluation](../guides/evaluation.md) for metrics and comparison units.
