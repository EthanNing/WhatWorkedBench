# Contributing

We welcome agent adapters, acquisition policies, reconstruction methods, and new executable tasks.

## Develop

Use Python 3.11+ on Linux. Install `python -m pip install -e ".[test]"` and run `python -m pytest`.
Open an issue describing the intended contribution, then submit a focused pull request with relevant tests and reproduction commands.

## Compare methods

Hold task, budget, free anchors, tool access, and information view fixed. Report the complete prediction table, consumed observations, failures, and all declared trials. Keep references and episode state on the trusted evaluator host. Describe the isolation used by your agent adapter.

## Add a task

Provide public workflow code, four or six binary options, source attribution, deterministic preparation, all native configuration outcomes, and an independent metric check. Record relationships to existing sources and task conditions. Keep third-party license notices with the source manifest.

## Website

The project page lives in `docs/`. Preview with `python -m http.server 8000 --directory docs`. GitHub Pages serves this directory from `main`.

Contributions to project code use the repository Apache 2.0 license. Dataset terms remain with their original rights holders.
