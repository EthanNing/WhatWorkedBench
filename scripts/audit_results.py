"""Recompute recorded numerical metrics from released reference tables."""
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whatworkedbench.episode import grade
root = Path(__file__).resolve().parents[1]
metrics = ("effect_mae", "interaction_mae", "grid_mae", "effect_recovery", "support_regret")
references = {p.stem: json.loads(p.read_text())["truth"] for p in (root / "benchmark/evaluator/references").glob("*.json")}
def verify(row):
    table = row["score"]["prediction_table"]
    name = row["instance"]
    if len(next(iter(table))) == 6 and not name.endswith("__d6"): name += "__d6"
    observed = row["observations"]
    if isinstance(observed, list): observed = {r["mask"]: r["utility"] for r in observed}
    score = grade(references[name], table, observed)
    for key in metrics:
        assert math.isclose(score[key], row["score"][key], abs_tol=1e-10), (name, key)
controls = [r for p in sorted((root / "results/controls").glob("*.json")) for r in json.loads(p.read_text())]
for row in controls: verify(row)
agents = json.loads((root / "results/agent_results.json").read_text())
valid = [r for r in agents if r["artifact_valid"]]
for row in valid: verify(row)
assert len(controls) == 4206
assert len(agents) == 108 and len(valid) == 94
assert len(references) == 36 and sum(map(len, references.values())) == 1248
print(json.dumps({"controls_verified": len(controls), "core_attempts": len(agents), "valid_agent_tables_verified": len(valid), "task_conditions": len(references), "native_outcomes": sum(map(len, references.values()))}, indent=2))
