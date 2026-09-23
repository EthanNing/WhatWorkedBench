"""Regrade the recorded structure diagnostic and its fixed-evidence projection."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whatworkedbench.episode import grade
from whatworkedbench.gaussian import canonical

root = Path(__file__).resolve().parents[1]
folder = root / "results/structure"
rows = json.loads((folder / "model_results.json").read_text())
analysis = json.loads((folder / "analysis.json").read_text())
expected = {(pair["instance"], arm): pair[arm]
            for pair in analysis["pairs"] for arm in ("code", "rules")}
before, after = [], []
for row in rows:
    name = row["instance"]
    truth = json.loads((root / "benchmark/evaluator/references" /
                        f"{name}__d6.json").read_text())["truth"]
    original = row["score"]["prediction_table"]
    observed = {item["mask"]: item["utility"] for item in row["observations"]}
    groups = defaultdict(list)
    for mask in original:
        groups[canonical(row["family"], mask)].append(mask)
    projected = {}
    for masks in groups.values():
        measured = {observed[mask] for mask in masks if mask in observed}
        assert len(measured) <= 1
        value = next(iter(measured)) if measured else math.fsum(original[mask] for mask in masks) / len(masks)
        projected.update({mask: value for mask in masks})
    submitted = grade(truth, original, observed)
    repaired = grade(truth, projected, observed)
    target = expected[name, row["arm"]]
    assert math.isclose(submitted["effect_recovery"], target["recovery"], abs_tol=1e-12)
    assert math.isclose(repaired["effect_recovery"], target["projected_recovery"], abs_tol=1e-12)
    before.append(submitted["effect_recovery"])
    after.append(repaired["effect_recovery"])
assert len(rows) == 8
print(json.dumps({"episodes": len(rows), "submitted_mean": math.fsum(before) / len(before),
                  "posthoc_mean": math.fsum(after) / len(after)}, indent=2))
