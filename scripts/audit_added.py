"""Regrade the eight added-workflow agent attempts from released artifacts."""
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whatworkedbench.episode import grade
from whatworkedbench.gaussian import geometry, posterior

root = Path(__file__).resolve().parents[1]
folder = root / "results/added_workflows"
plan = json.loads((folder / "plan.json").read_text())
saved = {row["job_id"]: row for row in json.loads((folder / "model_results.json").read_text())}
comparison = {row["source"]: row for row in csv.DictReader((folder / "source_comparison.csv").open())}
families = {"beat_detection": [], "graph_link_prediction": []}

for job in plan["jobs"]:
    name = job["instance"]
    row = saved[job["job_id"]]
    exported = folder / "artifacts" / job["job_id"]
    truth = json.loads((root / "benchmark/evaluator/references" / f"{name}.json").read_text())["truth"]
    receipts = json.loads((exported / "evidence.json").read_text())["observations"]
    observed = {item["mask"]: item["utility"] for item in receipts}
    assert len(receipts) == 34 and len(observed) == 34
    assert all(truth[mask] == value for mask, value in observed.items())

    submitted = exported / "submission.json"
    valid = submitted.exists()
    assert valid == row["artifact_valid"]
    agent = grade(truth, json.loads(submitted.read_text()), observed)["effect_recovery"] if valid else 0.0
    prediction, _, _ = posterior(geometry(6), observed)
    shared = grade(truth, prediction, observed)["effect_recovery"]
    source = comparison[name.removesuffix("__d6")]
    assert math.isclose(agent, float(source["delivered_recovery"]), abs_tol=1e-12)
    assert math.isclose(shared, float(source["shared_gp_recovery"]), abs_tol=1e-12)
    families[job["family"]].append((agent, shared, valid))

assert len(saved) == 8
family_means = {}
for name, rows in families.items():
    assert len(rows) == 4
    family_means[name] = {"agent": math.fsum(x[0] for x in rows) / 4,
                          "shared_gp": math.fsum(x[1] for x in rows) / 4,
                          "valid": sum(x[2] for x in rows)}
overall = {metric: math.fsum(family_means[name][metric] for name in families) / 2
           for metric in ("agent", "shared_gp")}
analysis = json.loads((folder / "analysis.json").read_text())
assert sum(family_means[name]["valid"] for name in families) == analysis["valid"] == 6
assert math.isclose(overall["agent"], analysis["family_macro"]["delivered_recovery"], abs_tol=1e-12)
assert math.isclose(overall["shared_gp"], analysis["family_macro"]["shared_gp_recovery"], abs_tol=1e-12)
print(json.dumps({"episodes": 8, "valid": 6, "family_means": family_means,
                  "family_macro": overall}, indent=2))
