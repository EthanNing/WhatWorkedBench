"""Run a budgeted numerical baseline using the trusted evaluator interface."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whatworkedbench.episode import Episode
from whatworkedbench.helpers import fit_surface, suggest_experiments
from whatworkedbench.gaussian import episode as gp_episode

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--task", default="scifact")
p.add_argument("--budget", type=int, default=8)
p.add_argument("--method", choices=["ridge-main", "ridge-pair", "gp-grid", "gp-effect"], default="ridge-pair")
p.add_argument("--output", type=Path, default=Path("output/scifact-ridge"))
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
reference = root / "benchmark/evaluator/references" / (a.task + ".json")
if not reference.is_file(): p.error("Unknown task; see benchmark/manifest.json")
e = Episode(a.output)
state = e.start(reference, a.budget)
if a.method.startswith("ridge"):
    degree = 1 if a.method == "ridge-main" else 2
    for _ in range(a.budget):
        rows = [{"mask": m, "utility": u} for m, u in state["observations"].items()]
        state = e.action("observe", suggest_experiments(rows, degree=degree))
    prediction = fit_surface([{"mask": m, "utility": u} for m, u in state["observations"].items()], degree=degree)["predictions"]
else:
    dimensions = len(next(iter(state["observations"])))
    def oracle(mask):
        return e.action("observe", [mask])["observations"][mask]
    prediction, _, _, _ = gp_episode(dimensions, a.budget, oracle, "grid_variance" if a.method == "gp-grid" else "effect_variance")
e.action("submit", prediction)
score = e.action("score")
(a.output / "predictions.json").write_text(json.dumps(prediction, indent=2) + "\n")
(a.output / "score.json").write_text(json.dumps(score, indent=2) + "\n")
print(json.dumps({k: score[k] for k in ("effect_recovery", "effect_mae", "interaction_mae", "support_regret", "exact_optimal_choice")}, indent=2))
