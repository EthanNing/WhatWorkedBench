"""Portable trusted episode engine; only the Python standard library is needed.

This file can be copied out of the repository. The evaluator owns this process,
the reference file, and the state directory. Agents receive tool return values;
giving an agent host filesystem access is not an evaluation boundary.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path


def strict_json(text):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate(table):
    if not isinstance(table, dict) or not table or not all(isinstance(m, str) for m in table):
        raise ValueError("Expected a complete mask-to-utility mapping")
    dimensions = len(next(iter(table)))
    if not 2 <= dimensions <= 8:
        raise ValueError("Expected 2 to 8 factors")
    masks = [format(i, f"0{dimensions}b") for i in range(2 ** dimensions)]
    if set(table) != set(masks):
        raise ValueError("Incomplete or invalid configuration masks")
    for value in table.values():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Utilities must be finite numbers in [0, 1]")
    return {m: float(table[m]) for m in masks}


def grade(truth, prediction, observations):
    truth, prediction = validate(truth), validate(prediction)
    if truth.keys() != prediction.keys():
        raise ValueError("Prediction and reference dimensions differ")
    for mask, value in observations.items():
        if mask not in truth or truth[mask] != value:
            raise ValueError("Observation does not match the trusted reference")
        prediction[mask] = value
    n = len(truth)
    d = len(next(iter(truth)))
    gold, pred = list(truth.values()), list(prediction.values())
    mean = lambda values: math.fsum(values) / len(values)
    true_edges, edge_errors, pair_errors = [], [], []
    for bit in range(d):
        for low in range(n):
            if low & (1 << bit):
                continue
            high = low | (1 << bit)
            effect = gold[high] - gold[low]
            true_edges.append(effect)
            edge_errors.append(abs((pred[high] - pred[low]) - effect))
    for a, b in itertools.combinations(range(d), 2):
        def interaction(values):
            return mean([values[i | (1 << a) | (1 << b)] - values[i | (1 << a)]
                         - values[i | (1 << b)] + values[i]
                         for i in range(n) if not i & ((1 << a) | (1 << b))])
        pair_errors.append(abs(interaction(pred) - interaction(gold)))
    amplitude = mean([abs(e) for e in true_edges])
    edge_mae, span = mean(edge_errors), max(gold) - min(gold)
    selected = min(prediction, key=lambda mask: (-prediction[mask], mask.count('1'), mask))
    regret = max(gold) - truth[selected]
    return {"effect_mae": edge_mae, "interaction_mae": mean(pair_errors),
            "grid_mae": mean([abs(a-b) for a,b in zip(gold,pred)]),
            "effect_recovery": max(0., 1 - edge_mae / amplitude) if amplitude > 1e-12 else float(edge_mae <= 1e-12),
            "support_regret": regret, "selected_mask": selected,
            "exact_optimal_choice": regret <= 1e-12, "near_optimal_choice": regret <= .02,
            "strict_effect_reconstruction": max(edge_errors) <= max(1e-12, .1 * span),
            "mean_absolute_true_edge_effect": amplitude, "true_utility_range": span,
            "n_effect_contrasts": len(edge_errors), "n_interaction_contrasts": len(pair_errors),
            "prediction_table": prediction}


class Episode:
    """Host-only persistent budget and final-artifact state, guarded by a lock."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "state.json"

    def _write(self, state):
        temporary = self.directory / "state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(self.path)

    def start(self, reference, budget):
        reference = Path(reference).resolve()
        reference_bytes = reference.read_bytes()
        record = strict_json(reference_bytes.decode())
        truth = validate(record["truth"])
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 <= budget <= len(truth)-2:
            raise ValueError("Invalid measurement budget")
        self.directory.mkdir(parents=True, exist_ok=False)
        masks = list(truth)
        state = {"reference": str(reference), "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                 "instance": record["instance"], "budget": budget,
                 "observations": {m: truth[m] for m in (masks[0], masks[-1])}, "submission": None}
        self._write(state)
        return self.public(state)

    @staticmethod
    def public(state):
        return {"instance": state["instance"], "observations": state["observations"],
                "credits_remaining": state["budget"] - len(state["observations"]) + 2,
                "closed": state["submission"] is not None}

    def action(self, action, payload=None):
        with (self.directory / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = strict_json(self.path.read_text())
            reference_bytes = Path(state["reference"]).read_bytes()
            if hashlib.sha256(reference_bytes).hexdigest() != state["reference_sha256"]:
                raise ValueError("Reference changed after episode start")
            truth = validate(strict_json(reference_bytes.decode())["truth"])
            if action == "status":
                return self.public(state)
            if action == "score":
                if state["submission"] is None:
                    raise ValueError("No final artifact")
                return grade(truth, state["submission"], state["observations"])
            if state["submission"] is not None:
                raise ValueError("Episode is already closed")
            if action == "observe":
                if not isinstance(payload, list) or any(not isinstance(m, str) or m not in truth for m in payload):
                    raise ValueError("Expected a list of legal masks")
                new = list(dict.fromkeys(m for m in payload if m not in state["observations"]))
                if len(new) > self.public(state)["credits_remaining"]:
                    raise ValueError("Insufficient credits; batch rejected without spending")
                state["observations"].update({m: truth[m] for m in new})
            elif action == "submit":
                prediction = validate(payload)
                if prediction.keys() != truth.keys():
                    raise ValueError("Wrong prediction dimensions")
                prediction.update(state["observations"])
                state["submission"] = prediction
            else:
                raise ValueError("Unknown action")
            self._write(state)
            return self.public(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "observe", "submit", "score"])
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--masks", nargs="*")
    parser.add_argument("--prediction", type=Path)
    args = parser.parse_args()
    episode = Episode(args.episode)
    if args.action == "start":
        if args.reference is None:
            parser.error("start requires --reference")
        result = episode.start(args.reference, args.budget)
    else:
        if args.action == "submit" and args.prediction is None:
            parser.error("submit requires --prediction")
        payload = strict_json(args.prediction.read_text()) if args.action == "submit" else args.masks
        result = episode.action(args.action, payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
