"""Budget, integrity and evaluation-boundary checks for the portable engine."""
import concurrent.futures
import json

import pytest

from whatworkedbench.episode import Episode, grade, strict_json, validate


def reference(tmp_path):
    # A tiny synthetic fixture checks the protocol, not benchmark difficulty.
    truth = {format(i, "04b"): i / 16 for i in range(16)}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"instance": "fixture", "truth": truth}))
    return path, truth


def test_budget_is_atomic_under_concurrent_requests(tmp_path):
    path, _ = reference(tmp_path)
    episode = Episode(tmp_path / "episode")
    episode.start(path, 1)
    with pytest.raises(ValueError, match="Insufficient"):
        episode.action("observe", ["0001", "0010"])
    assert episode.action("status")["credits_remaining"] == 1

    def request(mask):
        try:
            episode.action("observe", [mask])
            return True
        except ValueError:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(request, ["0001", "0010"])) == 1
    status = episode.action("status")
    assert status["credits_remaining"] == 0
    assert len(status["observations"]) == 3
    episode.action("observe", list(status["observations"]) * 3)
    assert episode.action("status") == status
    assert "reference" not in status and "truth" not in status


def test_reference_integrity_and_no_post_submission_queries(tmp_path):
    path, truth = reference(tmp_path)
    episode = Episode(tmp_path / "episode")
    episode.start(path, 4)
    receipt = episode.action("submit", truth)
    assert receipt["closed"] and "effect_recovery" not in receipt
    assert episode.action("score")["effect_recovery"] == 1
    with pytest.raises(ValueError, match="already closed"):
        episode.action("observe", ["0010"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Reference changed"):
        episode.action("status")


def test_strict_artifacts_and_canonicalized_observations(tmp_path):
    _, truth = reference(tmp_path)
    for malformed in ['{"00": 0, "00": 1}', '{"x": NaN}']:
        with pytest.raises(ValueError):
            strict_json(malformed)
    for change in [True, float('nan'), -1, 2]:
        with pytest.raises(ValueError):
            validate({**truth, "0000": change})
    with pytest.raises(ValueError, match="trusted reference"):
        grade(truth, truth, {"0000": .9})
    prediction = {m:.5 for m in truth}
    scored = grade(truth, prediction, truth)
    assert scored["effect_recovery"] == 1 and scored["strict_effect_reconstruction"]
    assert prediction["0000"] == .5  # Caller data is not mutated.


def test_extension_dimensions_are_scored_without_invented_task_counts():
    truth = {format(i, "06b"): .4 + .1 * bool(i & 1) for i in range(64)}
    result = grade(truth, truth, {})
    assert result["n_effect_contrasts"] == 192
    assert result["n_interaction_contrasts"] == 15
    assert result["effect_recovery"] == 1
