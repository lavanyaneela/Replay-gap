"""Local MLflow: a tracking store and a model registry, both files, neither a server.

The current MLflow release refuses the filesystem tracking backend outright, and that backend
never supported the model registry in any release. SQLite is the local, serverless backend that
does support both, so that is what this uses. These tests exercise it against a temporary
database rather than mocking it, because the thing worth testing is that the registry operation
actually works on the backend the repository ships with.
"""

from __future__ import annotations

import json

import pytest

from replay_gap.tracking import LocalTracking, default_tracking_uri


def test_the_default_uri_is_a_local_sqlite_file_and_not_a_server(tmp_path):
    uri = default_tracking_uri(tmp_path / "mlruns")
    assert uri.startswith("sqlite:///")
    assert "http" not in uri
    assert (tmp_path / "mlruns").is_dir()


def test_a_disabled_tracker_records_nothing_and_still_answers(tmp_path):
    """A reviewer whose machine cannot write the database should still get every number."""
    tracking = LocalTracking(default_tracking_uri(tmp_path / "mlruns"), enabled=False)
    assert tracking.log_estimate("p", "snips", {}, {"lift": 0.1}) is None
    assert tracking.register_policy("p", "d", {}) is None
    assert tracking.summary() == {
        "enabled": False,
        "uri": tracking.uri,
        "runs": 0,
        "registered_models": 0,
    }


def test_a_disabled_tracker_still_writes_a_summary_file(tmp_path):
    tracking = LocalTracking(default_tracking_uri(tmp_path / "mlruns"), enabled=False)
    target = tmp_path / "summary.json"
    tracking.write_summary(target)
    assert json.loads(target.read_text())["enabled"] is False


@pytest.mark.slow
def test_an_estimator_run_is_recorded_and_can_be_counted_back(tmp_path):
    tracking = LocalTracking(default_tracking_uri(tmp_path / "mlruns"))
    run_id = tracking.log_estimate(
        "oracle", "snips", {"n_sessions": 100}, {"lift": 0.12, "true_lift": 0.11}
    )
    assert run_id
    assert tracking.summary()["runs"] == 1


@pytest.mark.slow
def test_registering_a_policy_twice_produces_two_versions_of_one_model(tmp_path):
    """The registry's job is to say which version was evaluated against which log."""
    tracking = LocalTracking(default_tracking_uri(tmp_path / "mlruns"))
    first = tracking.register_policy("oracle", "a candidate", {"decision": "positive"})
    second = tracking.register_policy("oracle", "a candidate", {"decision": "positive"})
    assert (first, second) == (1, 2)
    assert tracking.summary()["registered_models"] == 1


@pytest.mark.slow
def test_the_tracking_summary_round_trips_to_disk(tmp_path):
    tracking = LocalTracking(default_tracking_uri(tmp_path / "mlruns"))
    tracking.log_estimate("p", "ips", {}, {"lift": 0.0})
    target = tmp_path / "tracking.json"
    tracking.write_summary(target)
    payload = json.loads(target.read_text())
    assert payload["enabled"] is True
    assert payload["runs"] == 1
