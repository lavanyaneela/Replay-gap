"""The experiments, run at a reduced size, so CI proves they execute rather than hoping.

These tests do not re-assert the published findings, which would be circular: the findings are
whatever the measurement says, and the receipts tooling is what keeps the prose matching them.
What they assert is that each experiment runs end to end, writes the keys the README and the
number checker read, and produces internally consistent arithmetic.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from experiments import (
    e1_sign_agreement,
    e2_overlap_curve,
    e3_estimator_comparison,
    e4_detectable_lift,
    e5_propensity_misspecification,
)
from experiments.common import Timer, provenance, scenario_policies, write_result
from replay_gap.simulator.generate import InteractionLog


def test_the_timer_records_a_positive_duration():
    with Timer("x") as timer:
        sum(range(10000))
    assert timer.seconds > 0.0


def test_an_experiment_payload_is_stamped_with_its_environment(tmp_path):
    target = write_result("demo", {"value": np.float64(1.5), "count": np.int64(3)}, tmp_path)
    payload = json.loads(target.read_text())
    assert payload["environment"]["python"]
    assert payload["value"] == 1.5
    assert payload["count"] == 3


def test_an_array_in_a_payload_is_written_as_a_list(tmp_path):
    target = write_result("demo", {"values": np.arange(3)}, tmp_path)
    assert json.loads(target.read_text())["values"] == [0, 1, 2]


def test_an_unserialisable_value_is_refused_rather_than_stringified(tmp_path):
    with pytest.raises(TypeError, match="cannot serialise"):
        write_result("demo", {"value": object()}, tmp_path)


def test_provenance_carries_the_configuration_digest(small_config):
    payload = provenance(small_config, 100)
    assert payload["n_sessions"] == 100
    assert payload["config_digest"]
    assert payload["n_candidates"] == small_config.slate.n_candidates


def test_the_scenario_family_weights_sum_to_one_for_every_draw(small_fixture):
    """Uniform over the simplex is what makes E1 a measurement rather than a demonstration."""
    for policy in scenario_policies(small_fixture, 12, seed=3):
        total = policy.truth_weight + policy.incumbent_weight + policy.noise_weight
        assert total == pytest.approx(1.0)


def test_the_scenario_family_is_reproducible_from_its_seed(small_fixture):
    first = scenario_policies(small_fixture, 5, seed=9)
    second = scenario_policies(small_fixture, 5, seed=9)
    assert [p.truth_weight for p in first] == [p.truth_weight for p in second]


def test_the_scenario_family_covers_a_range_of_distances_from_the_incumbent(small_fixture):
    weights = [p.incumbent_weight for p in scenario_policies(small_fixture, 60, seed=4)]
    assert min(weights) < 0.2
    assert max(weights) > 0.7


def test_corrupting_propensities_by_a_factor_of_one_leaves_the_log_untouched(small_log):
    assert e5_propensity_misspecification.corrupt_propensities(small_log, 1.0) is small_log


def test_corrupting_propensities_preserves_the_ordering_of_the_sessions(small_log):
    """Tempering is a monotone transform: what changes is the magnitudes, not the ranking."""
    corrupted = e5_propensity_misspecification.corrupt_propensities(small_log, 3.0)
    assert np.array_equal(
        np.argsort(corrupted.log_propensity), np.argsort(small_log.log_propensity)
    )


def test_corrupting_propensities_compresses_their_spread(small_log):
    corrupted = e5_propensity_misspecification.corrupt_propensities(small_log, 4.0)
    assert corrupted.log_propensity.std() < small_log.log_propensity.std()


def test_corrupting_propensities_holds_their_mean_fixed(small_log):
    """Without this the sweep would measure a constant offset that self normalisation cancels."""
    corrupted = e5_propensity_misspecification.corrupt_propensities(small_log, 6.0)
    assert corrupted.log_propensity.mean() == pytest.approx(small_log.log_propensity.mean())


def test_corrupting_propensities_leaves_every_other_column_alone(small_log):
    corrupted = e5_propensity_misspecification.corrupt_propensities(small_log, 2.0)
    assert isinstance(corrupted, InteractionLog)
    assert np.array_equal(corrupted.clicks, small_log.clicks)
    assert np.array_equal(corrupted.slate, small_log.slate)


def test_a_non_positive_corruption_factor_is_rejected(small_log):
    with pytest.raises(ValueError, match="factor must be positive"):
        e5_propensity_misspecification.corrupt_propensities(small_log, 0.0)


@pytest.fixture
def tiny_config(small_config, monkeypatch, tmp_path):
    """Point every experiment at a small world and a temporary output directory."""
    config = replace(
        small_config,
        log=replace(small_config.log, n_sessions=300),
        experiments=replace(
            small_config.experiments,
            e1_scenarios=5,
            e1_sessions=300,
            e2_divergence_steps=2,
            e3_replicates=2,
            e3_sessions=300,
            e5_factors=(1.0, 4.0),
            e5_sessions=300,
        ),
    )
    for module in (
        e1_sign_agreement,
        e2_overlap_curve,
        e3_estimator_comparison,
        e4_detectable_lift,
        e5_propensity_misspecification,
    ):
        monkeypatch.setattr(module, "load_config", lambda _config=config: _config)
    monkeypatch.setattr("experiments.common.EXPERIMENT_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    return config


@pytest.mark.slow
def test_e1_runs_and_reports_a_sign_disagreement_rate(tiny_config, capsys):
    payload = e1_sign_agreement.run()
    e1_sign_agreement.report(payload)
    assert 0.0 <= payload["naive_sign_disagreement_rate"] <= 1.0
    assert payload["n_decisive"] <= payload["n_scenarios"]
    assert len(payload["rows"]) == payload["n_scenarios"]
    assert "sign disagreement rate" in capsys.readouterr().out


@pytest.mark.slow
def test_e1_overclaim_counts_agree_with_its_own_rows(tiny_config):
    payload = e1_sign_agreement.run()
    counted = sum(
        1
        for row in payload["rows"]
        if np.isfinite(row["naive_lift"]) and row["naive_lift"] > row["true_lift"]
    )
    assert payload["naive_overclaims"] == counted


@pytest.mark.slow
def test_e2_runs_and_both_sweeps_start_at_the_incumbent(tiny_config, capsys):
    payload = e2_overlap_curve.run()
    e2_overlap_curve.report(payload)
    for curve in ("model_only", "model_and_serving"):
        first = payload[curve][0]
        assert first["true_lift"] == pytest.approx(0.0, abs=1e-9)
        assert first["ess_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert "effective n" in capsys.readouterr().out


@pytest.mark.slow
def test_e2_effective_sample_size_falls_as_the_candidate_moves_away(tiny_config):
    payload = e2_overlap_curve.run()
    fractions = [row["ess_fraction"] for row in payload["model_only"]]
    assert fractions[-1] < fractions[0]


@pytest.mark.slow
def test_e3_runs_and_every_policy_gets_every_estimator(tiny_config, capsys):
    payload = e3_estimator_comparison.run()
    e3_estimator_comparison.report(payload)
    for entry in payload["summary"].values():
        for estimator in payload["estimators"]:
            assert f"{estimator}_rmse" in entry
        assert entry["best_by_rmse"] in payload["estimators"]
    assert "RMSE" in capsys.readouterr().out


@pytest.mark.slow
def test_e3_reports_that_a_sharply_served_policy_gets_the_same_naive_estimate(tiny_config):
    """The structural blindness, asserted rather than assumed: serving does not reach replay."""
    payload = e3_estimator_comparison.run()
    for entry in payload["serving_blindness"]:
        assert entry["naive_lift_mild"] == pytest.approx(entry["naive_lift_sharp"], abs=1e-9)


@pytest.mark.slow
def test_e4_runs_and_its_session_counts_scale_with_the_square_of_the_lift(tiny_config, capsys):
    payload = e4_detectable_lift.run()
    e4_detectable_lift.report(payload)
    for row in payload["rows"]:
        if row["degenerate"]:
            continue
        expected = (row["minimum_detectable_lift"] / payload["target_lift"]) ** 2
        assert row["sessions_for_one_percent"] == pytest.approx(
            payload["n_sessions"] * expected, rel=0.01
        )
    assert "smallest lift" in capsys.readouterr().out


@pytest.mark.slow
def test_e4_excludes_the_degenerate_self_comparison_from_its_summary(tiny_config):
    payload = e4_detectable_lift.run()
    assert "incumbent" in payload["degenerate_policies"]
    assert payload["best_policy"] != "incumbent"


@pytest.mark.slow
def test_e5_runs_and_reports_when_the_correction_stops_helping(tiny_config, capsys):
    payload = e5_propensity_misspecification.run()
    e5_propensity_misspecification.report(payload)
    assert payload["factors"][0] == 1.0
    assert len(payload["rows"]) == len(payload["policies"]) * len(payload["factors"])
    assert "propensit" in capsys.readouterr().out.lower()


@pytest.mark.slow
def test_e5_leaves_the_uncorrupted_row_matching_the_clean_log(tiny_config):
    payload = e5_propensity_misspecification.run()
    for entry in payload["crossovers"].values():
        clean = next(
            row
            for row in payload["rows"]
            if row["policy"] == entry["policy"] and row["factor"] == 1.0
        )
        assert clean["snips_error"] == pytest.approx(entry["clean_snips_error"])
