"""The diagnostics, including the arithmetic that decides when the tool refuses."""

from __future__ import annotations

import numpy as np
import pytest

from replay_gap.diagnostics.detectable_lift import (
    minimum_detectable_lift,
    resolution_floor,
    sessions_needed_for_rate,
)
from replay_gap.diagnostics.ess import effective_sample_size, ess_fraction
from replay_gap.diagnostics.overlap import assess_overlap
from replay_gap.estimators.ips import importance_weights


def test_equal_weights_give_an_effective_sample_size_equal_to_the_sample_size():
    assert effective_sample_size(np.ones(500)) == pytest.approx(500.0)
    assert ess_fraction(np.ones(500)) == pytest.approx(1.0)


def test_scaling_every_weight_leaves_the_effective_sample_size_unchanged(rng):
    weights = rng.gamma(2.0, size=400)
    assert effective_sample_size(weights) == pytest.approx(effective_sample_size(13.0 * weights))


def test_one_session_carrying_all_the_weight_gives_an_effective_sample_size_of_one():
    weights = np.zeros(100)
    weights[0] = 1.0
    assert effective_sample_size(weights) == pytest.approx(1.0)


def test_more_variable_weights_give_a_smaller_effective_sample_size(rng):
    mild = rng.gamma(20.0, size=1000)
    wild = rng.gamma(0.4, size=1000)
    assert effective_sample_size(wild) < effective_sample_size(mild)


def test_all_zero_weights_give_an_effective_sample_size_of_zero():
    assert effective_sample_size(np.zeros(50)) == 0.0
    assert ess_fraction(np.zeros(50)) == 0.0


def test_an_empty_weight_array_gives_an_effective_sample_size_of_zero():
    assert effective_sample_size(np.array([])) == 0.0
    assert ess_fraction(np.array([])) == 0.0


def test_a_negative_weight_is_rejected_rather_than_silently_squared():
    with pytest.raises(ValueError, match="cannot be negative"):
        effective_sample_size(np.array([1.0, -1.0]))


def test_weights_that_are_not_one_dimensional_are_rejected():
    with pytest.raises(ValueError, match="one dimensional"):
        effective_sample_size(np.ones((2, 2)))


def test_the_incumbent_keeps_the_whole_log(small_log, incumbent_trace):
    weights = importance_weights(small_log, incumbent_trace, clip=None)
    report = assess_overlap(small_log, incumbent_trace, weights, 0.02, 0.05)
    assert report.ess_fraction == pytest.approx(1.0)
    assert report.usable
    assert "sufficient" in report.reason


def test_a_divergent_policy_loses_most_of_the_log(small_log, oracle_trace, incumbent_trace):
    oracle_weights = importance_weights(small_log, oracle_trace, clip=None)
    incumbent_weights = importance_weights(small_log, incumbent_trace, clip=None)
    assert ess_fraction(oracle_weights) < ess_fraction(incumbent_weights)


def test_overlap_refuses_when_the_effective_sample_size_falls_below_the_floor(
    small_log, oracle_trace
):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    report = assess_overlap(
        small_log, oracle_trace, weights, min_ess_fraction=0.99, max_weight_share=1.0
    )
    assert not report.usable
    assert "effective sample size" in report.reason


def test_overlap_refuses_when_one_session_dominates_the_weight(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    report = assess_overlap(
        small_log, oracle_trace, weights, min_ess_fraction=0.0001, max_weight_share=1e-9
    )
    assert not report.usable
    assert "one session carries" in report.reason


def test_overlap_reports_both_reasons_when_both_thresholds_are_breached(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    report = assess_overlap(small_log, oracle_trace, weights, 0.99, 1e-9)
    assert report.reason.count(";") == 1


def test_overlap_reports_the_share_of_the_log_naive_replay_would_keep(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    report = assess_overlap(small_log, oracle_trace, weights, 0.02, 0.05)
    expected = (oracle_trace.top1 == small_log.slate[:, 0]).mean()
    assert report.top1_agreement == pytest.approx(expected)


def test_overlap_reports_unseen_first_slot_exposure_as_a_probability(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    report = assess_overlap(small_log, oracle_trace, weights, 0.02, 0.05)
    assert 0.0 <= report.unseen_exposure <= 1.0


def test_overlap_rejects_a_weight_array_of_the_wrong_length(small_log, oracle_trace):
    with pytest.raises(ValueError, match="one entry per session"):
        assess_overlap(small_log, oracle_trace, np.ones(3), 0.02, 0.05)


def test_an_overlap_report_round_trips_through_a_dictionary(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    report = assess_overlap(small_log, oracle_trace, weights, 0.02, 0.05)
    payload = report.as_dict()
    assert payload["policy"] == oracle_trace.name
    assert payload["n_sessions"] == small_log.n_sessions


def test_a_usable_report_raises_nothing(small_log, incumbent_trace):
    weights = importance_weights(small_log, incumbent_trace, clip=None)
    assess_overlap(small_log, incumbent_trace, weights, 0.02, 0.05).require_usable()


def test_an_unusable_report_raises_a_refusal_carrying_its_remedy(small_log, oracle_trace):
    """The CLI turns this into exit code 2; a library caller gets the exception instead."""
    from replay_gap.errors import RefusalError

    weights = importance_weights(small_log, oracle_trace, clip=None)
    report = assess_overlap(small_log, oracle_trace, weights, 0.99, 1.0)
    with pytest.raises(RefusalError, match="cannot estimate the value of") as caught:
        report.require_usable()
    assert "online experiment" in caught.value.remedy


def test_the_minimum_detectable_lift_scales_with_the_standard_error():
    assert minimum_detectable_lift(0.02) == pytest.approx(2.0 * minimum_detectable_lift(0.01))


def test_the_minimum_detectable_lift_matches_the_two_quantile_formula():
    from scipy.stats import norm

    expected = (norm.ppf(0.975) + norm.ppf(0.8)) * 0.05
    assert minimum_detectable_lift(0.05, 0.05, 0.8) == pytest.approx(expected)


def test_demanding_more_power_raises_the_minimum_detectable_lift():
    assert minimum_detectable_lift(0.05, power=0.95) > minimum_detectable_lift(0.05, power=0.8)


def test_a_negative_standard_error_is_rejected():
    with pytest.raises(ValueError, match="standard error"):
        minimum_detectable_lift(-0.1)


@pytest.mark.parametrize(("alpha", "power"), [(0.0, 0.8), (1.0, 0.8), (0.05, 0.0), (0.05, 1.0)])
def test_an_alpha_or_power_outside_the_unit_interval_is_rejected(alpha, power):
    with pytest.raises(ValueError):
        minimum_detectable_lift(0.05, alpha, power)


def test_the_resolution_floor_is_one_over_the_sample_size():
    assert resolution_floor(34) == pytest.approx(1.0 / 34)


def test_the_resolution_floor_of_an_empty_sample_is_rejected():
    with pytest.raises(ValueError, match="n_trials"):
        resolution_floor(0)


def test_the_sample_needed_for_a_rate_rounds_up_rather_than_down():
    assert sessions_needed_for_rate(0.003) == 334
    assert sessions_needed_for_rate(0.01, n_events=5) == 500


@pytest.mark.parametrize(("rate", "events"), [(0.0, 1), (1.5, 1), (0.1, 0)])
def test_an_impossible_rate_or_event_count_is_rejected(rate, events):
    with pytest.raises(ValueError):
        sessions_needed_for_rate(rate, events)
