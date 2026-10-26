"""Estimator properties, and the identity that makes the bootstrap trustworthy.

The most load bearing test here is the one asserting that each estimator's ratio form, evaluated
with every session counted once, equals the direct function of the same name. The bootstrap only
ever evaluates the ratio form, so if the two drifted apart the intervals would be intervals
around a different estimator than the point estimate.
"""

from __future__ import annotations

import numpy as np
import pytest

from replay_gap.estimators.bootstrap import (
    BootstrapInterval,
    RatioEstimator,
    bootstrap_ratio_lifts,
)
from replay_gap.estimators.doubly_robust import dr_value, reward_model_terms
from replay_gap.estimators.inputs import build_estimator_inputs
from replay_gap.estimators.ips import importance_weights, ips_value
from replay_gap.estimators.naive import naive_match_mask, naive_match_rate, naive_replay_value
from replay_gap.estimators.snips import snips_value
from replay_gap.evaluation import trace_policy


def test_the_incumbents_importance_weights_are_exactly_one(small_log, incumbent_trace):
    """Re-evaluating the logging policy must leave the log untouched, exactly, not nearly."""
    weights = importance_weights(small_log, incumbent_trace, clip=None)
    assert np.allclose(weights, 1.0, atol=1e-12)


def test_importance_weights_are_the_ratio_of_the_two_slate_probabilities(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    expected = np.exp(oracle_trace.log_propensity - small_log.log_propensity)
    assert np.allclose(weights, expected)


def test_importance_weights_average_to_one_over_a_long_log(small_log, oracle_trace):
    """A property of the estimand: the weights are a likelihood ratio under the logging policy."""
    weights = importance_weights(small_log, oracle_trace, clip=None)
    standard_error = weights.std(ddof=1) / np.sqrt(weights.size)
    assert abs(weights.mean() - 1.0) < 4.0 * standard_error


def test_clipping_never_raises_a_weight(small_log, oracle_trace):
    raw = importance_weights(small_log, oracle_trace, clip=None)
    clipped = importance_weights(small_log, oracle_trace, clip=5.0)
    assert np.all(clipped <= raw + 1e-12)
    assert clipped.max() <= 5.0 + 1e-12


def test_a_non_positive_clip_is_rejected(small_log, oracle_trace):
    with pytest.raises(ValueError, match="clip must be positive"):
        importance_weights(small_log, oracle_trace, clip=0.0)


def test_ips_of_the_incumbent_equals_the_logged_mean_reward(small_log, incumbent_trace):
    weights = importance_weights(small_log, incumbent_trace, clip=None)
    assert ips_value(small_log.reward.astype(float), weights) == pytest.approx(
        small_log.reward.mean()
    )


def test_ips_over_a_subset_uses_only_that_subset():
    assert ips_value(np.array([2.0, 8.0]), np.array([1.0, 1.0]), np.array([0])) == pytest.approx(
        2.0
    )


def test_ips_over_an_empty_index_is_not_a_number(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    value = ips_value(small_log.reward.astype(float), weights, np.array([], dtype=int))
    assert np.isnan(value)


def test_snips_is_bounded_by_the_range_of_the_observed_rewards(small_log, oracle_trace):
    """The property IPS lacks, and the reason SNIPS is the default."""
    weights = importance_weights(small_log, oracle_trace, clip=None)
    reward = small_log.reward.astype(float)
    value = snips_value(reward, weights)
    assert reward.min() <= value <= reward.max()


def test_snips_of_the_incumbent_equals_the_logged_mean_reward(small_log, incumbent_trace):
    weights = importance_weights(small_log, incumbent_trace, clip=None)
    assert snips_value(small_log.reward.astype(float), weights) == pytest.approx(
        small_log.reward.mean()
    )


def test_snips_over_a_subset_uses_only_that_subset():
    reward = np.array([1.0, 9.0, 3.0])
    weight = np.array([1.0, 1.0, 1.0])
    assert snips_value(reward, weight, np.array([0, 2])) == pytest.approx(2.0)


def test_snips_with_weights_that_sum_to_zero_is_not_a_number():
    assert np.isnan(snips_value(np.array([1.0, 2.0]), np.array([0.0, 0.0])))


def test_snips_is_unchanged_when_every_weight_is_scaled_by_the_same_factor(small_log, oracle_trace):
    weights = importance_weights(small_log, oracle_trace, clip=None)
    reward = small_log.reward.astype(float)
    assert snips_value(reward, weights) == pytest.approx(snips_value(reward, 3.7 * weights))


def test_naive_replay_keeps_only_the_sessions_whose_first_slot_the_policy_agrees_with(
    small_log, oracle_trace
):
    matched = naive_match_mask(small_log, oracle_trace)
    assert np.array_equal(matched, oracle_trace.top1 == small_log.slate[:, 0])
    assert naive_match_rate(small_log, oracle_trace) == pytest.approx(matched.mean())


def test_naive_replay_averages_the_reward_over_the_matched_sessions(small_log, oracle_trace):
    matched = naive_match_mask(small_log, oracle_trace)
    reward = small_log.reward.astype(float)
    assert naive_replay_value(reward, matched) == pytest.approx(reward[matched].mean())


def test_naive_replay_over_a_subset_uses_only_that_subset():
    reward = np.array([10.0, 0.0, 4.0, 8.0])
    matched = np.array([True, True, False, True])
    assert naive_replay_value(reward, matched, np.array([0, 3])) == pytest.approx(9.0)


def test_naive_replay_with_nothing_matched_is_not_a_number():
    reward = np.array([1.0, 2.0, 3.0])
    assert np.isnan(naive_replay_value(reward, np.zeros(3, dtype=bool)))


def test_naive_replay_of_the_incumbent_overstates_its_own_value(small_log, incumbent_trace):
    """The headline, as a test. Replaying the deployed policy flatters the deployed policy."""
    matched = naive_match_mask(small_log, incumbent_trace)
    reward = small_log.reward.astype(float)
    assert naive_replay_value(reward, matched) > reward.mean()


def test_the_doubly_robust_estimator_reduces_to_the_direct_model_when_weights_are_zero():
    direct = np.array([0.4, 0.6, 0.5])
    logged = np.array([0.3, 0.3, 0.3])
    reward = np.array([1.0, 0.0, 2.0])
    assert dr_value(reward, np.zeros(3), direct, logged) == pytest.approx(direct.mean())


def test_the_doubly_robust_estimator_reduces_to_ips_when_the_model_predicts_zero():
    reward = np.array([1.0, 0.0, 2.0])
    weight = np.array([1.5, 0.5, 1.0])
    zeros = np.zeros(3)
    assert dr_value(reward, weight, zeros, zeros) == pytest.approx(ips_value(reward, weight))


def test_the_doubly_robust_estimator_over_an_empty_index_is_not_a_number():
    assert np.isnan(dr_value(np.ones(3), np.ones(3), np.ones(3), np.ones(3), np.array([], int)))


def test_the_outcome_model_terms_have_one_entry_per_session(
    small_log, oracle_trace, candidate_rewards
):
    direct, logged = reward_model_terms(small_log, oracle_trace, candidate_rewards)
    assert direct.shape == (small_log.n_sessions,)
    assert logged.shape == (small_log.n_sessions,)


def test_the_outcome_model_term_for_the_incumbent_matches_its_own_served_slates(
    small_log, incumbent_trace, candidate_rewards
):
    """The direct term integrates over the policy's slate distribution; for the logging policy
    that distribution is the one the log was drawn from, so the two agree on average."""
    direct, logged = reward_model_terms(small_log, incumbent_trace, candidate_rewards)
    assert abs(direct.mean() - logged.mean()) < 0.05 * abs(logged.mean())


def test_estimator_inputs_carry_one_row_per_session(small_log, oracle_trace, candidate_rewards):
    inputs = build_estimator_inputs(
        small_log, oracle_trace, clip=20.0, candidate_rewards=candidate_rewards
    )
    assert inputs.n_sessions == small_log.n_sessions
    assert inputs.has_reward_model


def test_estimator_inputs_without_an_outcome_model_omit_the_doubly_robust_estimator(
    small_log, oracle_trace
):
    inputs = build_estimator_inputs(small_log, oracle_trace, clip=20.0)
    assert not inputs.has_reward_model
    estimators, _ = inputs.ratio_estimators()
    assert "dr" not in estimators


def test_the_clipped_share_reports_how_often_the_cap_actually_bound(small_log, oracle_trace):
    loose = build_estimator_inputs(small_log, oracle_trace, clip=1e9)
    tight = build_estimator_inputs(small_log, oracle_trace, clip=1.001)
    assert loose.clipped_share == 0.0
    assert tight.clipped_share > loose.clipped_share


@pytest.mark.parametrize("name", ["naive", "naive_first_slot", "ips", "snips", "dr"])
def test_each_ratio_form_equals_the_direct_estimator_it_stands_for(
    name, small_log, oracle_trace, candidate_rewards
):
    """The identity the bootstrap rests on. Both forms, same arrays, same answer."""
    inputs = build_estimator_inputs(
        small_log, oracle_trace, clip=20.0, candidate_rewards=candidate_rewards
    )
    estimators, _ = inputs.ratio_estimators()
    direct = {
        "naive": lambda: naive_replay_value(inputs.reward, inputs.matched),
        "naive_first_slot": lambda: naive_replay_value(inputs.position0_click, inputs.matched),
        "ips": lambda: ips_value(inputs.reward, inputs.raw_weight),
        "snips": lambda: snips_value(inputs.reward, inputs.weight),
        "dr": lambda: dr_value(
            inputs.reward, inputs.weight, inputs.direct, inputs.logged_prediction
        ),
    }[name]
    assert estimators[name].value() == pytest.approx(direct())


def test_each_estimator_is_paired_with_a_baseline_denominated_in_the_same_unit(
    small_log, oracle_trace, candidate_rewards
):
    """A first slot replay divided by mean session reward is a units error that looks like
    a finding. This pins the pairing that was wrong in the first version."""
    inputs = build_estimator_inputs(
        small_log, oracle_trace, clip=20.0, candidate_rewards=candidate_rewards
    )
    estimators, baselines = inputs.ratio_estimators()
    assert set(estimators) == set(baselines)
    assert baselines["naive"].value() == pytest.approx(inputs.reward.mean())
    assert baselines["naive_first_slot"].value() == pytest.approx(inputs.position0_click.mean())


def test_a_ratio_estimator_rejects_mismatched_terms():
    with pytest.raises(ValueError, match="does not match"):
        RatioEstimator("x", np.ones(3), np.ones(4))


def test_a_ratio_estimator_rejects_terms_that_are_not_one_dimensional():
    with pytest.raises(ValueError, match="one dimensional"):
        RatioEstimator("x", np.ones((2, 2)), np.ones((2, 2)))


def test_a_ratio_estimator_with_a_zero_denominator_is_not_a_number():
    assert np.isnan(RatioEstimator("x", np.ones(3), np.zeros(3)).value())


def test_a_ratio_estimator_can_be_evaluated_over_a_subset():
    estimator = RatioEstimator("x", np.array([1.0, 5.0]), np.array([1.0, 1.0]))
    assert estimator.value(np.array([0])) == pytest.approx(1.0)


def test_the_bootstrap_interval_knows_when_it_excludes_zero():
    assert BootstrapInterval(0.1, 0.05, 0.15, 0.02, 100).excludes_zero
    assert BootstrapInterval(-0.1, -0.15, -0.05, 0.02, 100).excludes_zero
    assert not BootstrapInterval(0.0, -0.05, 0.05, 0.02, 100).excludes_zero
    assert not BootstrapInterval(0.0, float("nan"), 0.05, 0.02, 100).excludes_zero


def test_the_bootstrap_brackets_the_point_estimate(small_log, oracle_trace, candidate_rewards):
    inputs = build_estimator_inputs(
        small_log, oracle_trace, clip=20.0, candidate_rewards=candidate_rewards
    )
    estimators, baselines = inputs.ratio_estimators()
    intervals = bootstrap_ratio_lifts(
        estimators, baselines, resamples=120, rng=np.random.default_rng(1)
    )
    for name, interval in intervals.items():
        assert interval.low <= interval.point <= interval.high, name


def test_the_bootstrap_interval_narrows_as_the_resample_count_has_no_effect_on_the_point(
    small_log, oracle_trace
):
    inputs = build_estimator_inputs(small_log, oracle_trace, clip=20.0)
    estimators, baselines = inputs.ratio_estimators()
    few = bootstrap_ratio_lifts(estimators, baselines, 30, np.random.default_rng(1))
    many = bootstrap_ratio_lifts(estimators, baselines, 200, np.random.default_rng(1))
    assert few["snips"].point == pytest.approx(many["snips"].point)


def test_the_bootstrap_is_reproducible_from_its_seed(small_log, oracle_trace):
    inputs = build_estimator_inputs(small_log, oracle_trace, clip=20.0)
    estimators, baselines = inputs.ratio_estimators()
    first = bootstrap_ratio_lifts(estimators, baselines, 60, np.random.default_rng(5))
    second = bootstrap_ratio_lifts(estimators, baselines, 60, np.random.default_rng(5))
    assert first["snips"].low == second["snips"].low
    assert first["snips"].high == second["snips"].high


def test_the_incumbents_corrected_interval_collapses_onto_zero(small_log, incumbent_trace):
    """Weights are exactly one, so every resample gives exactly the logged mean."""
    inputs = build_estimator_inputs(small_log, incumbent_trace, clip=20.0)
    estimators, baselines = inputs.ratio_estimators()
    interval = bootstrap_ratio_lifts(estimators, baselines, 50, np.random.default_rng(2))["snips"]
    assert interval.low == pytest.approx(0.0, abs=1e-12)
    assert interval.high == pytest.approx(0.0, abs=1e-12)


def test_the_bootstrap_is_unchanged_by_the_chunk_boundary(small_log, oracle_trace):
    """Resamples are drawn in blocks; the block size must not change the draws."""
    inputs = build_estimator_inputs(small_log, oracle_trace, clip=20.0)
    estimators, baselines = inputs.ratio_estimators()
    import replay_gap.estimators.bootstrap as module

    original = module._COUNT_CHUNK
    try:
        module._COUNT_CHUNK = 7
        small = bootstrap_ratio_lifts(estimators, baselines, 21, np.random.default_rng(3))
        module._COUNT_CHUNK = 1000
        large = bootstrap_ratio_lifts(estimators, baselines, 21, np.random.default_rng(3))
    finally:
        module._COUNT_CHUNK = original
    assert small["snips"].low == pytest.approx(large["snips"].low)


def test_the_bootstrap_rejects_an_estimator_with_no_baseline():
    estimator = RatioEstimator("a", np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="no baseline"):
        bootstrap_ratio_lifts({"a": estimator}, {}, 10, np.random.default_rng(0))


def test_the_bootstrap_rejects_an_empty_estimator_set():
    with pytest.raises(ValueError, match="at least one estimator"):
        bootstrap_ratio_lifts({}, {}, 10, np.random.default_rng(0))


def test_the_bootstrap_rejects_fewer_than_two_resamples():
    estimator = RatioEstimator("a", np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_ratio_lifts({"a": estimator}, {"a": estimator}, 1, np.random.default_rng(0))


def test_the_bootstrap_rejects_an_alpha_outside_the_unit_interval():
    estimator = RatioEstimator("a", np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ratio_lifts(
            {"a": estimator}, {"a": estimator}, 10, np.random.default_rng(0), alpha=1.0
        )


def test_the_bootstrap_rejects_arms_covering_different_session_counts():
    a = RatioEstimator("a", np.ones(4), np.ones(4))
    b = RatioEstimator("b", np.ones(5), np.ones(5))
    with pytest.raises(ValueError, match="same sessions"):
        bootstrap_ratio_lifts({"a": a}, {"a": b}, 10, np.random.default_rng(0))


def test_a_degenerate_estimator_yields_an_interval_of_not_a_number():
    zero = RatioEstimator("a", np.ones(4), np.zeros(4))
    base = RatioEstimator("b", np.ones(4), np.ones(4))
    interval = bootstrap_ratio_lifts({"a": zero}, {"a": base}, 10, np.random.default_rng(0))["a"]
    assert np.isnan(interval.low)
    assert np.isnan(interval.standard_error)


def test_a_trace_rejects_a_policy_whose_scores_are_the_wrong_shape(small_log, small_world):
    from replay_gap.policies.base import ScorePolicy

    policy = ScorePolicy("broken", lambda context, _candidates: np.zeros((len(context), 2)))
    with pytest.raises(ValueError, match="logits of shape"):
        trace_policy(policy, small_log)


def test_a_trace_builds_its_own_permutation_index_when_none_is_given(small_log, oracle_policy):
    trace = trace_policy(oracle_policy, small_log)
    assert trace.n_sessions == small_log.n_sessions
    assert trace.marginals.shape == (
        small_log.n_sessions,
        small_log.n_positions,
        small_log.n_candidates,
    )
