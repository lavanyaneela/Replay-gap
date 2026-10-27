"""The world, the deployed policy, the log it produces, and the referee that judges it.

The most important test in this file is the one asserting that the mean reward the log actually
contains converges on the value the referee computes for the policy that produced it. If those
two disagree, either the generator or the referee is wrong, and every number in the repository
is downstream of both.
"""

from __future__ import annotations

import numpy as np
import pytest

from replay_gap.errors import FixtureError
from replay_gap.ranking import logits as pl_logits
from replay_gap.ranking import permutation_index
from replay_gap.simulator.generate import draw_candidate_sets, generate_log
from replay_gap.simulator.ground_truth import (
    per_session_true_value,
    true_value,
    true_value_lift,
)
from replay_gap.simulator.logging_policy import build_logging_policy
from replay_gap.simulator.user_model import build_world, draw_clicks, sigmoid, unit_rows


def test_the_stable_sigmoid_matches_the_naive_one_in_the_range_where_both_work():
    x = np.linspace(-20, 20, 101)
    assert np.allclose(sigmoid(x), 1.0 / (1.0 + np.exp(-x)))


def test_the_stable_sigmoid_does_not_overflow_on_a_large_negative_input():
    assert sigmoid(np.array([-800.0]))[0] == pytest.approx(0.0)
    assert sigmoid(np.array([800.0]))[0] == pytest.approx(1.0)


def test_unit_rows_gives_every_row_length_one(rng):
    result = unit_rows(rng.normal(size=(9, 4)))
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_unit_rows_leaves_a_zero_row_finite_rather_than_dividing_by_zero():
    assert np.all(np.isfinite(unit_rows(np.zeros((2, 3)))))


def test_a_negative_segment_jitter_is_rejected():
    from replay_gap.config import Config, WorldConfig
    from replay_gap.errors import ConfigError

    with pytest.raises(ConfigError, match="segment_jitter"):
        Config(world=WorldConfig(segment_jitter=-0.1))


def test_a_negative_clone_noise_is_rejected():
    from replay_gap.config import Config, PoliciesConfig
    from replay_gap.errors import ConfigError

    with pytest.raises(ConfigError, match="clone_noise"):
        Config(policies=PoliciesConfig(clone_noise=-0.1))


def test_a_wider_segment_jitter_makes_users_less_like_their_segment(small_config):
    """The knob that decides whether a ranker trained on the log has anything real to find."""
    from replay_gap.simulator.user_model import build_world

    def within_segment_similarity(jitter: float) -> float:
        world = build_world(
            n_items=20,
            n_item_features=5,
            n_segments=3,
            n_contexts=180,
            relevance_scale=2.0,
            quality_spread=0.5,
            segment_jitter=jitter,
            examination=np.array([1.0]),
            rng=np.random.default_rng(1),
        )
        same = [
            float(world.context_features[a] @ world.context_features[b])
            for a in range(60)
            for b in range(a + 1, 60)
            if world.context_segment[a] == world.context_segment[b]
        ]
        return float(np.mean(same))

    assert within_segment_similarity(0.1) > within_segment_similarity(1.2)


def test_the_world_reports_its_own_shape(small_world, small_config):
    assert small_world.n_items == small_config.world.n_items
    assert small_world.n_contexts == small_config.world.n_contexts
    assert small_world.n_features == small_config.world.n_item_features
    assert small_world.n_positions == small_config.slate.n_positions
    assert small_world.n_segments <= small_config.world.n_segments


def test_relevance_is_a_probability_for_every_context_and_item(small_world):
    contexts = np.arange(small_world.n_contexts)
    items = np.tile(np.arange(small_world.n_items), (small_world.n_contexts, 1))
    relevance = small_world.relevance(contexts, items)
    assert relevance.min() > 0.0
    assert relevance.max() < 1.0


def test_expected_click_value_is_the_same_quantity_as_relevance(small_world):
    contexts = np.arange(5)
    items = np.tile(np.arange(4), (5, 1))
    assert np.array_equal(
        small_world.relevance(contexts, items), small_world.expected_click_value(contexts, items)
    )


def test_relevance_rejects_a_context_vector_that_is_not_one_dimensional(small_world):
    with pytest.raises(ValueError, match="one dimensional"):
        small_world.relevance_logit(np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), np.int64))


def test_relevance_rejects_an_item_array_that_is_not_two_dimensional(small_world):
    with pytest.raises(ValueError, match="two dimensional"):
        small_world.relevance_logit(np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64))


def test_relevance_rejects_a_mismatch_between_contexts_and_item_rows(small_world):
    with pytest.raises(ValueError, match="rows"):
        small_world.relevance_logit(np.zeros(3, dtype=np.int64), np.zeros((2, 4), dtype=np.int64))


def test_users_in_the_same_segment_are_more_alike_than_users_across_segments(small_world):
    same = []
    different = []
    for a in range(60):
        for b in range(a + 1, 60):
            similarity = float(small_world.context_features[a] @ small_world.context_features[b])
            if small_world.context_segment[a] == small_world.context_segment[b]:
                same.append(similarity)
            else:
                different.append(similarity)
    assert np.mean(same) > np.mean(different)


def test_click_probability_falls_with_position_for_a_fixed_relevance(small_world):
    relevance = np.full((40000, 3), 0.5)
    clicks = draw_clicks(relevance, small_world.examination, np.random.default_rng(1))
    rates = clicks.mean(axis=0)
    assert rates[0] > rates[1] > rates[2]


def test_click_rates_match_examination_times_relevance(small_world):
    relevance = np.full((60000, 3), 0.4)
    clicks = draw_clicks(relevance, small_world.examination, np.random.default_rng(2))
    assert np.allclose(clicks.mean(axis=0), 0.4 * small_world.examination, atol=0.01)


def test_draw_clicks_rejects_a_relevance_array_of_the_wrong_width(small_world):
    with pytest.raises(ValueError, match="columns"):
        draw_clicks(np.zeros((4, 5)), small_world.examination, np.random.default_rng(0))


def test_draw_clicks_rejects_a_relevance_array_that_is_not_two_dimensional(small_world):
    with pytest.raises(ValueError, match="two dimensional"):
        draw_clicks(np.zeros(4), small_world.examination, np.random.default_rng(0))


def test_building_a_world_with_no_segments_is_rejected():
    with pytest.raises(ValueError, match="n_segments"):
        build_world(10, 3, 0, 5, 2.0, 0.5, 0.45, np.array([1.0]), np.random.default_rng(0))


def test_candidate_sets_hold_distinct_items(rng):
    candidates = draw_candidate_sets(500, 40, 6, rng)
    assert candidates.shape == (500, 6)
    assert all(len(set(row)) == 6 for row in candidates)


def test_candidate_sets_cannot_be_larger_than_the_catalogue(rng):
    with pytest.raises(ValueError, match="more candidates"):
        draw_candidate_sets(5, 4, 6, rng)


def test_candidate_sets_cover_the_catalogue_roughly_uniformly(rng):
    candidates = draw_candidate_sets(20000, 20, 5, rng)
    counts = np.bincount(candidates.reshape(-1), minlength=20)
    assert counts.std() / counts.mean() < 0.05


def test_the_logging_policy_rejects_a_negative_noise_scale(small_world):
    with pytest.raises(ValueError, match="noise scales"):
        build_logging_policy(small_world, -1.0, 0.1, 1.0, np.random.default_rng(0))


def test_the_logging_policy_agrees_with_true_relevance_more_often_than_chance(small_fixture):
    """A deployed ranker that was no better than random would make the whole setup pointless."""
    world, policy, log = small_fixture.world, small_fixture.logging_policy, small_fixture.log
    deployed = policy.scores(world, log.context_id, log.candidates)
    truth = world.relevance_logit(log.context_id, log.candidates)
    agreement = (deployed.argmax(axis=1) == truth.argmax(axis=1)).mean()
    assert 0.2 < agreement < 0.9


def test_recorded_propensities_match_the_policy_that_recorded_them(small_fixture):
    world, policy, log = small_fixture.world, small_fixture.logging_policy, small_fixture.log
    from replay_gap.ranking import slate_log_propensity

    recomputed = slate_log_propensity(
        policy.logits(world, log.context_id, log.candidates), log.slate
    )
    assert np.allclose(recomputed, log.log_propensity)


def test_every_recorded_propensity_is_a_probability(small_log):
    assert small_log.propensity.min() > 0.0
    assert small_log.propensity.max() < 1.0


def test_the_log_reports_its_own_shape(small_log, small_config):
    assert small_log.n_sessions == small_config.log.n_sessions
    assert small_log.n_candidates == small_config.slate.n_candidates
    assert small_log.n_positions == small_config.slate.n_positions


def test_reward_equals_the_clicks_the_log_recorded(small_log):
    assert np.array_equal(small_log.reward, small_log.clicks.sum(axis=1))


def test_the_items_shown_are_drawn_from_the_session_candidate_set(small_log):
    shown = small_log.slate_items()
    for row in range(small_log.n_sessions):
        assert set(shown[row]).issubset(set(small_log.candidates[row]))


def test_a_subset_of_the_log_is_itself_a_valid_log(small_log):
    subset = small_log.subset(np.arange(10))
    assert subset.n_sessions == 10
    assert np.array_equal(subset.reward, small_log.reward[:10])


def test_generating_a_log_with_no_sessions_is_rejected(small_fixture, small_config):
    with pytest.raises(ValueError, match="n_sessions"):
        generate_log(
            small_fixture.world,
            small_fixture.logging_policy,
            0,
            small_config.slate.n_candidates,
            np.random.default_rng(0),
        )


def test_a_log_whose_columns_disagree_on_length_is_refused(small_log):
    from replay_gap.simulator.generate import InteractionLog

    with pytest.raises(FixtureError, match="rows"):
        InteractionLog(
            session_id=small_log.session_id,
            context_id=small_log.context_id[:-1],
            candidates=small_log.candidates,
            slate=small_log.slate,
            log_propensity=small_log.log_propensity,
            clicks=small_log.clicks,
            reward=small_log.reward,
        )


def test_a_log_whose_slate_names_a_slot_outside_the_candidate_set_is_refused(small_log):
    from replay_gap.simulator.generate import InteractionLog

    broken = small_log.slate.copy()
    broken[0, 0] = small_log.n_candidates + 5
    with pytest.raises(FixtureError, match="outside the candidate set"):
        InteractionLog(
            session_id=small_log.session_id,
            context_id=small_log.context_id,
            candidates=small_log.candidates,
            slate=broken,
            log_propensity=small_log.log_propensity,
            clicks=small_log.clicks,
            reward=small_log.reward,
        )


def test_a_log_whose_clicks_and_slate_disagree_on_width_is_refused(small_log):
    from replay_gap.simulator.generate import InteractionLog

    with pytest.raises(FixtureError, match="same number of positions"):
        InteractionLog(
            session_id=small_log.session_id,
            context_id=small_log.context_id,
            candidates=small_log.candidates,
            slate=small_log.slate,
            log_propensity=small_log.log_propensity,
            clicks=small_log.clicks[:, :2],
            reward=small_log.reward,
        )


def test_the_same_seed_produces_a_byte_identical_log(small_config):
    from replay_gap.fixtures import build_fixture

    first = build_fixture(small_config)
    second = build_fixture(small_config)
    assert first.log.content_hash() == second.log.content_hash()


def test_a_different_seed_produces_a_different_log(small_config):
    from dataclasses import replace

    from replay_gap.fixtures import build_fixture

    first = build_fixture(small_config)
    second = build_fixture(replace(small_config, seed=small_config.seed + 1))
    assert first.log.content_hash() != second.log.content_hash()


def test_the_logged_mean_reward_converges_on_the_referees_value_for_the_logging_policy(
    small_fixture, small_perms
):
    """The generator and the referee are independent implementations of the same expectation.

    If this fails, one of them is wrong and nothing else in the repository can be trusted.
    """
    world, policy, log = small_fixture.world, small_fixture.logging_policy, small_fixture.log
    lg = policy.logits(world, log.context_id, log.candidates)
    exact = true_value(world, log, lg, small_perms)
    observed = float(log.reward.mean())
    standard_error = float(log.reward.std(ddof=1)) / np.sqrt(log.n_sessions)
    assert abs(observed - exact) < 4.0 * standard_error


def test_the_referee_rates_an_oracle_above_the_deployed_policy_above_a_uniform_one(
    small_fixture, small_perms
):
    world, policy, log = small_fixture.world, small_fixture.logging_policy, small_fixture.log
    deployed_logits = policy.logits(world, log.context_id, log.candidates)
    deployed = true_value(world, log, deployed_logits, small_perms)
    oracle_logits = pl_logits(world.relevance_logit(log.context_id, log.candidates), 1.0)
    oracle = true_value(world, log, oracle_logits, small_perms)
    uniform = true_value(
        world, log, pl_logits(np.zeros((log.n_sessions, log.n_candidates)), 1.0), small_perms
    )
    assert uniform < deployed < oracle


def test_the_referee_rejects_logits_of_the_wrong_shape(small_fixture):
    world, log = small_fixture.world, small_fixture.log
    with pytest.raises(ValueError, match="policy logits"):
        per_session_true_value(world, log, np.zeros((log.n_sessions, 2)))


def test_the_referee_builds_its_own_permutation_index_when_none_is_supplied(small_fixture):
    world, log = small_fixture.world, small_fixture.log
    lg = small_fixture.logging_policy.logits(world, log.context_id, log.candidates)
    with_index = true_value(world, log, lg, permutation_index(log.n_candidates, log.n_positions))
    without_index = true_value(world, log, lg)
    assert with_index == pytest.approx(without_index)


def test_relative_lift_is_the_ratio_minus_one():
    assert true_value_lift(1.1, 1.0) == pytest.approx(0.1)


def test_relative_lift_against_a_zero_baseline_is_rejected():
    with pytest.raises(ValueError, match="baseline value"):
        true_value_lift(1.0, 0.0)
