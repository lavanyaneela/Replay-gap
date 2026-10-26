"""The candidate policies, and the two properties the experiments depend on.

``incumbent`` must have a true lift of exactly zero over the logging policy, because it is the
logging policy. ``contrarian_lift`` must be genuinely better while agreeing with the logging
policy on the first slot in a small minority of sessions. Without both, E1 is measuring
something other than what it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest

from replay_gap.policies.base import ScorePolicy
from replay_gap.policies.baselines import (
    BlendPolicy,
    ContrarianPolicy,
    IncumbentPolicy,
    OraclePolicy,
    PopularityPolicy,
    UniformPolicy,
    _standardise_scores,
    build_baseline_policies,
    click_counts_from_log,
)
from replay_gap.policies.lightgbm_ranker import (
    LightGBMRankerPolicy,
    build_features,
    build_training_labels,
    train_lightgbm_ranker,
)
from replay_gap.simulator.ground_truth import true_value


@pytest.fixture(scope="module")
def baseline_set(small_fixture, small_config):
    counts = click_counts_from_log(
        small_fixture.world.n_items, small_fixture.log.slate_items(), small_fixture.log.clicks
    )
    return build_baseline_policies(
        small_fixture.world,
        small_fixture.logging_policy,
        counts,
        small_config.policies.temperature,
        small_config.policies.sharp_temperature,
    )


def test_every_baseline_scores_one_number_per_candidate(baseline_set, small_log):
    for policy in baseline_set.values():
        scores = policy.raw_scores(small_log.context_id, small_log.candidates)
        assert scores.shape == small_log.candidates.shape


def test_every_baseline_names_itself_consistently_with_its_key(baseline_set):
    for key, policy in baseline_set.items():
        assert policy.name == key


def test_the_incumbent_scores_exactly_what_the_logging_policy_scores(small_fixture):
    policy = IncumbentPolicy(small_fixture.world, small_fixture.logging_policy)
    log = small_fixture.log
    assert np.allclose(
        policy.raw_scores(log.context_id, log.candidates),
        small_fixture.logging_policy.scores(small_fixture.world, log.context_id, log.candidates),
    )


def test_the_incumbent_has_a_true_lift_of_exactly_zero_over_the_logging_policy(
    small_fixture, small_perms
):
    """E1 and the README both lean on this being exact rather than approximately zero."""
    world, log = small_fixture.world, small_fixture.log
    incumbent = IncumbentPolicy(world, small_fixture.logging_policy)
    deployed = small_fixture.logging_policy.logits(world, log.context_id, log.candidates)
    lg = incumbent.logits(log.context_id, log.candidates)
    assert true_value(world, log, lg, small_perms) == pytest.approx(
        true_value(world, log, deployed, small_perms)
    )


def test_the_contrarian_is_genuinely_better_than_the_incumbent(small_fixture, small_perms):
    world, log = small_fixture.world, small_fixture.log
    deployed = true_value(
        world,
        log,
        small_fixture.logging_policy.logits(world, log.context_id, log.candidates),
        small_perms,
    )
    contrarian = ContrarianPolicy(world, small_fixture.logging_policy)
    value = true_value(world, log, contrarian.logits(log.context_id, log.candidates), small_perms)
    assert value > deployed


def test_the_contrarian_disagrees_with_the_served_first_slot_most_of_the_time(small_fixture):
    log = small_fixture.log
    contrarian = ContrarianPolicy(small_fixture.world, small_fixture.logging_policy)
    agreement = (contrarian.top1(log.context_id, log.candidates) == log.slate[:, 0]).mean()
    assert agreement < 0.2


def test_the_sharp_contrarian_ranks_identically_to_the_ordinary_one(small_fixture, small_config):
    """Temperature changes how greedily a policy is served, never what it prefers."""
    log = small_fixture.log
    mild = ContrarianPolicy(small_fixture.world, small_fixture.logging_policy, 1.0)
    sharp = ContrarianPolicy(
        small_fixture.world, small_fixture.logging_policy, small_config.policies.sharp_temperature
    )
    assert np.array_equal(
        mild.top1(log.context_id, log.candidates), sharp.top1(log.context_id, log.candidates)
    )


def test_the_sharp_contrarian_puts_more_mass_on_its_favourite_slate(small_fixture, small_config):
    from replay_gap.ranking import permutation_index, permutation_probabilities

    log = small_fixture.log.subset(np.arange(50))
    mild = ContrarianPolicy(small_fixture.world, small_fixture.logging_policy, 1.0)
    sharp = ContrarianPolicy(
        small_fixture.world, small_fixture.logging_policy, small_config.policies.sharp_temperature
    )
    perms = permutation_index(log.n_candidates, log.n_positions)
    mild_max = permutation_probabilities(mild.logits(log.context_id, log.candidates), perms).max(1)
    sharp_max = permutation_probabilities(sharp.logits(log.context_id, log.candidates), perms).max(
        1
    )
    assert np.all(sharp_max > mild_max)


def test_the_oracle_ranks_by_true_relevance(small_fixture):
    log = small_fixture.log
    oracle = OraclePolicy(small_fixture.world)
    scores = oracle.raw_scores(log.context_id, log.candidates)
    truth = small_fixture.world.relevance(log.context_id, log.candidates)
    assert np.array_equal(scores.argmax(axis=1), truth.argmax(axis=1))


def test_the_uniform_policy_gives_every_candidate_the_same_score(small_log):
    scores = UniformPolicy().raw_scores(small_log.context_id, small_log.candidates)
    assert np.all(scores == 0.0)


def test_the_uniform_policy_induces_a_flat_distribution_over_slates(small_log, small_perms):
    from replay_gap.ranking import permutation_probabilities

    log = small_log.subset(np.arange(5))
    lg = UniformPolicy().logits(log.context_id, log.candidates)
    probabilities = permutation_probabilities(lg, small_perms)
    assert np.allclose(probabilities, 1.0 / small_perms.shape[0])


def test_popularity_ignores_the_user_and_ranks_the_catalogue_globally(small_fixture):
    log = small_fixture.log
    counts = click_counts_from_log(small_fixture.world.n_items, log.slate_items(), log.clicks)
    policy = PopularityPolicy(counts)
    scores = policy.raw_scores(log.context_id, log.candidates)
    expected = np.log1p(counts)[log.candidates]
    assert np.allclose(scores, expected)


def test_click_counts_recover_exactly_the_clicks_the_log_recorded(small_fixture):
    log = small_fixture.log
    counts = click_counts_from_log(small_fixture.world.n_items, log.slate_items(), log.clicks)
    assert counts.sum() == pytest.approx(float(log.clicks.sum()))


def test_a_blend_with_only_the_incumbent_weight_reproduces_the_incumbent_ordering(small_fixture):
    log = small_fixture.log
    blend = BlendPolicy(
        small_fixture.world, small_fixture.logging_policy, 0.0, 1.0, 0.0, noise_seed=1
    )
    incumbent = IncumbentPolicy(small_fixture.world, small_fixture.logging_policy)
    assert np.array_equal(
        blend.top1(log.context_id, log.candidates), incumbent.top1(log.context_id, log.candidates)
    )


def test_a_blend_with_only_the_truth_weight_reproduces_the_oracle_ordering(small_fixture):
    log = small_fixture.log
    blend = BlendPolicy(
        small_fixture.world, small_fixture.logging_policy, 1.0, 0.0, 0.0, noise_seed=1
    )
    oracle = OraclePolicy(small_fixture.world)
    assert np.array_equal(
        blend.top1(log.context_id, log.candidates), oracle.top1(log.context_id, log.candidates)
    )


def test_a_blend_is_a_fixed_function_rather_than_re_randomising_on_each_call(small_fixture):
    """A policy that scores differently each time it is asked is not a policy."""
    log = small_fixture.log
    blend = BlendPolicy(
        small_fixture.world, small_fixture.logging_policy, 0.3, 0.3, 0.4, noise_seed=99
    )
    first = blend.raw_scores(log.context_id, log.candidates)
    second = blend.raw_scores(log.context_id, log.candidates)
    assert np.array_equal(first, second)


def test_two_blends_with_the_same_noise_seed_are_identical(small_fixture):
    log = small_fixture.log
    args = (small_fixture.world, small_fixture.logging_policy, 0.2, 0.3, 0.5)
    a = BlendPolicy(*args, noise_seed=7)
    b = BlendPolicy(*args, noise_seed=7)
    assert np.array_equal(
        a.raw_scores(log.context_id, log.candidates), b.raw_scores(log.context_id, log.candidates)
    )


def test_two_blends_with_different_noise_seeds_differ(small_fixture):
    log = small_fixture.log
    args = (small_fixture.world, small_fixture.logging_policy, 0.2, 0.3, 0.5)
    a = BlendPolicy(*args, noise_seed=7)
    b = BlendPolicy(*args, noise_seed=8)
    assert not np.array_equal(
        a.raw_scores(log.context_id, log.candidates), b.raw_scores(log.context_id, log.candidates)
    )


def test_a_blend_with_a_negative_weight_is_rejected(small_fixture):
    with pytest.raises(ValueError, match="must not be negative"):
        BlendPolicy(small_fixture.world, small_fixture.logging_policy, -1.0, 1.0, 0.0, 1)


def test_a_blend_with_no_positive_weight_is_rejected(small_fixture):
    with pytest.raises(ValueError, match="at least one blend weight"):
        BlendPolicy(small_fixture.world, small_fixture.logging_policy, 0.0, 0.0, 0.0, 1)


def test_standardising_a_constant_score_matrix_gives_zeros():
    assert np.array_equal(_standardise_scores(np.full((3, 4), 9.0)), np.zeros((3, 4)))


def test_a_score_policy_wraps_any_callable(small_log):
    policy = ScorePolicy("custom", lambda _context, cand: np.zeros(cand.shape))
    assert policy.name == "custom"
    assert policy.raw_scores(small_log.context_id, small_log.candidates).shape == (
        small_log.candidates.shape
    )


def test_a_score_policy_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature must be positive"):
        ScorePolicy("custom", lambda _context, cand: np.zeros(cand.shape), temperature=0.0)


def test_ranker_features_hold_the_context_the_item_and_their_product(small_fixture):
    world, log = small_fixture.world, small_fixture.log
    features = build_features(world, log.context_id[:3], log.candidates[:3])
    assert features.shape == (3 * log.n_candidates, 3 * world.n_features)
    first_item = log.candidates[0, 0]
    expected = world.context_features[log.context_id[0]] * world.item_features[first_item]
    assert np.allclose(features[0, 2 * world.n_features :], expected)


def test_training_labels_mark_a_clicked_impression_and_nothing_else(small_fixture):
    log = small_fixture.log
    labels = build_training_labels(log).reshape(log.n_sessions, log.n_candidates)
    assert labels.sum() == int((log.clicks > 0).sum())
    for session in range(40):
        clicked_slots = log.slate[session][log.clicks[session] > 0]
        assert set(np.flatnonzero(labels[session])) == set(clicked_slots.tolist())


def test_an_un_impressed_candidate_is_labelled_a_negative(small_fixture):
    """The lie every implicit feedback ranker is told, pinned so it stays deliberate."""
    log = small_fixture.log
    labels = build_training_labels(log).reshape(log.n_sessions, log.n_candidates)
    for session in range(40):
        for slot in range(log.n_candidates):
            if slot not in set(log.slate[session].tolist()):
                assert labels[session, slot] == 0


@pytest.mark.slow
def test_the_trained_ranker_beats_the_uniform_policy(small_fixture, small_config, small_perms):
    world, log = small_fixture.world, small_fixture.log
    model = train_lightgbm_ranker(world, log, small_config.ranker)
    ranker = LightGBMRankerPolicy(world, model)
    value = true_value(world, log, ranker.logits(log.context_id, log.candidates), small_perms)
    uniform = true_value(
        world, log, UniformPolicy().logits(log.context_id, log.candidates), small_perms
    )
    assert value > uniform


@pytest.mark.slow
def test_training_the_ranker_twice_gives_identical_scores(small_fixture, small_config):
    """LightGBM is pinned to one thread in deterministic mode, so a published number holds."""
    world, log = small_fixture.world, small_fixture.log
    first = LightGBMRankerPolicy(world, train_lightgbm_ranker(world, log, small_config.ranker))
    second = LightGBMRankerPolicy(world, train_lightgbm_ranker(world, log, small_config.ranker))
    assert np.array_equal(
        first.raw_scores(log.context_id[:50], log.candidates[:50]),
        second.raw_scores(log.context_id[:50], log.candidates[:50]),
    )
