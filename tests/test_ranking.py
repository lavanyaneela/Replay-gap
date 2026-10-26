"""Properties of the Plackett-Luce layer. These underwrite every other number in the repository.

If ``permutation_probabilities`` does not sum to one, every value this repository publishes is
wrong by an amount nothing else would reveal, so the exactness properties are tested directly
rather than through anything that consumes them.
"""

from __future__ import annotations

import numpy as np
import pytest

from replay_gap.ranking import (
    expected_slate_value,
    logits,
    permutation_index,
    permutation_probabilities,
    position_marginals,
    sample_slates,
    slate_log_propensity,
    standardise,
    top1_choice,
)


def test_standardising_leaves_each_row_with_zero_mean_and_unit_spread(rng):
    scores = rng.normal(loc=7.0, scale=3.0, size=(20, 8))
    result = standardise(scores)
    assert np.allclose(result.mean(axis=1), 0.0, atol=1e-12)
    assert np.allclose(result.std(axis=1), 1.0, atol=1e-12)


def test_standardising_a_row_of_equal_scores_gives_zeros_rather_than_dividing_by_zero():
    result = standardise(np.full((3, 5), 2.5))
    assert np.array_equal(result, np.zeros((3, 5)))


def test_standardising_preserves_the_ordering_within_every_row(rng):
    scores = rng.normal(size=(12, 7))
    result = standardise(scores)
    assert np.array_equal(np.argsort(scores, axis=1), np.argsort(result, axis=1))


def test_standardising_rejects_scores_that_are_not_a_matrix():
    with pytest.raises(ValueError, match="two dimensional"):
        standardise(np.arange(5))


def test_a_lower_temperature_produces_a_sharper_distribution_over_slates():
    scores = np.array([[3.0, 1.0, 0.0, -1.0, -2.0]])
    perms = permutation_index(5, 2)
    sharp = permutation_probabilities(logits(scores, 0.3), perms).max()
    flat = permutation_probabilities(logits(scores, 2.0), perms).max()
    assert sharp > flat


def test_a_non_positive_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature must be positive"):
        logits(np.zeros((2, 3)), 0.0)


def test_the_permutation_index_holds_every_ordered_slate_exactly_once():
    perms = permutation_index(5, 3)
    assert perms.shape == (5 * 4 * 3, 3)
    assert len({tuple(row) for row in perms}) == perms.shape[0]
    assert all(len(set(row)) == 3 for row in perms)


def test_the_permutation_index_refuses_more_positions_than_candidates():
    with pytest.raises(ValueError, match="cannot fill"):
        permutation_index(2, 3)


def test_permutation_probabilities_sum_to_one_for_every_row(rng):
    perms = permutation_index(6, 3)
    lg = logits(rng.normal(size=(15, 6)), 0.8)
    assert np.allclose(permutation_probabilities(lg, perms).sum(axis=1), 1.0)


def test_permutation_probabilities_are_uniform_when_every_score_is_equal():
    perms = permutation_index(5, 2)
    lg = logits(np.zeros((4, 5)), 1.0)
    probabilities = permutation_probabilities(lg, perms)
    assert np.allclose(probabilities, 1.0 / perms.shape[0])


def test_permutation_probabilities_reject_a_one_dimensional_logit_array():
    with pytest.raises(ValueError, match="two dimensional"):
        permutation_probabilities(np.zeros(4), permutation_index(4, 2))


def test_permutation_probabilities_reject_a_permutation_index_that_is_too_wide():
    with pytest.raises(ValueError, match="names more positions"):
        permutation_probabilities(np.zeros((2, 3)), permutation_index(5, 4))


def test_the_log_propensity_of_a_slate_matches_its_enumerated_probability(rng):
    perms = permutation_index(6, 3)
    lg = logits(rng.normal(size=(9, 6)), 1.0)
    enumerated = permutation_probabilities(lg, perms)
    chosen = rng.integers(0, perms.shape[0], size=9)
    direct = np.exp(slate_log_propensity(lg, perms[chosen]))
    assert np.allclose(direct, enumerated[np.arange(9), chosen])


def test_log_propensity_rejects_a_slate_with_a_different_number_of_rows():
    with pytest.raises(ValueError, match="rows"):
        slate_log_propensity(np.zeros((3, 4)), np.zeros((2, 2), dtype=np.int64))


def test_log_propensity_rejects_inputs_that_are_not_matrices():
    with pytest.raises(ValueError, match="two dimensional"):
        slate_log_propensity(np.zeros(4), np.zeros((1, 2), dtype=np.int64))


def test_sampled_slates_never_repeat_an_item_within_one_slate(rng):
    lg = logits(rng.normal(size=(200, 7)), 1.0)
    slates = sample_slates(lg, 3, rng)
    assert all(len(set(row)) == 3 for row in slates)


def test_sampled_slate_frequencies_converge_on_the_enumerated_probabilities():
    """Gumbel top-k is a Plackett-Luce draw, which is the claim the propensities depend on."""
    scores = np.array([[2.0, 1.0, 0.5, -0.5]])
    perms = permutation_index(4, 2)
    lg = logits(scores, 1.0)
    expected = permutation_probabilities(lg, perms)[0]
    draws = sample_slates(np.repeat(lg, 60000, axis=0), 2, np.random.default_rng(3))
    lookup = {tuple(row): i for i, row in enumerate(perms)}
    observed = np.zeros(perms.shape[0])
    for row in draws:
        observed[lookup[tuple(row)]] += 1
    observed /= observed.sum()
    assert np.abs(observed - expected).max() < 0.01


def test_position_marginals_sum_to_one_within_each_position(rng):
    perms = permutation_index(6, 3)
    lg = logits(rng.normal(size=(11, 6)), 1.0)
    assert np.allclose(position_marginals(lg, perms).sum(axis=2), 1.0)


def test_position_marginals_agree_with_summing_over_every_permutation(rng):
    perms = permutation_index(5, 3)
    lg = logits(rng.normal(size=(7, 5)), 1.1)
    probabilities = permutation_probabilities(lg, perms)
    marginals = position_marginals(lg, perms)
    for position in range(3):
        expected = np.zeros((7, 5))
        for row, perm in enumerate(perms):
            expected[:, perm[position]] += probabilities[:, row]
        assert np.allclose(marginals[:, position, :], expected)


def test_the_first_position_marginal_is_the_plain_softmax_of_the_logits(rng):
    perms = permutation_index(5, 2)
    lg = logits(rng.normal(size=(6, 5)), 1.0)
    weights = np.exp(lg - lg.max(axis=1, keepdims=True))
    softmax = weights / weights.sum(axis=1, keepdims=True)
    assert np.allclose(position_marginals(lg, perms)[:, 0, :], softmax)


def test_expected_slate_value_agrees_with_weighting_the_position_marginals(rng):
    perms = permutation_index(6, 3)
    lg = logits(rng.normal(size=(13, 6)), 1.0)
    values = rng.uniform(0.05, 0.95, size=(13, 6))
    examination = np.array([1.0, 0.5, 0.25])
    enumerated = expected_slate_value(lg, values, examination, perms)
    marginals = position_marginals(lg, perms)
    assert np.allclose(enumerated, (marginals * values[:, None, :]).sum(axis=2) @ examination)


def test_expected_slate_value_matches_a_monte_carlo_average_of_sampled_slates():
    """The referee's exactness claim, checked against the sampling it replaces."""
    perms = permutation_index(5, 3)
    lg = logits(np.array([[1.5, 0.8, 0.0, -0.4, -1.2]]), 1.0)
    values = np.array([[0.9, 0.7, 0.5, 0.3, 0.1]])
    examination = np.array([1.0, 0.5, 0.25])
    exact = expected_slate_value(lg, values, examination, perms)[0]
    draws = sample_slates(np.repeat(lg, 200000, axis=0), 3, np.random.default_rng(8))
    sampled = (examination * values[0][draws]).sum(axis=1).mean()
    assert abs(exact - sampled) < 0.002


def test_expected_slate_value_is_identical_whatever_chunk_size_it_runs_in(rng):
    perms = permutation_index(5, 2)
    lg = logits(rng.normal(size=(37, 5)), 1.0)
    values = rng.uniform(size=(37, 5))
    examination = np.array([1.0, 0.5])
    one_pass = expected_slate_value(lg, values, examination, perms, chunk_rows=1000)
    chunked = expected_slate_value(lg, values, examination, perms, chunk_rows=4)
    assert np.allclose(one_pass, chunked)


def test_expected_slate_value_rejects_a_value_matrix_of_the_wrong_shape():
    perms = permutation_index(4, 2)
    with pytest.raises(ValueError, match="does not match"):
        expected_slate_value(np.zeros((3, 4)), np.zeros((3, 5)), np.ones(2), perms)


def test_expected_slate_value_rejects_an_examination_curve_of_the_wrong_length():
    perms = permutation_index(4, 2)
    with pytest.raises(ValueError, match="examination length"):
        expected_slate_value(np.zeros((3, 4)), np.zeros((3, 4)), np.ones(3), perms)


def test_expected_slate_value_rejects_a_chunk_size_below_one():
    perms = permutation_index(4, 2)
    with pytest.raises(ValueError, match="chunk_rows"):
        expected_slate_value(np.zeros((2, 4)), np.zeros((2, 4)), np.ones(2), perms, chunk_rows=0)


def test_a_policy_that_prefers_better_items_has_a_higher_expected_value():
    perms = permutation_index(4, 2)
    values = np.array([[0.9, 0.6, 0.3, 0.1]])
    examination = np.array([1.0, 0.5])
    good = expected_slate_value(logits(values, 0.3), values, examination, perms)[0]
    bad = expected_slate_value(logits(-values, 0.3), values, examination, perms)[0]
    assert good > bad


def test_top1_breaks_ties_towards_the_lowest_index_rather_than_by_hash_order():
    scores = np.array([[1.0, 1.0, 0.5], [0.2, 0.9, 0.9]])
    assert np.array_equal(top1_choice(scores), np.array([0, 1]))


def test_top1_picks_the_highest_scoring_candidate(rng):
    scores = rng.normal(size=(25, 9))
    assert np.array_equal(top1_choice(scores), scores.argmax(axis=1))
