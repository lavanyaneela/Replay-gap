"""Plackett-Luce over ordered slates: sampling, exact propensities, and exact policy values.

This module is the reason the repository can have a referee at all. Both the logging policy and
every candidate policy are Plackett-Luce distributions over the ordered top-k of a candidate set,
so a logged slate's propensity is a closed form rather than an estimate, and a policy's expected
reward can be summed over every ordered slate instead of sampled. With ten candidates and three
positions there are 720 ordered slates, which is small enough to enumerate for every session in a
fifty thousand session log.

Scores arriving here are standardised per session before the temperature is applied. Without that
step a temperature means something different for each policy, because a gradient boosted model and
a dot product do not produce scores on the same scale, and the comparison between policies would
silently be a comparison between score scales.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations

import numpy as np

_STD_FLOOR = 1e-12


def standardise(scores: np.ndarray) -> np.ndarray:
    """Centre and scale each row of scores, so a temperature means the same thing everywhere.

    A row whose scores are all equal has no ordering information, so it becomes all zeros, which
    the Plackett-Luce layer turns into a uniform distribution. That is the honest reading of a
    policy that cannot tell the candidates apart.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be two dimensional, got shape {scores.shape}")
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return np.where(spread < _STD_FLOOR, 0.0, centred / np.maximum(spread, _STD_FLOOR))


def logits(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Standardised scores divided by the temperature. Lower temperature, sharper policy."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return standardise(scores) / temperature


def permutation_index(n_candidates: int, n_positions: int) -> np.ndarray:
    """Every ordered slate, as an array of shape (n_permutations, n_positions).

    Built once and reused. For ten candidates and three positions this is 720 rows, so the cost
    of being exact instead of sampling is a 720 column intermediate per session.
    """
    if n_positions > n_candidates:
        raise ValueError(f"cannot fill {n_positions} positions from {n_candidates} candidates")
    return np.array(list(permutations(range(n_candidates), n_positions)), dtype=np.int64)


def slate_log_propensity(lg: np.ndarray, slate: np.ndarray) -> np.ndarray:
    """Log probability of each observed ordered slate under the given logits.

    Sequential selection without replacement: each position divides by the mass still available,
    which is the full normaliser minus the mass of everything already placed. Computed in log
    space because the product of three small probabilities underflows a float32 cheerfully and a
    float64 eventually.
    """
    lg = np.asarray(lg, dtype=np.float64)
    slate = np.asarray(slate, dtype=np.int64)
    if lg.ndim != 2 or slate.ndim != 2:
        raise ValueError("logits and slate must both be two dimensional")
    if lg.shape[0] != slate.shape[0]:
        raise ValueError(f"logits has {lg.shape[0]} rows but slate has {slate.shape[0]}")
    shift = lg.max(axis=1, keepdims=True)
    weights = np.exp(lg - shift)
    remaining = weights.sum(axis=1)
    total = np.zeros(lg.shape[0], dtype=np.float64)
    rows = np.arange(lg.shape[0])
    for position in range(slate.shape[1]):
        chosen = slate[:, position]
        picked = weights[rows, chosen]
        total += np.log(picked) - np.log(remaining)
        remaining = remaining - picked
    return total


def sample_slates(lg: np.ndarray, n_positions: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one ordered slate per row by the Gumbel top-k trick.

    Adding independent Gumbel noise to logits and taking the top k in order is exactly a
    Plackett-Luce draw, and unlike sequential categorical sampling it vectorises over every
    session in the log at once.
    """
    lg = np.asarray(lg, dtype=np.float64)
    gumbel = rng.gumbel(size=lg.shape)
    perturbed = lg + gumbel
    order = np.argsort(-perturbed, axis=1, kind="stable")
    return order[:, :n_positions].astype(np.int64)


@lru_cache(maxsize=8)
def _position_selectors(n_candidates: int, n_positions: int) -> tuple[np.ndarray, ...]:
    """One indicator matrix per position, shape (n_permutations, n_candidates).

    Collapsing the slate distribution onto positions is a sum over permutations grouped by which
    item each one placed where, and that grouping is fixed by the permutation index. Expressing
    it as a matrix turns the collapse into a dense product instead of a scatter add over tens of
    millions of entries, which on a fifty thousand session log was the single slowest operation
    in the pipeline before it was measured.
    """
    perms = permutation_index(n_candidates, n_positions)
    selectors = []
    for position in range(n_positions):
        indicator = np.zeros((perms.shape[0], n_candidates), dtype=np.float64)
        indicator[np.arange(perms.shape[0]), perms[:, position]] = 1.0
        selectors.append(indicator)
    return tuple(selectors)


def position_marginals(lg: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """Probability that each candidate lands in each position, summed over every ordered slate.

    Returns shape (n_rows, n_positions, n_candidates). This is the exact slate distribution
    collapsed to what the reward actually depends on, which is which item sits where.
    """
    perms = np.asarray(perms, dtype=np.int64)
    n_candidates = int(np.asarray(lg).shape[1])
    probabilities = _permutation_probabilities_t(lg, perms)
    selectors = _position_selectors(n_candidates, perms.shape[1])
    stacked = [probabilities.T @ selector for selector in selectors]
    return np.stack(stacked, axis=1)


def permutation_probabilities(lg: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """Probability of every ordered slate, shape (n_rows, n_permutations).

    The same sequential normaliser as the propensity of a single slate, applied to all of them at
    once. The work happens in a (n_permutations, n_rows) layout and is transposed back at the end:
    gathering 720 whole rows out of a (n_candidates, n_rows) array is a sequence of contiguous
    copies, while gathering 720 columns out of the natural layout is strided, and on a fifty
    thousand session log the difference measured four to one.
    """
    return _permutation_probabilities_t(lg, perms).T.copy()


def _permutation_probabilities_t(lg: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """Permutation probabilities in the transposed (n_permutations, n_rows) working layout."""
    lg = np.asarray(lg, dtype=np.float64)
    perms = np.asarray(perms, dtype=np.int64)
    if lg.ndim != 2:
        raise ValueError("logits must be two dimensional")
    if perms.shape[1] > lg.shape[1]:
        raise ValueError("the permutation index names more positions than there are candidates")
    weights = np.ascontiguousarray(np.exp(lg - lg.max(axis=1, keepdims=True)).T)
    remaining = weights.sum(axis=0)
    probabilities = np.ones((perms.shape[0], lg.shape[0]), dtype=np.float64)
    for position in range(perms.shape[1]):
        picked = weights[perms[:, position]]
        probabilities *= picked / remaining
        remaining = remaining - picked
    return probabilities


def expected_slate_value(
    lg: np.ndarray,
    per_item_value: np.ndarray,
    examination: np.ndarray,
    perms: np.ndarray,
    chunk_rows: int = 8192,
) -> np.ndarray:
    """Exact expected reward per row under the policy described by these logits.

    ``per_item_value[r, i]`` is the expected reward contributed by item ``i`` if a user examines
    it in session ``r``, and ``examination[k]`` is the probability that position ``k`` is examined.
    The expectation runs over the full slate distribution with no sampling, which is what makes
    this function usable as a referee rather than as another estimator.

    Chunked over rows because the intermediate is 720 by rows and a fifty thousand row log would
    otherwise allocate several hundred megabytes at once.
    """
    lg = np.asarray(lg, dtype=np.float64)
    per_item_value = np.asarray(per_item_value, dtype=np.float64)
    examination = np.asarray(examination, dtype=np.float64)
    if lg.shape != per_item_value.shape:
        raise ValueError(
            f"logits shape {lg.shape} does not match per item value shape {per_item_value.shape}"
        )
    if examination.shape[0] != perms.shape[1]:
        raise ValueError("examination length must match the number of slate positions")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be at least 1")

    out = np.empty(lg.shape[0], dtype=np.float64)
    for start in range(0, lg.shape[0], chunk_rows):
        stop = min(start + chunk_rows, lg.shape[0])
        probabilities = _permutation_probabilities_t(lg[start:stop], perms)
        values = np.ascontiguousarray(per_item_value[start:stop].T)
        payoff = np.zeros_like(probabilities)
        for position in range(perms.shape[1]):
            payoff += examination[position] * values[perms[:, position]]
        out[start:stop] = (probabilities * payoff).sum(axis=0)
    return out


def top1_choice(scores: np.ndarray) -> np.ndarray:
    """The item a deterministic argmax ranker would put in the first slot.

    Ties break to the lowest index, deterministically, because a tie broken by hash order makes
    every number downstream depend on the Python process.
    """
    return np.asarray(scores, dtype=np.float64).argmax(axis=1).astype(np.int64)
