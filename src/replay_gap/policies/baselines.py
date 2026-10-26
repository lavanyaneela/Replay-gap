"""The baseline policies, two of which exist specifically to break naive replay.

``incumbent`` is the deployed ranker re-evaluated as though it were a candidate. Its true lift
over itself is exactly zero, which makes it the cleanest possible probe: any estimator that
reports a non-zero lift for it is reporting its own bias, with no model quality mixed in.

``contrarian_lift`` is genuinely better than the incumbent and agrees with it on the first slot
in under one session in ten. The brief for this repository predicted that naive replay would
under-rate it. The measurement disagreed, and E1 publishes what actually happens.
"""

from __future__ import annotations

import numpy as np

from replay_gap.policies.base import Policy
from replay_gap.simulator.logging_policy import LoggingPolicy
from replay_gap.simulator.user_model import World


def _standardise_scores(scores: np.ndarray) -> np.ndarray:
    """Centre and scale over the whole score matrix, not per row.

    Per row standardisation happens later, inside the Plackett-Luce layer. Here the goal is only
    to put the three score sources a blend mixes onto one scale, so a blend weight means a share
    of influence rather than a share of whatever units that source happened to use.
    """
    scores = np.asarray(scores, dtype=np.float64)
    spread = scores.std()
    if spread < 1e-12:
        return np.zeros_like(scores)
    return (scores - scores.mean()) / spread


class OraclePolicy(Policy):
    """Ranks by true relevance. The ceiling, and not a policy anybody can build."""

    def __init__(self, world: World, temperature: float = 1.0) -> None:
        self.name = "oracle"
        self.temperature = float(temperature)
        self._world = world

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return self._world.relevance_logit(context_ids, candidates)


class IncumbentPolicy(Policy):
    """The deployed ranker, re-evaluated as a candidate. True lift over itself is zero."""

    def __init__(
        self, world: World, logging_policy: LoggingPolicy, temperature: float = 1.0
    ) -> None:
        self.name = "incumbent"
        self.temperature = float(temperature)
        self._world = world
        self._policy = logging_policy

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return self._policy.scores(self._world, context_ids, candidates)


class UniformPolicy(Policy):
    """Scores every candidate equally, so the induced distribution is uniform over slates."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.name = "uniform"
        self.temperature = float(temperature)

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:  # noqa: ARG002
        # Context is ignored on purpose: a policy that cannot tell users apart is the floor the
        # other baselines are measured against.
        return np.zeros(np.asarray(candidates).shape, dtype=np.float64)


class PopularityPolicy(Policy):
    """Ranks by clicks observed in the log. A real baseline a real team would ship.

    It is also the one baseline whose scores come from the log rather than from the world, which
    means it inherits the logging policy's exposure bias: an item the deployed ranker never shows
    cannot accumulate clicks and is therefore never popular.
    """

    def __init__(self, click_counts: np.ndarray, temperature: float = 1.0) -> None:
        self.name = "popularity"
        self.temperature = float(temperature)
        self._scores = np.log1p(np.asarray(click_counts, dtype=np.float64))

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:  # noqa: ARG002
        # Context is ignored because popularity is a global ordering. That is the baseline's
        # defining weakness and the reason it is worth including.
        return self._scores[np.asarray(candidates, dtype=np.int64)]


class BlendPolicy(Policy):
    """A convex blend of three score sources: true relevance, the incumbent, and fixed noise.

    This is the family E1 samples from. The three sources are the only things a candidate ranker
    can really be made of in this world (it knows something true, it inherits something from the
    model that produced its training data, and it is wrong in some arbitrary direction), and
    drawing the weights uniformly over the simplex is the least rigged way to sample it.

    The noise component is keyed by a seed and generated from the candidate ids and context ids,
    so the policy is a fixed function rather than something that re-randomises each time it is
    asked, which would make it not a policy at all.
    """

    def __init__(
        self,
        world: World,
        logging_policy: LoggingPolicy,
        truth_weight: float,
        incumbent_weight: float,
        noise_weight: float,
        noise_seed: int,
        temperature: float = 1.0,
        name: str = "blend",
    ) -> None:
        if min(truth_weight, incumbent_weight, noise_weight) < 0:
            raise ValueError("blend weights must not be negative")
        if truth_weight + incumbent_weight + noise_weight <= 0:
            raise ValueError("at least one blend weight must be positive")
        self.name = name
        self.temperature = float(temperature)
        self.truth_weight = float(truth_weight)
        self.incumbent_weight = float(incumbent_weight)
        self.noise_weight = float(noise_weight)
        self.noise_seed = int(noise_seed)
        self._world = world
        self._policy = logging_policy
        self._noise = _item_noise(world.n_items, world.n_segments, noise_seed)

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        context_ids = np.asarray(context_ids, dtype=np.int64)
        candidates = np.asarray(candidates, dtype=np.int64)
        total = np.zeros(candidates.shape, dtype=np.float64)
        if self.truth_weight:
            total += self.truth_weight * _standardise_scores(
                self._world.relevance_logit(context_ids, candidates)
            )
        if self.incumbent_weight:
            total += self.incumbent_weight * _standardise_scores(
                self._policy.scores(self._world, context_ids, candidates)
            )
        if self.noise_weight:
            segment = self._world.context_segment[context_ids]
            total += self.noise_weight * self._noise[segment[:, None], candidates]
        return total


def _item_noise(n_items: int, n_segments: int, seed: int) -> np.ndarray:
    """A fixed per segment, per item score offset. Deterministic given the seed."""
    return np.random.default_rng(seed).normal(size=(n_segments, n_items))


def click_counts_from_log(n_items: int, slate_items: np.ndarray, clicks: np.ndarray) -> np.ndarray:
    """Clicks per item as the log records them, which is all a popularity baseline can see."""
    counts = np.zeros(n_items, dtype=np.float64)
    np.add.at(counts, slate_items.reshape(-1), clicks.reshape(-1).astype(np.float64))
    return counts


class ContrarianPolicy(Policy):
    """Genuinely better than the incumbent, and almost never agrees with it on the first slot.

    Built by leaning on true relevance and subtracting the incumbent's opinion, which is a
    caricature of a model trained to correct a known bias in production. It exists so that
    "disagrees a lot" and "is worse" are separated in the experiment, because in a log produced
    by the incumbent those two are exactly what naive replay confuses.
    """

    def __init__(
        self,
        world: World,
        logging_policy: LoggingPolicy,
        temperature: float = 1.0,
        name: str = "contrarian_lift",
    ) -> None:
        self.name = name
        self.temperature = float(temperature)
        self._world = world
        self._policy = logging_policy

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        truth = _standardise_scores(self._world.relevance_logit(context_ids, candidates))
        incumbent = _standardise_scores(self._policy.scores(self._world, context_ids, candidates))
        return 1.6 * truth - 0.9 * incumbent


def build_baseline_policies(
    world: World,
    logging_policy: LoggingPolicy,
    click_counts: np.ndarray,
    temperature: float = 1.0,
    sharp_temperature: float = 0.25,
) -> dict[str, Policy]:
    """The fixed baseline set, in the order the report prints them.

    ``contrarian_sharp`` is the same ranker as ``contrarian_lift``, served at a lower
    temperature. Nothing about the model changed; only how greedily it is served. It is in the
    set because serving greedily is what a team actually does, and because it is the policy that
    makes this log unusable: its importance weights reach four figures and the effective sample
    size falls below the floor, so the tool refuses rather than estimating.
    """
    return {
        "incumbent": IncumbentPolicy(world, logging_policy, temperature),
        "incumbent_clone": BlendPolicy(
            world,
            logging_policy,
            truth_weight=0.0,
            incumbent_weight=1.0,
            noise_weight=0.10,
            noise_seed=4101,
            temperature=temperature,
            name="incumbent_clone",
        ),
        "popularity": PopularityPolicy(click_counts, temperature),
        "uniform": UniformPolicy(temperature),
        "contrarian_lift": ContrarianPolicy(world, logging_policy, temperature),
        "contrarian_sharp": ContrarianPolicy(
            world, logging_policy, sharp_temperature, name="contrarian_sharp"
        ),
        "oracle": OraclePolicy(world, temperature),
    }
