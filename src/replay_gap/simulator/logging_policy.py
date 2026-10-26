"""The ranker that is already deployed, and therefore the reason the log is confounded.

Its view of the world is a perturbed copy of the truth: the same item embeddings with noise added
and the same quality term with noise added. That shape matters. A logging policy that ranked at
random would produce a log every estimator handles correctly, and a logging policy that ranked
perfectly would leave nothing to improve. A policy that is roughly right is the one that makes
replay look convincing and be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replay_gap.ranking import logits as pl_logits
from replay_gap.ranking import sample_slates, slate_log_propensity
from replay_gap.simulator.user_model import World, unit_rows


@dataclass(frozen=True)
class LoggingPolicy:
    """A Plackett-Luce policy over a perturbed view of true relevance.

    Because the policy is explicitly Plackett-Luce, the propensity of a slate it produced is a
    closed form rather than an estimate. ADR-005 is about what changes when it is not.
    """

    item_features: np.ndarray
    item_quality: np.ndarray
    relevance_scale: float
    temperature: float
    name: str = "logging_policy"

    def scores(self, world: World, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Score every candidate in every session, shape (n_sessions, n_candidates)."""
        context_ids = np.asarray(context_ids, dtype=np.int64)
        candidates = np.asarray(candidates, dtype=np.int64)
        user = world.context_features[context_ids][:, None, :]
        item = self.item_features[candidates]
        affinity = (user * item).sum(axis=-1)
        return self.relevance_scale * affinity + self.item_quality[candidates]

    def logits(self, world: World, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Standardised scores at this policy's temperature."""
        return pl_logits(self.scores(world, context_ids, candidates), self.temperature)

    def serve(
        self,
        world: World,
        context_ids: np.ndarray,
        candidates: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a slate per session and return it with its exact log propensity.

        The slate is returned as indices into the session's candidate set rather than as item
        ids, because every estimator downstream works in candidate space and converting once here
        removes a class of off by one that is invisible until a number comes out wrong.
        """
        lg = self.logits(world, context_ids, candidates)
        slate = sample_slates(lg, world.n_positions, rng)
        log_propensity = slate_log_propensity(lg, slate)
        return slate, log_propensity


def build_logging_policy(
    world: World,
    item_embedding_noise: float,
    quality_noise: float,
    temperature: float,
    rng: np.random.Generator,
) -> LoggingPolicy:
    """Perturb the world's own parameters to produce the deployed ranker's view of it."""
    if item_embedding_noise < 0 or quality_noise < 0:
        raise ValueError("noise scales must not be negative")
    perturbed_features = unit_rows(
        world.item_features + rng.normal(scale=item_embedding_noise, size=world.item_features.shape)
    )
    perturbed_quality = world.item_quality + rng.normal(
        scale=quality_noise, size=world.item_quality.shape
    )
    return LoggingPolicy(
        item_features=perturbed_features,
        item_quality=perturbed_quality,
        relevance_scale=world.relevance_scale,
        temperature=float(temperature),
    )
