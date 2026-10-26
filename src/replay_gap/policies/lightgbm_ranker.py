"""A LambdaMART ranker trained on the logged data, which is the policy a team would actually ship.

Two details here are deliberate and are part of what the repository measures rather than
incidental choices.

First, every candidate the logging policy did not show is treated as a negative. That is what
most teams do, and it bakes the logging policy's exposure decisions into the labels: an item
never shown cannot be clicked, so the ranker learns that it is bad. The confounding this
repository is about therefore reaches the candidate model too, not only the evaluation of it.

Second, training is pinned to one thread with LightGBM's deterministic mode. A ranker whose
scores depend on thread count would make every published number depend on the machine, and the
receipts mechanism would be measuring the runner.
"""

from __future__ import annotations

import numpy as np

from replay_gap.config import GBDTConfig
from replay_gap.policies.base import Policy
from replay_gap.simulator.generate import InteractionLog
from replay_gap.simulator.user_model import World


def build_features(world: World, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Context features, item features, and their elementwise product, per candidate.

    The product term is there because true relevance is a dot product and an axis aligned tree
    cannot represent one from the two factors alone. Leaving it out would produce a ranker that
    is bad for a reason the repository is not about.
    """
    context_ids = np.asarray(context_ids, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    n_sessions, n_candidates = candidates.shape
    user = world.context_features[context_ids][:, None, :]
    user = np.broadcast_to(user, (n_sessions, n_candidates, world.n_features))
    item = world.item_features[candidates]
    product = user * item
    stacked = np.concatenate([user, item, product], axis=2)
    return stacked.reshape(n_sessions * n_candidates, 3 * world.n_features)


def build_training_labels(log: InteractionLog) -> np.ndarray:
    """One label per candidate: clicked if shown and clicked, zero otherwise.

    The zero for an un-impressed candidate is a lie the data cannot correct, and it is the same
    lie every production ranker trained on implicit feedback is told.
    """
    labels = np.zeros((log.n_sessions, log.n_candidates), dtype=np.int32)
    rows = np.arange(log.n_sessions)[:, None]
    np.add.at(labels, (rows, log.slate), log.clicks.astype(np.int32))
    return np.minimum(labels, 1).reshape(-1)


def train_lightgbm_ranker(world: World, log: InteractionLog, cfg: GBDTConfig):
    """Fit LambdaMART on the log. Returns the fitted booster wrapper."""
    import lightgbm as lgb

    features = build_features(world, log.context_id, log.candidates)
    labels = build_training_labels(log)
    groups = np.full(log.n_sessions, log.n_candidates, dtype=np.int32)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        random_state=cfg.seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(features, labels, group=groups)
    return model


class LightGBMRankerPolicy(Policy):
    """Wraps a fitted LambdaMART model so it can be scored like any other policy."""

    def __init__(self, world: World, model, temperature: float = 1.0) -> None:
        self.name = "lightgbm_ranker"
        self.temperature = float(temperature)
        self._world = world
        self._model = model

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int64)
        features = build_features(self._world, context_ids, candidates)
        scores = self._model.predict(features, num_threads=1)
        return np.asarray(scores, dtype=np.float64).reshape(candidates.shape)
