"""The latent world and the user who reacts to a slate.

Everything the estimators are not allowed to see lives here: true relevance per context and item,
and the position examination curve that turns a slate into clicks. The split matters because the
credibility of every number in the repository rests on the estimators having access to exactly
what a production log would carry and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so a dot product lands in [-1, 1] and the sigmoid behaves."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic, written out because the obvious form overflows on -800."""
    x = np.asarray(x, dtype=np.float64)
    positive = x >= 0
    out = np.empty_like(x)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


@dataclass(frozen=True)
class World:
    """Items, users, and the true relevance that links them.

    ``item_quality`` is a per item term independent of the user. It exists so that a popularity
    baseline is a sensible policy rather than noise, which in turn makes the comparison between
    estimators resemble the comparison a real team runs.
    """

    item_features: np.ndarray
    item_quality: np.ndarray
    context_features: np.ndarray
    context_segment: np.ndarray
    examination: np.ndarray
    relevance_scale: float

    @property
    def n_items(self) -> int:
        return int(self.item_features.shape[0])

    @property
    def n_contexts(self) -> int:
        return int(self.context_features.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.item_features.shape[1])

    @property
    def n_positions(self) -> int:
        return int(self.examination.shape[0])

    @property
    def n_segments(self) -> int:
        return int(self.context_segment.max()) + 1

    def relevance_logit(self, context_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Log odds of a click given that the user examined this item, shape (n_rows, n_items_in).

        ``context_ids`` is one dimensional with one entry per row; ``item_ids`` is two dimensional
        and names the items in that row. Written this way because every caller holds a candidate
        set per session, and spelling the broadcast out here keeps it out of five call sites.
        """
        context_ids = np.asarray(context_ids, dtype=np.int64)
        item_ids = np.asarray(item_ids, dtype=np.int64)
        if context_ids.ndim != 1:
            raise ValueError("context_ids must be one dimensional")
        if item_ids.ndim != 2:
            raise ValueError("item_ids must be two dimensional, one row per context")
        if item_ids.shape[0] != context_ids.shape[0]:
            raise ValueError(
                f"item_ids has {item_ids.shape[0]} rows but context_ids has {context_ids.shape[0]}"
            )
        user = self.context_features[context_ids][:, None, :]
        item = self.item_features[item_ids]
        affinity = (user * item).sum(axis=-1)
        return self.relevance_scale * affinity + self.item_quality[item_ids]

    def relevance(self, context_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Probability that an examined item is clicked. This is the quantity nothing else sees."""
        return sigmoid(self.relevance_logit(context_ids, item_ids))

    def expected_click_value(self, context_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Alias that names what the number is used for: the payoff of showing an item."""
        return self.relevance(context_ids, item_ids)


def build_world(
    n_items: int,
    n_item_features: int,
    n_segments: int,
    n_contexts: int,
    relevance_scale: float,
    quality_spread: float,
    examination: np.ndarray,
    rng: np.random.Generator,
) -> World:
    """Draw a world once, from one generator, so the whole repository shares one ground truth.

    Users sit in segments around a centroid rather than being independent draws, because a
    structureless world is one no ranker can learn, and a log produced by a ranker that learned
    nothing is not the log this repository is about.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1")
    item_features = unit_rows(rng.normal(size=(n_items, n_item_features)))
    item_quality = rng.normal(scale=quality_spread, size=n_items)

    centroids = unit_rows(rng.normal(size=(n_segments, n_item_features)))
    segment = rng.integers(0, n_segments, size=n_contexts)
    jitter = rng.normal(scale=0.45, size=(n_contexts, n_item_features))
    context_features = unit_rows(centroids[segment] + jitter)

    return World(
        item_features=item_features,
        item_quality=item_quality,
        context_features=context_features,
        context_segment=segment.astype(np.int64),
        examination=np.asarray(examination, dtype=np.float64),
        relevance_scale=float(relevance_scale),
    )


def draw_clicks(
    relevance: np.ndarray, examination: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Clicks on a slate: the user examines position k with probability examination[k], and clicks
    an examined item with its relevance.

    The two Bernoullis collapse into one with probability ``examination[k] * relevance``, which is
    both faster and, more usefully, makes it obvious in the code that position bias and relevance
    are not separately identifiable from a click alone.
    """
    relevance = np.asarray(relevance, dtype=np.float64)
    examination = np.asarray(examination, dtype=np.float64)
    if relevance.ndim != 2:
        raise ValueError("relevance must be two dimensional, one row per session")
    if relevance.shape[1] != examination.shape[0]:
        raise ValueError(
            f"relevance has {relevance.shape[1]} columns but examination has "
            f"{examination.shape[0]} positions"
        )
    probability = relevance * examination[None, :]
    return (rng.random(size=probability.shape) < probability).astype(np.int8)
