"""What every candidate policy has to provide, and nothing more.

A policy is a function from a session's candidate set to scores. Everything else, the
standardisation, the temperature, the Plackett-Luce distribution, happens once here, so that
comparing two policies is a comparison of rankings rather than an accidental comparison of score
scales. A gradient boosted model and a dot product do not produce numbers on the same scale, and
without this the temperature would silently mean something different for each of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from replay_gap.ranking import logits as pl_logits
from replay_gap.ranking import top1_choice


class Policy(ABC):
    """A named scoring function over candidate sets, plus the distribution it induces."""

    name: str
    temperature: float

    @abstractmethod
    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Score every candidate in every session, shape (n_sessions, n_candidates)."""

    def logits(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Standardised scores at this policy's temperature."""
        return pl_logits(self.raw_scores(context_ids, candidates), self.temperature)

    def top1(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """The slot a deterministic argmax version of this policy would put first.

        Naive replay needs this and only this, which is itself part of the story: the estimator
        that is most often used throws away everything the policy says about positions two and
        three.
        """
        return top1_choice(self.raw_scores(context_ids, candidates))


class ScorePolicy(Policy):
    """A policy built from a callable, used where defining a class would be ceremony."""

    def __init__(self, name: str, scorer, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.name = name
        self.temperature = float(temperature)
        self._scorer = scorer

    def raw_scores(self, context_ids: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return np.asarray(self._scorer(context_ids, candidates), dtype=np.float64)
