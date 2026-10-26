"""The per session quantities every estimator is a function of.

Computing these once and then expressing each estimator as a function of a session index array
is what lets the bootstrap resample sessions rather than resample an estimator's output. It also
makes the four estimators directly comparable, because they are then provably reading the same
rewards and the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replay_gap.estimators.bootstrap import RatioEstimator
from replay_gap.estimators.doubly_robust import reward_model_terms
from replay_gap.estimators.ips import importance_weights
from replay_gap.estimators.naive import naive_match_mask
from replay_gap.evaluation import PolicyTrace
from replay_gap.simulator.generate import InteractionLog


@dataclass(frozen=True)
class EstimatorInputs:
    """Everything an estimator reads, per session, for one candidate policy."""

    policy: str
    reward: np.ndarray
    position0_click: np.ndarray
    raw_weight: np.ndarray
    weight: np.ndarray
    matched: np.ndarray
    direct: np.ndarray | None = None
    logged_prediction: np.ndarray | None = None

    @property
    def n_sessions(self) -> int:
        return int(self.reward.shape[0])

    @property
    def has_reward_model(self) -> bool:
        return self.direct is not None and self.logged_prediction is not None

    def ratio_estimators(self) -> tuple[dict[str, RatioEstimator], dict[str, RatioEstimator]]:
        """Each estimator and the baseline it has to be divided by, as session level sums.

        The baselines are not all the same, and that is the point of returning them together.
        Naive replay on session reward is compared against the log's mean session reward; naive
        replay scored on the first slot alone is compared against the log's first slot click
        rate. Using one baseline for both produces a lift of minus thirty eight percent for a
        policy that is the incumbent, which looks like a finding and is a units error.
        """
        ones = np.ones_like(self.reward)
        estimators = {
            "naive": RatioEstimator(
                "naive", self.reward * self.matched, self.matched.astype(np.float64)
            ),
            "naive_first_slot": RatioEstimator(
                "naive_first_slot",
                self.position0_click * self.matched,
                self.matched.astype(np.float64),
            ),
            "ips": RatioEstimator("ips", self.raw_weight * self.reward, ones),
            "snips": RatioEstimator("snips", self.weight * self.reward, self.weight),
        }
        baselines = {
            "naive": RatioEstimator("logged_reward", self.reward, ones),
            "naive_first_slot": RatioEstimator("logged_first_slot", self.position0_click, ones),
            "ips": RatioEstimator("logged_reward", self.reward, ones),
            "snips": RatioEstimator("logged_reward", self.reward, ones),
        }
        if self.has_reward_model:
            corrected = self.direct + self.weight * (self.reward - self.logged_prediction)
            estimators["dr"] = RatioEstimator("dr", corrected, ones)
            baselines["dr"] = RatioEstimator("logged_reward", self.reward, ones)
        return estimators, baselines

    @property
    def clipped_share(self) -> float:
        """Share of sessions whose weight the cap actually changed.

        Published next to any clipped estimate, because a cap that binds on one session in a
        thousand is a footnote and a cap that binds on one in five is the estimator.
        """
        return float(np.mean(self.raw_weight > self.weight + 1e-12))


def build_estimator_inputs(
    log: InteractionLog,
    trace: PolicyTrace,
    clip: float | None,
    candidate_rewards: np.ndarray | None = None,
) -> EstimatorInputs:
    """Reduce a log and a policy trace to the arrays the estimators consume."""
    raw = importance_weights(log, trace, clip=None)
    clipped = importance_weights(log, trace, clip=clip)
    direct = None
    logged_prediction = None
    if candidate_rewards is not None:
        direct, logged_prediction = reward_model_terms(log, trace, candidate_rewards)
    return EstimatorInputs(
        policy=trace.name,
        reward=log.reward.astype(np.float64),
        position0_click=log.clicks[:, 0].astype(np.float64),
        raw_weight=raw,
        weight=clipped,
        matched=naive_match_mask(log, trace),
        direct=direct,
        logged_prediction=logged_prediction,
    )
