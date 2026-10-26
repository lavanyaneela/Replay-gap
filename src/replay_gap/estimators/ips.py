"""Inverse propensity scoring, and the clipping decision that ADR-004 is about.

The weight is the ratio of the candidate policy's probability of the served slate to the logging
policy's. It is computed as a difference of log probabilities because the two numbers are each
around a thousandth and their ratio is the quantity of interest: subtracting logs keeps four more
significant figures than dividing floats does.
"""

from __future__ import annotations

import numpy as np

from replay_gap.evaluation import PolicyTrace
from replay_gap.simulator.generate import InteractionLog


def importance_weights(
    log: InteractionLog, trace: PolicyTrace, clip: float | None = None
) -> np.ndarray:
    """Per session importance weights, optionally capped.

    Clipping caps variance and introduces bias, and the bias runs in the direction that flatters
    a candidate policy, because the sessions whose weights are cut are exactly the ones where the
    candidate most disagrees with the log. The cap is a configuration value, never a constant in
    the code, and E3 prices it.
    """
    weights = np.exp(trace.log_propensity - log.log_propensity)
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive")
        weights = np.minimum(weights, clip)
    return weights


def ips_value(reward: np.ndarray, weight: np.ndarray, index: np.ndarray | None = None) -> float:
    """Plain IPS: unbiased when the weights are exact, and unusable when overlap is poor."""
    if index is not None:
        reward = reward[index]
        weight = weight[index]
    if reward.shape[0] == 0:
        return float("nan")
    return float(np.mean(weight * reward))
