"""Self-normalised inverse propensity scoring, the default estimator in this repository.

Dividing by the sum of the weights instead of the sample size trades a small bias for a large
variance reduction, and it has a property that matters more than either: the estimate is bounded
by the range of the observed rewards. Plain IPS is not, and a value estimate above the maximum
achievable reward is a number no reviewer will read past.
"""

from __future__ import annotations

import numpy as np


def snips_value(reward: np.ndarray, weight: np.ndarray, index: np.ndarray | None = None) -> float:
    """The self-normalised estimate. Returns NaN when the weights sum to zero."""
    if index is not None:
        reward = reward[index]
        weight = weight[index]
    total = float(np.sum(weight))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(weight * reward) / total)
