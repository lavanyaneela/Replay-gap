"""Effective sample size: what a logged session is worth once it has been reweighted.

Kish's formula, the ratio of the squared sum of weights to the sum of squared weights. It is the
number this repository leads with, because it converts an abstract warning about variance into
the sentence a product manager understands: this fifty thousand session log is worth some much
smaller number of sessions for this particular question.
"""

from __future__ import annotations

import numpy as np


def effective_sample_size(weight: np.ndarray) -> float:
    """Kish effective sample size. Zero when every weight is zero."""
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 1:
        raise ValueError("weights must be one dimensional")
    if weight.size == 0:
        return 0.0
    if np.any(weight < 0):
        raise ValueError("importance weights cannot be negative")
    squared_sum = float(np.sum(weight)) ** 2
    sum_squared = float(np.sum(weight**2))
    if sum_squared <= 0.0:
        return 0.0
    return squared_sum / sum_squared


def ess_fraction(weight: np.ndarray) -> float:
    """Effective sample size as a share of the sessions available."""
    weight = np.asarray(weight, dtype=np.float64)
    if weight.size == 0:
        return 0.0
    return effective_sample_size(weight) / weight.size
