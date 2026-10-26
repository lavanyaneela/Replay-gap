"""Naive replay, written out in full because it is the thing this repository is arguing with.

The estimator keeps the sessions where the logging policy happened to put the candidate policy's
own first choice in the first slot, and averages the reward over those sessions. It is the most
widely used offline check there is, it needs no propensities, and it is wrong in a specific and
predictable way: the sessions it keeps are selected on agreement with the logging policy, and
agreement with the logging policy is correlated with reward.

A second variant scores only the click in the first slot rather than the whole session. It is
here because the obvious objection to the main result is that it depends on how reward was
defined, and E3 answers that objection with a measurement instead of an argument.
"""

from __future__ import annotations

import numpy as np

from replay_gap.evaluation import PolicyTrace
from replay_gap.simulator.generate import InteractionLog


def naive_match_mask(log: InteractionLog, trace: PolicyTrace) -> np.ndarray:
    """Sessions where the served first slot is the one this policy would have chosen."""
    return trace.top1 == log.slate[:, 0]


def naive_match_rate(log: InteractionLog, trace: PolicyTrace) -> float:
    """Share of the log a naive replay of this policy actually uses."""
    return float(naive_match_mask(log, trace).mean())


def naive_replay_value(
    reward: np.ndarray, matched: np.ndarray, index: np.ndarray | None = None
) -> float:
    """Mean reward over matched sessions. NaN when nothing matched."""
    if index is not None:
        reward = reward[index]
        matched = matched[index]
    kept = reward[matched]
    if kept.shape[0] == 0:
        return float("nan")
    return float(kept.mean())
