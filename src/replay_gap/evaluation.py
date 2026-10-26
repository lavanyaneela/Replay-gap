"""Everything a policy has to be reduced to before any estimator may look at it.

A trace is computed once per policy per log. The reason is not only speed, although scoring half
a million candidate rows through a gradient boosted model is not free. It is that an estimator
which holds a policy object can ask it anything, including, by accident, for the true relevance
that the policy was built from. A trace carries exactly three things: the slate distribution, the
probability it assigns to the slate that was actually served, and the first slot a deterministic
version of it would pick. Nothing else about the policy is reachable from the estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from replay_gap.policies.base import Policy
from replay_gap.ranking import permutation_index, position_marginals, slate_log_propensity
from replay_gap.simulator.generate import InteractionLog


@dataclass(frozen=True)
class PolicyTrace:
    """A candidate policy, flattened to what the estimators are allowed to see."""

    name: str
    logits: np.ndarray
    log_propensity: np.ndarray
    top1: np.ndarray
    marginals: np.ndarray

    @property
    def n_sessions(self) -> int:
        return int(self.logits.shape[0])


def trace_policy(
    policy: Policy, log: InteractionLog, perms: np.ndarray | None = None
) -> PolicyTrace:
    """Score a policy against a log once and keep only what an estimator needs.

    The position marginals are computed here rather than by each consumer. Two things need them,
    the overlap diagnostic and the outcome model term, and collapsing the slate distribution
    twice per policy was measurably the second most expensive thing the pipeline did.
    """
    if perms is None:
        perms = permutation_index(log.n_candidates, log.n_positions)
    lg = policy.logits(log.context_id, log.candidates)
    if lg.shape != (log.n_sessions, log.n_candidates):
        raise ValueError(
            f"policy {policy.name} produced logits of shape {lg.shape}, expected "
            f"{(log.n_sessions, log.n_candidates)}"
        )
    return PolicyTrace(
        name=policy.name,
        logits=lg,
        log_propensity=slate_log_propensity(lg, log.slate),
        top1=policy.top1(log.context_id, log.candidates),
        marginals=position_marginals(lg, perms),
    )
