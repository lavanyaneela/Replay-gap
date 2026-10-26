"""The referee: the exact online value of a policy, with no sampling error at all.

Every off-policy estimator in this repository is trying to estimate the expected reward a policy
would earn over the context and candidate set distribution the log contains. That quantity is
computed here in closed form: the slate distribution is Plackett-Luce and enumerable, the click
model is Bernoulli and its expectation is its probability, and the context distribution is the
empirical one the log already carries. So the referee's answer is not an estimate of the
estimand, it is the estimand.

This is the single property that lets the repository say "naive replay got the sign wrong"
instead of "naive replay is theoretically biased", and it is why ADR-001 rejects every public
learning to rank dataset.
"""

from __future__ import annotations

import numpy as np

from replay_gap.ranking import expected_slate_value, permutation_index
from replay_gap.simulator.generate import InteractionLog
from replay_gap.simulator.user_model import World


def per_session_true_value(
    world: World, log: InteractionLog, policy_logits: np.ndarray, perms: np.ndarray | None = None
) -> np.ndarray:
    """Exact expected reward for each logged session under the given policy logits."""
    if policy_logits.shape != (log.n_sessions, log.n_candidates):
        raise ValueError(
            f"policy logits have shape {policy_logits.shape}, expected "
            f"{(log.n_sessions, log.n_candidates)}"
        )
    if perms is None:
        perms = permutation_index(log.n_candidates, log.n_positions)
    relevance = world.relevance(log.context_id, log.candidates)
    return expected_slate_value(policy_logits, relevance, world.examination, perms)


def true_value(
    world: World, log: InteractionLog, policy_logits: np.ndarray, perms: np.ndarray | None = None
) -> float:
    """The policy's exact value, averaged over the sessions the log actually contains."""
    return float(per_session_true_value(world, log, policy_logits, perms).mean())


def true_value_lift(candidate_value: float, baseline_value: float) -> float:
    """Relative lift of a candidate over a baseline, as a fraction.

    Expressed relative rather than absolute because that is the unit a team argues about, and
    because an absolute click delta on a simulated world means nothing to a reader.
    """
    if baseline_value <= 0:
        raise ValueError("baseline value must be positive to express a relative lift")
    return candidate_value / baseline_value - 1.0
