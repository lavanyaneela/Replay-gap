"""Overlap: whether the log contains the behaviour the candidate policy wants to try.

Four readings, each answering a different objection.

``ess_fraction`` says how much of the log survives reweighting. ``max_weight_share`` says whether
one session is carrying the estimate, which is the failure mode an effective sample size in the
thousands can still hide. ``top1_agreement`` says how much of the log naive replay would keep,
and is the clearest single predictor of how badly naive replay will be wrong. ``unseen_exposure``
says how much of the candidate policy's first slot mass sits on items the log never showed in the
first slot at all, which is the part of the question no amount of reweighting can answer because
the data simply is not there.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from replay_gap.diagnostics.ess import effective_sample_size, ess_fraction
from replay_gap.evaluation import PolicyTrace
from replay_gap.simulator.generate import InteractionLog


@dataclass(frozen=True)
class OverlapReport:
    """The four overlap readings, plus the verdict the CLI turns into an exit code."""

    policy: str
    n_sessions: int
    effective_sample_size: float
    ess_fraction: float
    max_weight_share: float
    weight_p99: float
    top1_agreement: float
    unseen_exposure: float
    usable: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def assess_overlap(
    log: InteractionLog,
    trace: PolicyTrace,
    weight: np.ndarray,
    min_ess_fraction: float,
    max_weight_share: float,
) -> OverlapReport:
    """Measure overlap and decide whether any offline estimate from this log is honest."""
    weight = np.asarray(weight, dtype=np.float64)
    if weight.shape[0] != log.n_sessions:
        raise ValueError("weights must have one entry per session")
    total = float(weight.sum())
    share = float(weight.max() / total) if total > 0 else 1.0
    fraction = ess_fraction(weight)
    agreement = float((trace.top1 == log.slate[:, 0]).mean())

    marginals = trace.marginals
    shown_first = np.zeros(int(log.candidates.max()) + 1, dtype=bool)
    shown_first[log.slate_items()[:, 0]] = True
    never_shown = ~shown_first[log.candidates]
    unseen = float((marginals[:, 0, :] * never_shown).sum(axis=1).mean())

    reasons = []
    if fraction < min_ess_fraction:
        reasons.append(
            f"effective sample size is {fraction:.2%} of the log, below the "
            f"{min_ess_fraction:.2%} floor"
        )
    if share > max_weight_share:
        reasons.append(
            f"one session carries {share:.2%} of the weight, above the "
            f"{max_weight_share:.2%} ceiling"
        )
    return OverlapReport(
        policy=trace.name,
        n_sessions=log.n_sessions,
        effective_sample_size=effective_sample_size(weight),
        ess_fraction=fraction,
        max_weight_share=share,
        weight_p99=float(np.quantile(weight, 0.99)),
        top1_agreement=agreement,
        unseen_exposure=unseen,
        usable=not reasons,
        reason="; ".join(reasons) if reasons else "overlap is sufficient for an offline estimate",
    )
