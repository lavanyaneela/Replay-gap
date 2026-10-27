"""The evaluation itself: log in, one verdict per candidate policy out.

Everything the CLI does goes through here, and the order matters. Overlap is assessed before any
estimate is printed, because an estimate printed next to a warning gets read and the warning does
not. When overlap fails the pipeline does not compute a number and label it unreliable; it
refuses, and the refusal carries the remedy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from replay_gap.config import Config
from replay_gap.diagnostics.detectable_lift import minimum_detectable_lift
from replay_gap.diagnostics.overlap import OverlapReport, assess_overlap
from replay_gap.estimators.bootstrap import BootstrapInterval, bootstrap_ratio_lifts
from replay_gap.estimators.doubly_robust import fit_reward_model, predict_candidate_rewards
from replay_gap.estimators.inputs import build_estimator_inputs
from replay_gap.evaluation import PolicyTrace, trace_policy
from replay_gap.fixtures import Fixture
from replay_gap.policies import (
    LightGBMRankerPolicy,
    build_baseline_policies,
    click_counts_from_log,
    train_lightgbm_ranker,
)
from replay_gap.policies.base import Policy
from replay_gap.ranking import permutation_index
from replay_gap.simulator.ground_truth import true_value

EXIT_POSITIVE = 0
EXIT_NOT_POSITIVE = 1
EXIT_REFUSED = 2
EXIT_TOOL_ERROR = 3

ESTIMATOR_NAMES: tuple[str, ...] = ("naive", "naive_first_slot", "ips", "snips", "dr")


@dataclass(frozen=True)
class PolicyResult:
    """One candidate policy, judged four ways and refereed once."""

    policy: str
    n_sessions: int
    logged_value: float
    true_value: float
    true_lift: float
    naive_match_rate: float
    naive_matched_sessions: int
    clipped_share: float
    lifts: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    standard_errors: dict[str, float] = field(default_factory=dict)
    overlap: OverlapReport | None = None
    minimum_detectable_lift: float = float("nan")
    decision: str = "unknown"
    exit_code: int = EXIT_TOOL_ERROR
    reason: str = ""

    def error(self, estimator: str) -> float:
        """Signed error of one estimator against the referee, in lift points."""
        return self.lifts[estimator] - self.true_lift

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["overlap"] = self.overlap.as_dict() if self.overlap else None
        return payload


def _decide(overlap: OverlapReport, interval: BootstrapInterval) -> tuple[str, int, str]:
    """Turn the diagnostics and the corrected interval into one of the documented exit codes."""
    if not overlap.usable:
        refusal = overlap.refusal()
        return "refused", EXIT_REFUSED, f"{overlap.reason}. Remedy: {refusal.remedy}."
    if not np.isfinite(interval.low) or not np.isfinite(interval.high):
        return (
            "refused",
            EXIT_REFUSED,
            "the bootstrap produced no finite interval, so no corrected estimate exists",
        )
    if interval.low > 0:
        return (
            "positive",
            EXIT_POSITIVE,
            f"corrected lift of {interval.point:+.2%} with the interval entirely above zero",
        )
    if interval.high < 0:
        return (
            "not_positive",
            EXIT_NOT_POSITIVE,
            f"corrected lift of {interval.point:+.2%} with the interval entirely below zero",
        )
    return (
        "not_positive",
        EXIT_NOT_POSITIVE,
        f"corrected lift of {interval.point:+.2%} is not distinguishable from zero",
    )


def compare_exit_code(results: list[PolicyResult]) -> int:
    """The exit code for a whole comparison, in priority order.

    A refusal anywhere outranks a win anywhere. A caller running this in a pipeline needs to
    know that some question went unanswered even when a different policy did win, because the
    unanswered one is the one somebody will otherwise ship on the strength of a naive replay.
    """
    if any(result.decision == "refused" for result in results):
        return EXIT_REFUSED
    if any(result.decision == "positive" for result in results):
        return EXIT_POSITIVE
    return EXIT_NOT_POSITIVE


def evaluate_policy(
    config: Config,
    fixture: Fixture,
    policy: Policy,
    candidate_rewards: np.ndarray | None,
    perms: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    baseline_true_value: float | None = None,
) -> PolicyResult:
    """Run every estimator, the referee and the diagnostics over one candidate policy."""
    log = fixture.log
    if perms is None:
        perms = permutation_index(log.n_candidates, log.n_positions)
    if rng is None:
        rng = np.random.default_rng(config.estimators.bootstrap_seed)

    trace: PolicyTrace = trace_policy(policy, log, perms)
    inputs = build_estimator_inputs(
        log,
        trace,
        clip=config.estimators.clip_weight,
        candidate_rewards=candidate_rewards,
    )
    overlap = assess_overlap(
        log,
        trace,
        inputs.weight,
        min_ess_fraction=config.diagnostics.min_ess_fraction,
        max_weight_share=config.diagnostics.max_weight_share,
    )

    estimators, baselines = inputs.ratio_estimators()
    intervals = bootstrap_ratio_lifts(
        estimators,
        baselines,
        resamples=config.estimators.bootstrap_resamples,
        rng=rng,
        alpha=config.diagnostics.alpha,
    )

    if baseline_true_value is None:
        incumbent_logits = fixture.logging_policy.logits(
            fixture.world, log.context_id, log.candidates
        )
        baseline_true_value = true_value(fixture.world, log, incumbent_logits, perms)
    candidate_true_value = true_value(fixture.world, log, trace.logits, perms)

    decision, exit_code, reason = _decide(overlap, intervals["snips"])
    return PolicyResult(
        policy=policy.name,
        n_sessions=log.n_sessions,
        logged_value=float(inputs.reward.mean()),
        true_value=candidate_true_value,
        true_lift=candidate_true_value / baseline_true_value - 1.0,
        naive_match_rate=float(inputs.matched.mean()),
        naive_matched_sessions=int(inputs.matched.sum()),
        clipped_share=inputs.clipped_share,
        lifts={name: interval.point for name, interval in intervals.items()},
        intervals={name: (i.low, i.high) for name, i in intervals.items()},
        standard_errors={name: i.standard_error for name, i in intervals.items()},
        overlap=overlap,
        minimum_detectable_lift=minimum_detectable_lift(
            intervals["snips"].standard_error,
            alpha=config.diagnostics.alpha,
            power=config.diagnostics.power,
        ),
        decision=decision,
        exit_code=exit_code,
        reason=reason,
    )


@dataclass(frozen=True)
class EvaluationRun:
    """The whole evaluation: the log it read, the policies it judged, and how it judged them."""

    config_digest: str
    n_sessions: int
    n_candidates: int
    n_positions: int
    logged_mean_reward: float
    baseline_true_value: float
    results: list[PolicyResult]

    def by_name(self, name: str) -> PolicyResult:
        for result in self.results:
            if result.policy == name:
                return result
        known = ", ".join(r.policy for r in self.results)
        raise KeyError(f"no result for policy {name}; evaluated: {known}")

    @property
    def policy_names(self) -> list[str]:
        return [r.policy for r in self.results]

    def as_dict(self) -> dict:
        return {
            "config_digest": self.config_digest,
            "n_sessions": self.n_sessions,
            "n_candidates": self.n_candidates,
            "n_positions": self.n_positions,
            "logged_mean_reward": self.logged_mean_reward,
            "baseline_true_value": self.baseline_true_value,
            "results": [r.as_dict() for r in self.results],
        }


def build_policies(config: Config, fixture: Fixture, with_ranker: bool = True) -> dict[str, Policy]:
    """The baseline set plus, optionally, the LambdaMART ranker trained on this log."""
    counts = click_counts_from_log(
        fixture.world.n_items, fixture.log.slate_items(), fixture.log.clicks
    )
    policies = build_baseline_policies(
        fixture.world,
        fixture.logging_policy,
        counts,
        config.policies.temperature,
        config.policies.sharp_temperature,
        config.policies.clone_noise,
    )
    if with_ranker:
        model = train_lightgbm_ranker(fixture.world, fixture.log, config.ranker)
        ranker = LightGBMRankerPolicy(fixture.world, model, config.policies.temperature)
        policies[ranker.name] = ranker
    return policies


def run_evaluation(
    config: Config,
    fixture: Fixture,
    policies: dict[str, Policy],
    with_reward_model: bool = True,
) -> EvaluationRun:
    """Evaluate every policy against the log, sharing the referee and the outcome model."""
    from replay_gap.fixtures import config_digest

    log = fixture.log
    perms = permutation_index(log.n_candidates, log.n_positions)
    candidate_rewards = None
    if with_reward_model:
        model = fit_reward_model(fixture.world, log, config.reward_model)
        candidate_rewards = predict_candidate_rewards(fixture.world, log, model)

    incumbent_logits = fixture.logging_policy.logits(fixture.world, log.context_id, log.candidates)
    baseline_true_value = true_value(fixture.world, log, incumbent_logits, perms)

    results = [
        evaluate_policy(
            config,
            fixture,
            policy,
            candidate_rewards,
            perms=perms,
            rng=np.random.default_rng(config.estimators.bootstrap_seed),
            baseline_true_value=baseline_true_value,
        )
        for policy in policies.values()
    ]
    return EvaluationRun(
        config_digest=config_digest(config),
        n_sessions=log.n_sessions,
        n_candidates=log.n_candidates,
        n_positions=log.n_positions,
        logged_mean_reward=float(log.reward.mean()),
        baseline_true_value=baseline_true_value,
        results=results,
    )
