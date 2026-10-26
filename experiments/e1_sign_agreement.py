"""E1: how often does naive replay disagree with the truth about which policy is better?

This is the experiment the repository leads with, and the brief that commissioned it predicted
the wrong answer. The prediction was a symmetric failure: naive replay flatters a policy that
agrees with the deployed ranker and punishes one that disagrees. The measurement says the
failure is one sided. Naive replay comes in above the truth for almost every candidate, because
the sessions it keeps are selected on agreement with the logging policy and agreement with the
logging policy is itself correlated with reward. There is no corresponding population of
sessions that would drag a divergent policy's estimate down, because those sessions were
discarded rather than down weighted.

Two numbers come out of this. The sign disagreement rate over decisive scenarios, which is the
share of cases where a team acting on naive replay would have made the opposite call. And the
worst overclaim, which is the single scenario where the gap between what replay said and what
was true is largest.
"""

from __future__ import annotations

import numpy as np

from experiments.common import (
    Timer,
    experiment_fixture,
    header,
    provenance,
    scenario_policies,
    write_result,
)
from replay_gap.config import load_config
from replay_gap.estimators.inputs import build_estimator_inputs
from replay_gap.evaluation import trace_policy
from replay_gap.policies.baselines import IncumbentPolicy
from replay_gap.ranking import permutation_index
from replay_gap.simulator.ground_truth import true_value

NAME = "e1_sign_agreement"


def run() -> dict:
    config = load_config()
    n_sessions = config.experiments.e1_sessions
    n_scenarios = config.experiments.e1_scenarios
    decisive_threshold = config.experiments.e1_decisive_lift_pct / 100.0

    fixture = experiment_fixture(config, n_sessions)
    log = fixture.log
    perms = permutation_index(log.n_candidates, log.n_positions)

    incumbent = IncumbentPolicy(fixture.world, fixture.logging_policy, config.policies.temperature)
    baseline_true = true_value(
        fixture.world, log, trace_policy(incumbent, log, perms).logits, perms
    )
    logged_mean_reward = float(log.reward.mean())

    rows = []
    with Timer("scenarios") as timer:
        for policy in scenario_policies(
            fixture, n_scenarios, seed=config.seed + 1, temperature=config.policies.temperature
        ):
            trace = trace_policy(policy, log, perms)
            inputs = build_estimator_inputs(log, trace, clip=config.estimators.clip_weight)
            estimators, baselines = inputs.ratio_estimators()
            matched = int(inputs.matched.sum())
            truth = true_value(fixture.world, log, trace.logits, perms) / baseline_true - 1.0
            naive = (
                estimators["naive"].value() / logged_mean_reward - 1.0 if matched else float("nan")
            )
            snips = estimators["snips"].value() / baselines["snips"].value() - 1.0
            rows.append(
                {
                    "policy": policy.name,
                    "truth_weight": policy.truth_weight,
                    "incumbent_weight": policy.incumbent_weight,
                    "noise_weight": policy.noise_weight,
                    "true_lift": truth,
                    "naive_lift": naive,
                    "snips_lift": snips,
                    "match_rate": float(inputs.matched.mean()),
                    "matched_sessions": matched,
                }
            )

    evaluated = [row for row in rows if np.isfinite(row["naive_lift"])]
    decisive = [row for row in evaluated if abs(row["true_lift"]) >= decisive_threshold]
    naive_wrong_sign = [
        row for row in decisive if np.sign(row["naive_lift"]) != np.sign(row["true_lift"])
    ]
    snips_wrong_sign = [
        row for row in decisive if np.sign(row["snips_lift"]) != np.sign(row["true_lift"])
    ]
    overclaims = [row for row in evaluated if row["naive_lift"] > row["true_lift"]]
    worst = max(evaluated, key=lambda row: row["naive_lift"] - row["true_lift"])
    worst_shipped = max(
        (row for row in evaluated if row["true_lift"] < 0 and row["naive_lift"] > 0),
        key=lambda row: row["naive_lift"] - row["true_lift"],
        default=None,
    )

    payload = {
        "experiment": NAME,
        "question": (
            "How often does naive replay disagree with the truth about whether a candidate "
            "ranker beats the deployed one?"
        ),
        **provenance(config, n_sessions),
        "n_scenarios": n_scenarios,
        "decisive_threshold": decisive_threshold,
        "n_decisive": len(decisive),
        "naive_sign_disagreement_rate": len(naive_wrong_sign) / len(decisive),
        "naive_sign_disagreements": len(naive_wrong_sign),
        "snips_sign_disagreement_rate": len(snips_wrong_sign) / len(decisive),
        "snips_sign_disagreements": len(snips_wrong_sign),
        "naive_overclaim_rate": len(overclaims) / len(evaluated),
        "naive_overclaims": len(overclaims),
        "naive_mean_signed_error": float(
            np.mean([row["naive_lift"] - row["true_lift"] for row in evaluated])
        ),
        "snips_mean_signed_error": float(
            np.mean([row["snips_lift"] - row["true_lift"] for row in evaluated])
        ),
        "naive_mean_absolute_error": float(
            np.mean([abs(row["naive_lift"] - row["true_lift"]) for row in evaluated])
        ),
        "snips_mean_absolute_error": float(
            np.mean([abs(row["snips_lift"] - row["true_lift"]) for row in evaluated])
        ),
        "worst_overclaim_gap": worst["naive_lift"] - worst["true_lift"],
        "worst_overclaim_true_lift": worst["true_lift"],
        "worst_overclaim_naive_lift": worst["naive_lift"],
        "worst_shipped_true_lift": worst_shipped["true_lift"] if worst_shipped else None,
        "worst_shipped_naive_lift": worst_shipped["naive_lift"] if worst_shipped else None,
        "seconds": timer.seconds,
        "rows": rows,
    }
    write_result(NAME, payload)
    return payload


def report(payload: dict) -> None:
    header("E1: naive replay against the referee, over sampled candidate rankers")
    print(
        f"{payload['n_scenarios']} candidate rankers drawn uniformly over the three way simplex "
        f"of truth, incumbent and noise,"
    )
    print(f"each evaluated against the same {payload['n_sessions']:,} session log.")
    print()
    print(f"{'measure':<52}{'naive replay':>16}{'SNIPS':>16}")
    print("-" * 84)
    print(
        f"{'sign disagreements with the referee':<52}"
        f"{payload['naive_sign_disagreements']:>10} / {payload['n_decisive']:<3}"
        f"{payload['snips_sign_disagreements']:>10} / {payload['n_decisive']:<3}"
    )
    print(
        f"{'sign disagreement rate':<52}"
        f"{payload['naive_sign_disagreement_rate']:>15.1%} "
        f"{payload['snips_sign_disagreement_rate']:>15.1%}"
    )
    print(
        f"{'mean signed error (positive means overstated)':<52}"
        f"{payload['naive_mean_signed_error']:>15.2%} "
        f"{payload['snips_mean_signed_error']:>15.2%}"
    )
    print(
        f"{'mean absolute error':<52}"
        f"{payload['naive_mean_absolute_error']:>15.2%} "
        f"{payload['snips_mean_absolute_error']:>15.2%}"
    )
    print()
    print(
        f"naive replay came in above the truth in {payload['naive_overclaims']} of "
        f"{payload['n_scenarios']} scenarios ({payload['naive_overclaim_rate']:.0%})."
    )
    print(
        f"worst overclaim: a ranker whose true lift is "
        f"{payload['worst_overclaim_true_lift']:+.2%} read as "
        f"{payload['worst_overclaim_naive_lift']:+.2%}, a gap of "
        f"{payload['worst_overclaim_gap']:.2%}."
    )
    if payload["worst_shipped_true_lift"] is not None:
        print(
            f"worst case a team would have shipped: true lift "
            f"{payload['worst_shipped_true_lift']:+.2%}, replay said "
            f"{payload['worst_shipped_naive_lift']:+.2%}."
        )


if __name__ == "__main__":
    report(run())
