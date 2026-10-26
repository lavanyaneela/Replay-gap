"""E4: what is the smallest true lift this log could actually have detected?

Teams ship on single digit percentage improvements. This experiment asks what a fifty thousand
session log can resolve, and reports the answer next to the lift a team would typically act on,
so the comparison is in the reader's hands rather than implied.

The number is derived from the bootstrap spread of the corrected estimate, converted to a
minimum detectable lift at the configured alpha and power. It is reported per policy, because it
is not a property of the log: it is a property of the log and the question together. The same
fifty thousand sessions resolve a tenth of a percent for a policy that barely moves and several
percent for one that does.

Two framings go in the output. The first is how many sessions would be needed to resolve a one
percent true lift for each policy, which is usually a number nobody has. The second is receipts
rule four applied honestly: where a count comes out at zero, the claim is priced at the
resolution floor of one over the sample size rather than at zero, and the sample that would be
needed to support the stronger claim is stated.
"""

from __future__ import annotations

import math

import numpy as np

from experiments.common import Timer, experiment_fixture, header, provenance, write_result
from replay_gap.config import load_config
from replay_gap.diagnostics.detectable_lift import (
    minimum_detectable_lift,
    resolution_floor,
    sessions_needed_for_rate,
)
from replay_gap.estimators.bootstrap import bootstrap_ratio_lifts
from replay_gap.estimators.doubly_robust import fit_reward_model, predict_candidate_rewards
from replay_gap.estimators.inputs import build_estimator_inputs
from replay_gap.evaluation import trace_policy
from replay_gap.pipeline import build_policies
from replay_gap.policies.baselines import IncumbentPolicy
from replay_gap.ranking import permutation_index
from replay_gap.simulator.ground_truth import true_value

NAME = "e4_detectable_lift"
TARGET_LIFT = 0.01


def run() -> dict:
    config = load_config()
    n_sessions = config.log.n_sessions
    fixture = experiment_fixture(config, n_sessions)
    log = fixture.log
    perms = permutation_index(log.n_candidates, log.n_positions)
    model = fit_reward_model(fixture.world, log, config.reward_model)
    candidate_rewards = predict_candidate_rewards(fixture.world, log, model)

    incumbent = IncumbentPolicy(fixture.world, fixture.logging_policy, config.policies.temperature)
    baseline_true = true_value(
        fixture.world, log, trace_policy(incumbent, log, perms).logits, perms
    )

    rows = []
    with Timer("policies") as timer:
        for name, policy in build_policies(config, fixture, with_ranker=False).items():
            trace = trace_policy(policy, log, perms)
            inputs = build_estimator_inputs(
                log, trace, config.estimators.clip_weight, candidate_rewards
            )
            estimators, baselines = inputs.ratio_estimators()
            intervals = bootstrap_ratio_lifts(
                estimators,
                baselines,
                resamples=config.estimators.bootstrap_resamples,
                rng=np.random.default_rng(config.estimators.bootstrap_seed),
                alpha=config.diagnostics.alpha,
            )
            truth = true_value(fixture.world, log, trace.logits, perms) / baseline_true - 1.0
            detectable = minimum_detectable_lift(
                intervals["snips"].standard_error,
                alpha=config.diagnostics.alpha,
                power=config.diagnostics.power,
            )
            naive_detectable = minimum_detectable_lift(
                intervals["naive"].standard_error,
                alpha=config.diagnostics.alpha,
                power=config.diagnostics.power,
            )
            # Variance of a mean falls as one over n, so the standard error falls as one over
            # root n, and the sessions needed scale as the square of the ratio of the lifts.
            scale = (detectable / TARGET_LIFT) ** 2 if detectable > 0 else 0.0
            rows.append(
                {
                    "policy": name,
                    "true_lift": truth,
                    "snips_lift": intervals["snips"].point,
                    "snips_standard_error": intervals["snips"].standard_error,
                    "minimum_detectable_lift": detectable,
                    "naive_minimum_detectable_lift": naive_detectable,
                    "ess_fraction": inputs.weight.sum() ** 2
                    / (inputs.weight**2).sum()
                    / log.n_sessions,
                    "sessions_for_one_percent": int(math.ceil(n_sessions * scale)),
                    "degenerate": bool(detectable == 0.0),
                    "true_lift_is_detectable": bool(abs(truth) >= detectable > 0.0),
                }
            )

    # The incumbent re-evaluated against its own log has importance weights of exactly one, so
    # every resample returns exactly the logged mean and the spread is exactly zero. That is
    # correct and it is not a resolution anybody can use, so it is excluded from best and worst
    # rather than published as a log that can detect any lift at all.
    informative = [row for row in rows if not row["degenerate"]]
    undetectable = [row for row in informative if not row["true_lift_is_detectable"]]
    worst = max(informative, key=lambda row: row["minimum_detectable_lift"])
    best = min(informative, key=lambda row: row["minimum_detectable_lift"])
    floor = resolution_floor(n_sessions)

    payload = {
        "experiment": NAME,
        "question": "What is the smallest true lift this log could actually have detected?",
        **provenance(config, n_sessions),
        "alpha": config.diagnostics.alpha,
        "power": config.diagnostics.power,
        "target_lift": TARGET_LIFT,
        "resolution_floor": floor,
        "sessions_for_one_in_ten_thousand": sessions_needed_for_rate(0.0001),
        "n_policies": len(rows),
        "n_informative_policies": len(informative),
        "degenerate_policies": [row["policy"] for row in rows if row["degenerate"]],
        "n_undetectable": len(undetectable),
        "undetectable_policies": [row["policy"] for row in undetectable],
        "worst_policy": worst["policy"],
        "worst_minimum_detectable_lift": worst["minimum_detectable_lift"],
        "worst_sessions_for_one_percent": worst["sessions_for_one_percent"],
        "best_policy": best["policy"],
        "best_minimum_detectable_lift": best["minimum_detectable_lift"],
        "median_minimum_detectable_lift": float(
            np.median([row["minimum_detectable_lift"] for row in informative])
        ),
        "policies_above_target": sum(
            1 for row in informative if row["minimum_detectable_lift"] > TARGET_LIFT
        ),
        "seconds": timer.seconds,
        "by_policy": {row["policy"]: row for row in rows},
        "rows": rows,
    }
    write_result(NAME, payload)
    return payload


def report(payload: dict) -> None:
    header("E4: the smallest lift this log could have detected")
    print(
        f"{payload['n_sessions']:,} sessions, two sided at alpha {payload['alpha']}, "
        f"power {payload['power']:.0%}. Derived from the bootstrap spread of SNIPS."
    )
    print()
    print(
        f"{'policy':<18}{'true lift':>12}{'SNIPS says':>13}{'detectable':>13}"
        f"{'detected?':>11}{'sessions for 1%':>18}"
    )
    print("-" * 85)
    for row in payload["rows"]:
        if row["degenerate"]:
            print(
                f"{row['policy']:<18}{row['true_lift']:>+12.2%}{row['snips_lift']:>+13.2%}"
                f"{'exact':>13}{'n/a':>11}{'n/a':>18}"
            )
            continue
        print(
            f"{row['policy']:<18}{row['true_lift']:>+12.2%}{row['snips_lift']:>+13.2%}"
            f"{row['minimum_detectable_lift']:>13.2%}"
            f"{('yes' if row['true_lift_is_detectable'] else 'no'):>11}"
            f"{row['sessions_for_one_percent']:>18,}"
        )
    if payload["degenerate_policies"]:
        print()
        print(
            "Marked exact: "
            + ", ".join(payload["degenerate_policies"])
            + ". Its importance weights are exactly one, so every bootstrap resample returns "
            "exactly the logged mean. The spread is genuinely zero and it is not a resolution "
            "anybody can use, so it is left out of the summary below."
        )
    print()
    print(
        f"{payload['policies_above_target']} of {payload['n_informative_policies']} policies "
        f"have a minimum detectable lift above the {payload['target_lift']:.0%} a team would "
        f"typically ship on."
    )
    print(
        f"The best case is {payload['best_policy']} at "
        f"{payload['best_minimum_detectable_lift']:.2%}, where the candidate barely moves away "
        f"from the deployed ranker and the log stays almost whole."
    )
    print(
        f"The worst is {payload['worst_policy']} at "
        f"{payload['worst_minimum_detectable_lift']:.2%}. Resolving a one percent lift there "
        f"would need about {payload['worst_sessions_for_one_percent']:,} sessions, "
        f"{payload['worst_sessions_for_one_percent'] / payload['n_sessions']:.0f} times this log."
    )
    if payload["undetectable_policies"]:
        print(
            f"{payload['n_undetectable']} of {payload['n_informative_policies']} policies have "
            f"a true lift smaller than what this log can resolve: "
            + ", ".join(payload["undetectable_policies"])
            + "."
        )
    print()
    print(
        f"On a measured zero: nothing observed zero times in {payload['n_sessions']:,} sessions "
        f"is zero. It is below {payload['resolution_floor']:.2e}, which is one over the sample "
        f"size. A claim about a one in ten thousand event needs about "
        f"{payload['sessions_for_one_in_ten_thousand']:,} sessions before the sample can "
        f"distinguish it from never."
    )


if __name__ == "__main__":
    report(run())
