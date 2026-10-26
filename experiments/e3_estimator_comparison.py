"""E3: which estimator is actually best, and where does the recommended one lose?

Four estimators are run against the same logs, over independent replicates of the world, for
three candidate policies chosen to sit at different distances from the deployed ranker. Each
replicate draws a fresh log from a fresh seed, so the spread reported here is the spread a team
would see if it reran the same evaluation on a different week's data.

Three results are worth stating before the table, and the first two were not what this
repository expected to find.

The doubly robust estimator is the best of the four in most regimes, including the one where it
was predicted to fail. It loses in exactly one place: a policy served greedily, where its outcome
model is extrapolating onto slates the deployed ranker almost never served. There it is beaten by
the plain self normalised estimator, and the repository publishes that rather than the
recommendation it started with.

Naive replay cannot see the serving temperature at all. It keeps the sessions where the
candidate's top choice was served and averages their reward, and neither of those steps involves
anything below the first slot. So a policy served greedily and the same policy served
stochastically get an identical naive estimate from an identical set of sessions, while their
true values differ by more than ten points. The experiment reports that gap, because it is the
cleanest statement of what the estimator is structurally unable to represent.

The third result is the one that would cost money. For the popularity baseline, which is
genuinely worse than production, naive replay gets the sign wrong on every single replicate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from experiments.common import Timer, header, provenance, write_result
from replay_gap.config import load_config
from replay_gap.diagnostics.ess import ess_fraction
from replay_gap.estimators.doubly_robust import fit_reward_model, predict_candidate_rewards
from replay_gap.estimators.inputs import build_estimator_inputs
from replay_gap.evaluation import trace_policy
from replay_gap.fixtures import build_fixture
from replay_gap.policies.baselines import (
    ContrarianPolicy,
    IncumbentPolicy,
    OraclePolicy,
    PopularityPolicy,
    click_counts_from_log,
)
from replay_gap.ranking import permutation_index
from replay_gap.simulator.ground_truth import true_value

NAME = "e3_estimator_comparison"
ESTIMATORS = ("naive", "ips", "snips", "dr")
ESTIMATOR_LABELS = {
    "naive": "naive replay",
    "ips": "IPS",
    "snips": "SNIPS",
    "dr": "doubly robust",
}


def candidate_policies(fixture, config) -> dict:
    """Five candidates spanning distance from the deployed ranker and greed of serving.

    The two sharp variants rank identically to their ordinary counterparts. Only the serving
    temperature differs, which is why they are the pair that exposes what naive replay cannot
    see.
    """
    counts = click_counts_from_log(
        fixture.world.n_items, fixture.log.slate_items(), fixture.log.clicks
    )
    return {
        "popularity": PopularityPolicy(counts, config.policies.temperature),
        "oracle": OraclePolicy(fixture.world, config.policies.temperature),
        "oracle_sharp": OraclePolicy(fixture.world, config.policies.sharp_temperature),
        "contrarian_lift": ContrarianPolicy(
            fixture.world, fixture.logging_policy, config.policies.temperature
        ),
        "contrarian_sharp": ContrarianPolicy(
            fixture.world, fixture.logging_policy, config.policies.sharp_temperature
        ),
    }


SERVING_PAIRS = (("oracle", "oracle_sharp"), ("contrarian_lift", "contrarian_sharp"))


def run() -> dict:
    config = load_config()
    n_sessions = config.experiments.e3_sessions
    replicates = config.experiments.e3_replicates
    rows = []

    with Timer("replicates") as timer:
        for replicate in range(replicates):
            seeded = replace(config, seed=config.seed + 1000 + replicate)
            fixture = build_fixture(seeded, n_sessions=n_sessions)
            log = fixture.log
            perms = permutation_index(log.n_candidates, log.n_positions)
            model = fit_reward_model(fixture.world, log, config.reward_model)
            candidate_rewards = predict_candidate_rewards(fixture.world, log, model)

            incumbent = IncumbentPolicy(
                fixture.world, fixture.logging_policy, config.policies.temperature
            )
            baseline_true = true_value(
                fixture.world, log, trace_policy(incumbent, log, perms).logits, perms
            )
            logged_mean = float(log.reward.mean())

            for name, policy in candidate_policies(fixture, config).items():
                trace = trace_policy(policy, log, perms)
                inputs = build_estimator_inputs(
                    log, trace, config.estimators.clip_weight, candidate_rewards
                )
                estimators, baselines = inputs.ratio_estimators()
                truth = true_value(fixture.world, log, trace.logits, perms) / baseline_true - 1.0
                row = {
                    "replicate": replicate,
                    "policy": name,
                    "true_lift": truth,
                    "ess_fraction": ess_fraction(inputs.weight),
                    "match_rate": float(inputs.matched.mean()),
                }
                for estimator in ESTIMATORS:
                    denominator = (
                        logged_mean if estimator == "naive" else baselines[estimator].value()
                    )
                    value = estimators[estimator].value() / denominator - 1.0
                    row[f"{estimator}_lift"] = value
                    row[f"{estimator}_error"] = value - truth
                rows.append(row)

    summary = {}
    for name in candidate_policies(build_fixture(config, n_sessions=64), config):
        subset = [row for row in rows if row["policy"] == name]
        entry = {
            "policy": name,
            "replicates": len(subset),
            "mean_true_lift": float(np.mean([row["true_lift"] for row in subset])),
            "mean_ess_fraction": float(np.mean([row["ess_fraction"] for row in subset])),
            "mean_match_rate": float(np.mean([row["match_rate"] for row in subset])),
        }
        for estimator in ESTIMATORS:
            errors = np.array([row[f"{estimator}_error"] for row in subset])
            entry[f"{estimator}_bias"] = float(errors.mean())
            entry[f"{estimator}_rmse"] = float(np.sqrt((errors**2).mean()))
            entry[f"{estimator}_sign_errors"] = int(
                sum(
                    np.sign(row[f"{estimator}_lift"]) != np.sign(row["true_lift"]) for row in subset
                )
            )
        entry["best_by_rmse"] = min(ESTIMATORS, key=lambda estimator: entry[f"{estimator}_rmse"])
        summary[name] = entry

    serving_blindness = []
    for mild, sharp in SERVING_PAIRS:
        serving_blindness.append(
            {
                "mild": mild,
                "sharp": sharp,
                "naive_lift_mild": summary[mild]["naive_bias"] + summary[mild]["mean_true_lift"],
                "naive_lift_sharp": summary[sharp]["naive_bias"] + summary[sharp]["mean_true_lift"],
                "true_lift_mild": summary[mild]["mean_true_lift"],
                "true_lift_sharp": summary[sharp]["mean_true_lift"],
                "true_lift_gap": summary[sharp]["mean_true_lift"] - summary[mild]["mean_true_lift"],
                "naive_lift_gap": (summary[sharp]["naive_bias"] + summary[sharp]["mean_true_lift"])
                - (summary[mild]["naive_bias"] + summary[mild]["mean_true_lift"]),
            }
        )
    for entry in serving_blindness:
        entry["naive_estimates_identical"] = bool(abs(entry["naive_lift_gap"]) < 1e-9)
    widest_blindness = max(serving_blindness, key=lambda entry: entry["true_lift_gap"])

    dr_losses = [name for name, entry in summary.items() if entry["best_by_rmse"] != "dr"]
    snips_wins = [name for name, entry in summary.items() if entry["best_by_rmse"] == "snips"]
    always_wrong_sign = [
        name for name, entry in summary.items() if entry["naive_sign_errors"] == entry["replicates"]
    ]
    payload = {
        "experiment": NAME,
        "question": "Which estimator is actually best, and where does the recommended one lose?",
        **provenance(config, n_sessions),
        "replicates": replicates,
        "estimators": list(ESTIMATORS),
        "summary": summary,
        "dr_not_best_for": dr_losses,
        "snips_best_for": snips_wins,
        "naive_always_wrong_sign_for": always_wrong_sign,
        "serving_blindness": serving_blindness,
        "widest_serving_blindness": widest_blindness,
        "n_policies": len(summary),
        "seconds": timer.seconds,
        "rows": rows,
    }
    write_result(NAME, payload)
    return payload


def report(payload: dict) -> None:
    header("E3: four estimators, over independent replicates of the world")
    print(
        f"{payload['replicates']} replicates, each a fresh world and a fresh "
        f"{payload['n_sessions']:,} session log. Error is against the referee."
    )
    for name, entry in payload["summary"].items():
        print()
        print(
            f"{name}  (true lift {entry['mean_true_lift']:+.2%}, effective sample "
            f"{entry['mean_ess_fraction']:.1%} of the log, naive replay keeps "
            f"{entry['mean_match_rate']:.1%})"
        )
        print(f"  {'estimator':<16}{'bias':>12}{'RMSE':>12}{'sign errors':>14}")
        print("  " + "-" * 54)
        for estimator in payload["estimators"]:
            marker = "  <- best" if entry["best_by_rmse"] == estimator else ""
            print(
                f"  {ESTIMATOR_LABELS[estimator]:<16}{entry[f'{estimator}_bias']:>+12.2%}"
                f"{entry[f'{estimator}_rmse']:>12.2%}"
                f"{entry[f'{estimator}_sign_errors']:>8} / {entry['replicates']:<3}{marker}"
            )
    print()
    if payload["dr_not_best_for"]:
        print(
            "The doubly robust estimator is not best for: "
            + ", ".join(payload["dr_not_best_for"])
            + ". Its outcome model is fitted on the same confounded log, so on slates the "
            "deployed ranker rarely served it is extrapolating."
        )
    else:
        print("The doubly robust estimator was best for every policy in this run.")
    for name in payload["naive_always_wrong_sign_for"]:
        entry = payload["summary"][name]
        print(
            f"For {name}, true lift {entry['mean_true_lift']:+.2%}, naive replay got the sign "
            f"wrong on all {entry['replicates']} replicates."
        )
    print()
    print("What naive replay cannot see: the same ranker, served two ways")
    print(f"  {'pair':<34}{'true lift':>12}{'naive says':>13}")
    print("  " + "-" * 57)
    for entry in payload["serving_blindness"]:
        print(
            f"  {entry['mild']:<34}{entry['true_lift_mild']:>+12.2%}"
            f"{entry['naive_lift_mild']:>+13.2%}"
        )
        print(
            f"  {entry['sharp']:<34}{entry['true_lift_sharp']:>+12.2%}"
            f"{entry['naive_lift_sharp']:>+13.2%}"
        )
    widest = payload["widest_serving_blindness"]
    print()
    print(
        f"{widest['mild']} and {widest['sharp']} rank identically and differ only in how "
        f"greedily they are served."
    )
    if widest["naive_estimates_identical"]:
        print(
            f"Their true values differ by {widest['true_lift_gap']:.2%}. Naive replay returns "
            f"the same estimate for both, to every digit, because it reads only the first slot "
            f"of the sessions where that slot already matched."
        )
    else:
        print(
            f"Their true values differ by {widest['true_lift_gap']:.2%}, and naive replay's "
            f"estimates differ by {abs(widest['naive_lift_gap']):.2%}."
        )
    refusal_zone = [
        name
        for name, entry in payload["summary"].items()
        if entry["best_by_rmse"] == "naive" and entry["mean_ess_fraction"] < 0.02
    ]
    if refusal_zone:
        print()
        print(
            "Worth saying plainly: naive replay has the smallest error for "
            + ", ".join(refusal_zone)
            + ". That is the regime where the effective sample size is below the floor and the "
            "tool refuses to answer at all, so it is an estimator landing near the truth by "
            "accident in a case where nothing here should be trusted."
        )


if __name__ == "__main__":
    report(run())
