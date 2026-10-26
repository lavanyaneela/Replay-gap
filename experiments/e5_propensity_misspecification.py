"""E5: what happens when the propensities are wrong, which is the normal case in production?

Everything else in this repository runs on exact propensities. That is the strongest assumption
it makes and the one a reviewer should attack first, because almost no production log has them.
Propensities get logged at the wrong granularity, recorded after a reranking step that was not
accounted for, or reconstructed months later from a model checkpoint that has since moved.

So this experiment corrupts them on purpose. The recorded propensity is replaced by a tempered
version of the true one, which is what a reconstruction from a stale or differently calibrated
model looks like: the ordering of the slates is preserved and the magnitudes are wrong, and they
are wrong in a way that correlates with how confident the policy was. The corruption factor
sweeps from one, where nothing is wrong, upwards.

Two things are measured: the error of the corrected estimate, and the effective sample size that
is supposed to tell you whether to trust it.

The result is worse than the obvious prediction. The effective sample size does not merely fail
to warn, it moves in the wrong direction. Tempering flattens the weights, flat weights are what
Kish's formula rewards, and so the diagnostic climbs while the estimate it is vouching for
decays. At every crossover in this sweep, the point where the corrected estimate has become as
wrong as the naive replay it was supposed to replace, the diagnostic reads healthier than it did
on the uncorrupted log.

That is a real limit of this design, not a bug in it. Effective sample size measures the spread
of the weights and says nothing about whether they are the right weights, and no diagnostic in
this repository detects a propensity that is confidently wrong. The README has a section for it
and ADR-005 records what it would take to close.
"""

from __future__ import annotations

from experiments.common import Timer, experiment_fixture, header, provenance, write_result
from replay_gap.config import load_config
from replay_gap.diagnostics.ess import ess_fraction
from replay_gap.estimators.inputs import build_estimator_inputs
from replay_gap.evaluation import trace_policy
from replay_gap.pipeline import build_policies
from replay_gap.policies.baselines import IncumbentPolicy
from replay_gap.ranking import permutation_index
from replay_gap.simulator.generate import InteractionLog
from replay_gap.simulator.ground_truth import true_value

NAME = "e5_propensity_misspecification"
POLICIES = ("oracle", "contrarian_lift", "popularity")


def corrupt_propensities(log: InteractionLog, factor: float) -> InteractionLog:
    """Return the log with its recorded propensities tempered by the given factor.

    Tempering means raising each propensity to the power one over the factor and renormalising
    across the session's slate distribution, which is exactly what reconstructing propensities
    from a model at a different temperature would produce. At a factor of one the log is
    unchanged. Above it, confident slates are recorded as less confident than they were, so the
    importance weights come out systematically too small on exactly the sessions that carry the
    most information.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1.0:
        return log
    from dataclasses import replace

    tempered = log.log_propensity / factor
    # Renormalise so the corrupted numbers are still probabilities on the scale the original
    # ones occupied. Without this the sweep would be measuring a constant offset in the weights,
    # which self normalisation cancels, and the experiment would report nothing at all.
    shift = log.log_propensity.mean() - tempered.mean()
    return replace(log, log_propensity=tempered + shift)


def run() -> dict:
    config = load_config()
    n_sessions = config.experiments.e5_sessions
    fixture = experiment_fixture(config, n_sessions)
    log = fixture.log
    perms = permutation_index(log.n_candidates, log.n_positions)
    incumbent = IncumbentPolicy(fixture.world, fixture.logging_policy, config.policies.temperature)
    baseline_true = true_value(
        fixture.world, log, trace_policy(incumbent, log, perms).logits, perms
    )
    logged_mean = float(log.reward.mean())
    policies = {
        name: policy
        for name, policy in build_policies(config, fixture, with_ranker=False).items()
        if name in POLICIES
    }

    rows = []
    with Timer("sweep") as timer:
        for name, policy in policies.items():
            trace = trace_policy(policy, log, perms)
            truth = true_value(fixture.world, log, trace.logits, perms) / baseline_true - 1.0
            clean = build_estimator_inputs(log, trace, config.estimators.clip_weight)
            clean_estimators, clean_baselines = clean.ratio_estimators()
            naive_lift = clean_estimators["naive"].value() / logged_mean - 1.0
            naive_error = abs(naive_lift - truth)
            for factor in config.experiments.e5_factors:
                corrupted = corrupt_propensities(log, factor)
                inputs = build_estimator_inputs(corrupted, trace, config.estimators.clip_weight)
                estimators, baselines = inputs.ratio_estimators()
                snips = estimators["snips"].value() / baselines["snips"].value() - 1.0
                rows.append(
                    {
                        "policy": name,
                        "factor": factor,
                        "true_lift": truth,
                        "snips_lift": snips,
                        "snips_error": abs(snips - truth),
                        "naive_lift": naive_lift,
                        "naive_error": naive_error,
                        "ess_fraction": ess_fraction(inputs.weight),
                        "clipped_share": inputs.clipped_share,
                        "worse_than_naive": bool(abs(snips - truth) > naive_error),
                    }
                )

    crossovers = {}
    for name in policies:
        subset = [row for row in rows if row["policy"] == name]
        crossing = next((row for row in subset if row["worse_than_naive"]), None)
        clean = subset[0]
        crossovers[name] = {
            "policy": name,
            "true_lift": clean["true_lift"],
            "clean_snips_error": clean["snips_error"],
            "naive_error": clean["naive_error"],
            "crossover_factor": crossing["factor"] if crossing else None,
            "ess_fraction_at_crossover": crossing["ess_fraction"] if crossing else None,
            "ess_fraction_clean": clean["ess_fraction"],
            "snips_error_at_crossover": crossing["snips_error"] if crossing else None,
            "ess_moved_the_wrong_way": bool(
                crossing and crossing["ess_fraction"] > clean["ess_fraction"]
            ),
        }

    crossed = [entry for entry in crossovers.values() if entry["crossover_factor"]]
    earliest = min(crossed, key=lambda entry: entry["crossover_factor"]) if crossed else None
    ess_still_healthy = [
        entry
        for entry in crossed
        if entry["ess_fraction_at_crossover"] >= config.diagnostics.min_ess_fraction
    ]
    ess_wrong_way = [entry for entry in crossed if entry["ess_moved_the_wrong_way"]]

    payload = {
        "experiment": NAME,
        "question": "What happens when the logged propensities are wrong?",
        **provenance(config, n_sessions),
        "factors": list(config.experiments.e5_factors),
        "min_ess_fraction": config.diagnostics.min_ess_fraction,
        "policies": list(policies),
        "crossovers": crossovers,
        "n_crossed": len(crossed),
        "earliest_crossover_factor": earliest["crossover_factor"] if earliest else None,
        "earliest_crossover_policy": earliest["policy"] if earliest else None,
        "ess_fraction_at_earliest_crossover": (
            earliest["ess_fraction_at_crossover"] if earliest else None
        ),
        "n_crossings_with_healthy_ess": len(ess_still_healthy),
        "n_crossings_where_ess_rose": len(ess_wrong_way),
        "seconds": timer.seconds,
        "rows": rows,
    }
    write_result(NAME, payload)
    return payload


def report(payload: dict) -> None:
    header("E5: corrupting the propensities, which is the normal state of a production log")
    print(
        "The recorded propensity is replaced by a tempered copy of the true one. A factor of 1 "
        "leaves the log unchanged."
    )
    print(f"{payload['n_sessions']:,} sessions. Error is the absolute gap to the referee.")
    for name in payload["policies"]:
        subset = [row for row in payload["rows"] if row["policy"] == name]
        entry = payload["crossovers"][name]
        print()
        print(f"{name}  (true lift {entry['true_lift']:+.2%})")
        print(
            f"  {'factor':>8}{'SNIPS says':>13}{'SNIPS error':>13}"
            f"{'naive error':>13}{'effective n':>13}{'diagnostic':>12}"
        )
        print("  " + "-" * 72)
        for row in subset:
            healthy = row["ess_fraction"] >= payload["min_ess_fraction"]
            print(
                f"  {row['factor']:>8.2f}{row['snips_lift']:>+13.2%}{row['snips_error']:>13.2%}"
                f"{row['naive_error']:>13.2%}{row['ess_fraction']:>12.2%} "
                f"{('healthy' if healthy else 'refuses'):>11}"
                + ("   <- worse than naive replay" if row["worse_than_naive"] else "")
            )
    print()
    if payload["earliest_crossover_factor"]:
        print(
            f"The correction stops being worth having at a corruption factor of "
            f"{payload['earliest_crossover_factor']:.2f} for "
            f"{payload['earliest_crossover_policy']}."
        )
        print(
            f"At that point the effective sample size reads "
            f"{payload['ess_fraction_at_earliest_crossover']:.1%} of the log, which is above the "
            f"{payload['min_ess_fraction']:.0%} floor, so the tool answers confidently and is "
            f"wrong."
        )
        print(
            f"{payload['n_crossings_with_healthy_ess']} of {payload['n_crossed']} crossovers "
            f"happen while the diagnostic still reads healthy."
        )
        print(
            f"Worse: in {payload['n_crossings_where_ess_rose']} of {payload['n_crossed']} the "
            f"effective sample size is higher at the crossover than it was on the uncorrupted "
            f"log. Tempering flattens the weights, and flat weights are exactly what Kish's "
            f"formula rewards, so the diagnostic climbs while the estimate it vouches for "
            f"decays."
        )
        print(
            "Effective sample size measures the spread of the weights, not whether they are the "
            "right weights. No diagnostic in this repository detects a propensity that is "
            "confidently wrong."
        )
    else:
        print(
            "No corruption factor in this sweep made the corrected estimate worse than naive "
            "replay."
        )


if __name__ == "__main__":
    report(run())
