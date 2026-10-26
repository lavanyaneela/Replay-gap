"""E2: how fast does a fifty thousand session log stop being able to answer?

The deployed ranker's log is not a fixed quantity of evidence. Its value depends entirely on
which question you ask of it, and this experiment measures the rate at which it evaporates.

A candidate policy is swept from the incumbent towards a target it genuinely improves on, and at
each step three things are recorded: how far the candidate has moved, what the log is worth in
effective sessions, and what naive replay would still have reported. The shape is the finding.
Naive replay keeps producing confident numbers all the way along the sweep, including at the end
where the effective sample size has fallen by two orders of magnitude, because the quantity
naive replay reports has no term in it for how much of the log it had to throw away.

The sweep runs twice, and the two curves answer different objections.

The first walks towards true relevance at a fixed serving temperature. This is the optimistic
case, a strictly better model served exactly as the old one was, and the log survives it: the
effective sample size falls by a single order of magnitude and never crosses the refusal floor.
The deployed ranker is already roughly right, so a candidate that is more right still overlaps
with it.

The second walks towards a model that actively disagrees with the deployed ranker, which is what
a team builds when it believes production has a bias worth correcting, and sharpens the serving
temperature on the way, which is what a team does when it stops hedging. Neither of those is a
decision anybody writes a design document about. Together they take the log below the floor and
the tool refuses.
"""

from __future__ import annotations

import numpy as np

from experiments.common import Timer, experiment_fixture, header, provenance, write_result
from replay_gap.config import load_config
from replay_gap.diagnostics.ess import effective_sample_size, ess_fraction
from replay_gap.estimators.inputs import build_estimator_inputs
from replay_gap.evaluation import trace_policy
from replay_gap.policies.base import ScorePolicy
from replay_gap.policies.baselines import (
    BlendPolicy,
    ContrarianPolicy,
    IncumbentPolicy,
    _standardise_scores,
)
from replay_gap.ranking import permutation_index
from replay_gap.simulator.ground_truth import true_value

NAME = "e2_overlap_curve"


def sweep(config, fixture, perms, baseline_true, logged_mean, temperatures, label, target=None):
    """Walk a candidate from the incumbent towards a target, recording what the log is worth.

    With no target the walk is towards true relevance, using the ordinary blend policy. With a
    target the walk is a convex mix of the incumbent's standardised scores and the target's, so
    a direction the blend policy cannot express (one that leans against the incumbent) can still
    be swept the same way.
    """
    rows = []
    steps = config.experiments.e2_divergence_steps
    log = fixture.log
    incumbent_scores = None
    target_scores = None
    if target is not None:
        incumbent_scores = _standardise_scores(
            fixture.logging_policy.scores(fixture.world, log.context_id, log.candidates)
        )
        target_scores = _standardise_scores(target.raw_scores(log.context_id, log.candidates))

    for index in range(steps + 1):
        share = index / steps
        temperature = temperatures[index]
        if target is None:
            policy = BlendPolicy(
                fixture.world,
                fixture.logging_policy,
                truth_weight=share,
                incumbent_weight=1.0 - share,
                noise_weight=0.0,
                noise_seed=1,
                temperature=temperature,
                name=f"{label}_{index:02d}",
            )
        else:
            mixed = (1.0 - share) * incumbent_scores + share * target_scores
            policy = ScorePolicy(
                f"{label}_{index:02d}",
                lambda _context, _candidates, scores=mixed: scores,
                temperature=temperature,
            )
        trace = trace_policy(policy, log, perms)
        inputs = build_estimator_inputs(log, trace, clip=config.estimators.clip_weight)
        estimators, baselines = inputs.ratio_estimators()
        matched = int(inputs.matched.sum())
        truth = true_value(fixture.world, log, trace.logits, perms) / baseline_true - 1.0
        rows.append(
            {
                "step": index,
                "truth_weight": share,
                "temperature": temperature,
                "true_lift": truth,
                "naive_lift": (
                    estimators["naive"].value() / logged_mean - 1.0 if matched else float("nan")
                ),
                "snips_lift": estimators["snips"].value() / baselines["snips"].value() - 1.0,
                "match_rate": float(inputs.matched.mean()),
                "effective_sample_size": effective_sample_size(inputs.weight),
                "ess_fraction": ess_fraction(inputs.weight),
                "max_weight_share": float(inputs.weight.max() / inputs.weight.sum()),
                "max_raw_weight": float(inputs.raw_weight.max()),
                "clipped_share": inputs.clipped_share,
            }
        )
    return rows


def run() -> dict:
    config = load_config()
    n_sessions = config.log.n_sessions
    fixture = experiment_fixture(config, n_sessions)
    log = fixture.log
    perms = permutation_index(log.n_candidates, log.n_positions)
    incumbent = IncumbentPolicy(fixture.world, fixture.logging_policy, config.policies.temperature)
    baseline_true = true_value(
        fixture.world, log, trace_policy(incumbent, log, perms).logits, perms
    )
    logged_mean = float(log.reward.mean())
    steps = config.experiments.e2_divergence_steps

    with Timer("sweep") as timer:
        fixed = sweep(
            config,
            fixture,
            perms,
            baseline_true,
            logged_mean,
            [config.policies.temperature] * (steps + 1),
            "model_only",
        )
        sharpened = sweep(
            config,
            fixture,
            perms,
            baseline_true,
            logged_mean,
            list(
                np.linspace(
                    config.policies.temperature, config.policies.sharp_temperature, steps + 1
                )
            ),
            "model_and_serving",
            target=ContrarianPolicy(fixture.world, fixture.logging_policy),
        )

    floor = config.diagnostics.min_ess_fraction
    first_refusal = next((row for row in fixed if row["ess_fraction"] < floor), None)
    first_refusal_sharp = next((row for row in sharpened if row["ess_fraction"] < floor), None)
    last = fixed[-1]
    last_sharp = sharpened[-1]

    payload = {
        "experiment": NAME,
        "question": (
            "How fast does the log stop being able to answer, as the candidate policy moves "
            "away from the one that produced it?"
        ),
        **provenance(config, n_sessions),
        "steps": steps,
        "min_ess_fraction": floor,
        "start_ess_fraction": fixed[0]["ess_fraction"],
        "end_ess_fraction": last["ess_fraction"],
        "ess_collapse_factor": fixed[0]["effective_sample_size"]
        / max(last["effective_sample_size"], 1e-12),
        "end_ess_fraction_sharpened": last_sharp["ess_fraction"],
        "sharpening_extra_collapse_factor": last["effective_sample_size"]
        / max(last_sharp["effective_sample_size"], 1e-12),
        "first_refusal_step": first_refusal["step"] if first_refusal else None,
        "first_refusal_step_sharpened": (
            first_refusal_sharp["step"] if first_refusal_sharp else None
        ),
        "naive_lift_at_end": last["naive_lift"],
        "true_lift_at_end": last["true_lift"],
        "max_raw_weight_at_end": last["max_raw_weight"],
        "max_raw_weight_at_end_sharpened": last_sharp["max_raw_weight"],
        "seconds": timer.seconds,
        "model_only": fixed,
        "model_and_serving": sharpened,
    }
    write_result(NAME, payload)
    return payload


def report(payload: dict) -> None:
    header("E2: what the log is worth, as the candidate moves away from the deployed ranker")
    print(
        f"One log of {payload['n_sessions']:,} sessions. The candidate is swept from the "
        f"incumbent towards true relevance."
    )
    print()
    print(
        f"{'step':>5}{'toward truth':>14}{'true lift':>12}{'naive says':>12}"
        f"{'SNIPS says':>12}{'effective n':>14}{'share of log':>14}{'largest weight':>16}"
    )
    print("-" * 99)
    for row in payload["model_only"]:
        print(
            f"{row['step']:>5}{row['truth_weight']:>13.0%} {row['true_lift']:>+11.2%}"
            f"{row['naive_lift']:>+12.2%}{row['snips_lift']:>+12.2%}"
            f"{row['effective_sample_size']:>14,.0f}{row['ess_fraction']:>13.2%} "
            f"{row['max_raw_weight']:>15,.0f}"
        )
    print()
    print(
        f"The log falls from {payload['start_ess_fraction']:.1%} to "
        f"{payload['end_ess_fraction']:.2%} of its nominal size, a factor of "
        f"{payload['ess_collapse_factor']:.0f}."
    )
    if payload["first_refusal_step"] is not None:
        print(
            f"The tool refuses from step {payload['first_refusal_step']} of "
            f"{payload['steps']} onwards."
        )
    else:
        print(
            f"No step in this sweep falls below the {payload['min_ess_fraction']:.0%} floor "
            f"on the model axis alone."
        )
    print(
        f"Naive replay still reports {payload['naive_lift_at_end']:+.2%} at the end of the "
        f"sweep, where the truth is {payload['true_lift_at_end']:+.2%}."
    )
    print()
    print(
        f"{'step':>5}{'toward target':>15}{'temperature':>13}{'true lift':>12}{'naive says':>12}"
        f"{'effective n':>14}{'share of log':>14}{'largest weight':>16}"
    )
    print("-" * 101)
    for row in payload["model_and_serving"]:
        print(
            f"{row['step']:>5}{row['truth_weight']:>14.0%}{row['temperature']:>13.2f}"
            f"{row['true_lift']:>+12.2%}{row['naive_lift']:>+12.2%}"
            f"{row['effective_sample_size']:>14,.0f}{row['ess_fraction']:>13.2%} "
            f"{row['max_raw_weight']:>15,.0f}"
        )
    print()
    print(
        f"A model that disagrees with production, served greedily, ends at "
        f"{payload['end_ess_fraction_sharpened']:.2%} of the log, a further factor of "
        f"{payload['sharpening_extra_collapse_factor']:.1f} below the optimistic sweep."
    )
    if payload["first_refusal_step_sharpened"] is not None:
        print(
            f"The tool refuses from step {payload['first_refusal_step_sharpened']} of "
            f"{payload['steps']} onwards on this curve, while naive replay keeps answering at "
            f"every step."
        )


if __name__ == "__main__":
    report(run())
