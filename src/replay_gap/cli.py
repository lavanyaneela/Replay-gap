"""The command line, and the four exit codes that are the whole point of it.

Most offline evaluation tools have two outcomes: a number, or a crash. This one has four, and
the distinction between 1 and 2 is the reason the repository exists.

    0  Evaluated. The corrected lift is positive and the interval excludes zero.
    1  Evaluated. The lift is not positive. The tool worked; the answer is no.
    2  Refused. Overlap between the candidate policy and the log is too poor for any offline
       estimate to be honest. The log cannot answer this question and no amount of correction
       changes that. Run an experiment instead.
    3  Tool error. Something in the tool failed, which is not the same as the log being unable
       to answer, and a caller must be able to tell them apart.

Code 2 is the heart of it. An evaluation tool with no way to decline will always produce a
number, and a number is what gets shipped on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from replay_gap.config import Config, load_config
from replay_gap.errors import ReplayGapError
from replay_gap.fixtures import (
    FIXTURE_DIR,
    build_fixture,
    load_or_build,
    read_manifest,
    write_fixture,
)
from replay_gap.pipeline import (
    EXIT_NOT_POSITIVE,
    EXIT_POSITIVE,
    EXIT_TOOL_ERROR,
    EvaluationRun,
    build_policies,
    compare_exit_code,
    run_evaluation,
)
from replay_gap.report import render_gap_table, render_html, render_json, render_policy_detail
from replay_gap.report import render_table as render_summary_table
from replay_gap.store import DuckDBStore
from replay_gap.tracking import LocalTracking

DEFAULT_REPORT_DIR = Path("reports")


def build_parser() -> argparse.ArgumentParser:
    """The command surface. Every threshold comes from the configuration, not from a flag."""
    parser = argparse.ArgumentParser(
        prog="replay-gap",
        description=(
            "Offline ranker evaluation that reports when it cannot answer. Exit codes: "
            "0 positive lift, 1 no positive lift, 2 refused for lack of overlap, 3 tool error."
        ),
    )
    parser.add_argument("--config", default=None, help="path to a YAML configuration file")
    parser.add_argument(
        "--fixture-dir", default=str(FIXTURE_DIR), help="directory holding the committed log"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="regenerate the committed fixture from its seed")
    generate.add_argument(
        "--force", action="store_true", help="overwrite an existing fixture in place"
    )
    generate.add_argument(
        "--sessions", type=int, default=None, help="override the configured session count"
    )

    evaluate = sub.add_parser("evaluate", help="evaluate one candidate policy and exit")
    evaluate.add_argument("policy", help="name of the policy to evaluate")
    evaluate.add_argument(
        "--no-ranker", action="store_true", help="skip training the LambdaMART candidate"
    )
    evaluate.add_argument("--track", action="store_true", help="record the run in local MLflow")

    compare = sub.add_parser("compare", help="evaluate every policy and print the table")
    compare.add_argument(
        "--no-ranker", action="store_true", help="skip training the LambdaMART candidate"
    )
    compare.add_argument("--gap", action="store_true", help="print estimator error, not lift")
    compare.add_argument("--track", action="store_true", help="record the runs in local MLflow")
    compare.add_argument(
        "--report-dir", default=str(DEFAULT_REPORT_DIR), help="where to write JSON and HTML"
    )

    sub.add_parser("diagnose", help="print the log's own diagnostics, with no candidate policy")
    sub.add_parser("policies", help="list the candidate policies this build knows about")
    return parser


def _load(args: argparse.Namespace) -> tuple[Config, object]:
    config = load_config(args.config)
    return config, load_or_build(config, args.fixture_dir)


def command_generate(args: argparse.Namespace) -> int:
    """Rebuild the committed fixture and print its manifest."""
    config = load_config(args.config)
    directory = Path(args.fixture_dir)
    if (directory / "manifest.json").exists() and not args.force:
        print(f"fixture already present at {directory}; pass --force to regenerate", flush=True)
        return EXIT_NOT_POSITIVE
    fixture = build_fixture(config, n_sessions=args.sessions)
    manifest = write_fixture(fixture, config, directory)
    print(json.dumps(manifest, indent=2))
    return EXIT_POSITIVE


def command_policies(args: argparse.Namespace) -> int:
    """List what can be evaluated, without training anything."""
    config, fixture = _load(args)
    for name in build_policies(config, fixture, with_ranker=False):
        print(name)
    print("lightgbm_ranker")
    return EXIT_POSITIVE


def command_diagnose(args: argparse.Namespace) -> int:
    """What the log looks like before any candidate policy is involved."""
    config, fixture = _load(args)
    manifest = (
        read_manifest(args.fixture_dir)
        if (Path(args.fixture_dir) / "manifest.json").exists()
        else {"log_content_hash": "not committed"}
    )
    log = fixture.log
    print(f"sessions                {log.n_sessions:,}")
    print(f"candidates per session  {log.n_candidates}")
    print(f"slate positions         {log.n_positions}")
    print(f"mean reward             {log.reward.mean():.5f}")
    print(f"content hash            {manifest['log_content_hash'][:16]}")
    print()
    with DuckDBStore() as store:
        store.load(log)
        counts = store.row_counts()
        print("rows per table          " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
        mismatched = len(store.run("002_session_rewards"))
        print(f"reward integrity        {mismatched} session(s) disagree with their impressions")
        print()
        print("click through rate by position")
        print(store.run("003_position_ctr").to_string(index=False))
        print()
        print("mean reward by logging propensity decile")
        print(store.run("004_propensity_deciles").to_string(index=False))
    return EXIT_POSITIVE


def _track(run: EvaluationRun, enabled: bool) -> None:
    """Record every estimator run and register every policy, if tracking was asked for."""
    if not enabled:
        return
    tracking = LocalTracking()
    for result in run.results:
        tracking.register_policy(
            result.policy,
            description=f"candidate policy evaluated against a {result.n_sessions} session log",
            tags={"decision": result.decision, "true_lift": f"{result.true_lift:.6f}"},
        )
        for estimator, lift in result.lifts.items():
            tracking.log_estimate(
                policy=result.policy,
                estimator=estimator,
                params={"n_sessions": result.n_sessions},
                metrics={
                    "lift": lift,
                    "true_lift": result.true_lift,
                    "error": lift - result.true_lift,
                    "ess_fraction": result.overlap.ess_fraction if result.overlap else 0.0,
                },
            )


def command_evaluate(args: argparse.Namespace) -> int:
    """Evaluate one policy and return the exit code that describes the answer."""
    config, fixture = _load(args)
    policies = build_policies(config, fixture, with_ranker=not args.no_ranker)
    if args.policy not in policies:
        known = ", ".join(policies)
        print(f"unknown policy {args.policy}; available: {known}", file=sys.stderr)
        return EXIT_TOOL_ERROR
    run = run_evaluation(config, fixture, {args.policy: policies[args.policy]})
    result = run.by_name(args.policy)
    print(render_policy_detail(result))
    _track(run, args.track)
    return result.exit_code


def command_compare(args: argparse.Namespace) -> int:
    """Evaluate everything and write the report. The exit code follows the best policy."""
    config, fixture = _load(args)
    policies = build_policies(config, fixture, with_ranker=not args.no_ranker)
    run = run_evaluation(config, fixture, policies)
    print(render_gap_table(run) if args.gap else render_summary_table(run))
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation.json").write_text(render_json(run), encoding="utf-8")
    (report_dir / "evaluation.html").write_text(render_html(run), encoding="utf-8")
    print()
    print(f"wrote {report_dir / 'evaluation.json'} and {report_dir / 'evaluation.html'}")
    _track(run, args.track)
    refused = [r.policy for r in run.results if r.decision == "refused"]
    if refused:
        print(f"refused for {len(refused)} policy(ies): {', '.join(refused)}")
    return compare_exit_code(run.results)


COMMANDS = {
    "generate": command_generate,
    "evaluate": command_evaluate,
    "compare": command_compare,
    "diagnose": command_diagnose,
    "policies": command_policies,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Every expected failure becomes an exit code, never a traceback."""
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except ReplayGapError as error:
        print(f"replay-gap: {error}", file=sys.stderr)
        remedy = getattr(error, "remedy", None)
        if remedy:
            print(f"remedy: {remedy}", file=sys.stderr)
        return EXIT_TOOL_ERROR
    except BrokenPipeError:  # pragma: no cover - only reachable under a closed stdout
        return EXIT_TOOL_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
