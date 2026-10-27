"""Re-measure every published figure and write docs/metrics.json. Nothing is typed by hand.

The test count and the coverage percentage are read from pytest's own machine readable outputs,
the junit XML and the coverage JSON, rather than parsed out of a progress line. A parsed progress
line quietly becomes whatever the last run happened to print, which is the failure mode this
whole mechanism exists to prevent.

The suite runs as its own CI step and this script is then called with --skip-tests, so a red test
surfaces as a red test rather than as a CalledProcessError traceback from inside a metrics
script. Diagnosing that the second way costs an hour.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_spec import SPEC  # noqa: E402

REPORT_DIR = Path("reports")
EXPERIMENT_DIR = Path("docs/experiments")
METRICS_PATH = Path("docs/metrics.json")


def run_suite(report_dir: Path) -> None:
    """Run pytest with the machine readable reports the metrics are read from."""
    report_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--junitxml={report_dir / 'junit.xml'}",
            "--cov=replay_gap",
            f"--cov-report=json:{report_dir / 'coverage.json'}",
            f"--cov-report=xml:{report_dir / 'coverage.xml'}",
        ],
        check=True,
    )


def read_suite_metrics(report_dir: Path) -> dict:
    """Test count and line coverage, from the reports rather than from a progress line."""
    junit = report_dir / "junit.xml"
    coverage = report_dir / "coverage.json"
    for required in (junit, coverage):
        if not required.exists():
            raise SystemExit(f"{required} is missing. Run 'make test' first, or drop --skip-tests.")
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    total = int(suite.get("tests", 0))
    skipped = int(suite.get("skipped", 0))
    failures = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
    if failures:
        raise SystemExit(f"{failures} test(s) failed; metrics are not collected from a red suite")
    payload = json.loads(coverage.read_text(encoding="utf-8"))
    return {
        # Both numbers, because the difference between them is the PostgreSQL parity module,
        # which is skipped wherever no server is reachable. Publishing only the total would
        # claim a test ran that did not.
        "tests_total": total,
        "tests_skipped": skipped,
        "tests_executed": total - skipped,
        "tests_failed": failures,
        "coverage_line_pct": float(payload["totals"]["percent_covered"]) / 100.0,
    }


def read_evaluation_metrics() -> dict:
    """The default comparison, re-run so the figures come from this build and not from a cache."""
    from replay_gap.config import load_config
    from replay_gap.fixtures import load_or_build
    from replay_gap.pipeline import build_policies, run_evaluation

    config = load_config()
    fixture = load_or_build(config)
    run = run_evaluation(config, fixture, build_policies(config, fixture))
    payload = {
        "n_sessions": run.n_sessions,
        "n_candidates": run.n_candidates,
        "n_positions": run.n_positions,
        "logged_mean_reward": run.logged_mean_reward,
    }
    for result in run.results:
        payload[result.policy] = result.as_dict()

    # The war story's number, re-measured rather than remembered. The first version of this
    # repository compared a replay scored on first slot clicks against the log's mean session
    # reward, which are different units, and reported this for a policy that is the deployed
    # ranker itself. Recomputing it here keeps the README's account of the bug honest.
    incumbent = run.by_name("incumbent")
    first_slot_value = (1.0 + incumbent.lifts["naive_first_slot"]) * float(
        fixture.log.clicks[:, 0].mean()
    )
    payload["units_error_lift"] = first_slot_value / run.logged_mean_reward - 1.0
    return payload


def read_experiment(name: str) -> dict:
    path = EXPERIMENT_DIR / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run 'make experiments' first.")
    return json.loads(path.read_text(encoding="utf-8"))


def dig(payload: dict, path: tuple[str, ...], where: str):
    """Walk a path through the payload, naming the whole path when a step is missing.

    A numeric step indexes a list, because several published figures live in a sweep that is
    naturally a list of steps rather than a mapping.
    """
    cursor = payload
    for key in path:
        if isinstance(cursor, list | tuple) and key.lstrip("-").isdigit():
            index = int(key)
            if not -len(cursor) <= index < len(cursor):
                raise SystemExit(f"{where} has no element {index} at {'.'.join(path)}")
            cursor = cursor[index]
            continue
        if not isinstance(cursor, dict) or key not in cursor:
            raise SystemExit(f"{where} has no value at {'.'.join(path)}")
        cursor = cursor[key]
    if cursor is None:
        raise SystemExit(f"{where} holds null at {'.'.join(path)}")
    return cursor


def collect(skip_tests: bool, report_dir: Path) -> dict:
    """Gather every metric the specification names, from the sources it names."""
    if not skip_tests:
        run_suite(report_dir)
    sources: dict[str, dict] = {
        "suite": read_suite_metrics(report_dir),
        "evaluation": read_evaluation_metrics(),
    }
    for metric in SPEC.metrics:
        if metric.source not in sources:
            sources[metric.source] = read_experiment(metric.source)

    metrics: dict[str, float] = {}
    formatted: dict[str, str] = {}
    anchors: dict[str, list[str]] = {}
    descriptions: dict[str, str] = {}
    for metric in SPEC.metrics:
        value = dig(sources[metric.source], metric.path, metric.source)
        metrics[metric.key] = value
        formatted[metric.key] = metric.format(value)
        anchors[metric.key] = [phrase.format(metric.format(value)) for phrase in metric.anchors]
        descriptions[metric.key] = metric.description

    rows: dict[str, str] = {}
    for row in SPEC.rows:
        if row.source not in sources:
            sources[row.source] = read_experiment(row.source)
        rows[row.key] = row.render(sources[row.source], dig)
        descriptions[row.key] = row.description

    return {
        "metrics": metrics,
        "formatted": formatted,
        "anchors": anchors,
        "rows": rows,
        "descriptions": descriptions,
        "checked_documents": list(SPEC.documents),
        "note": (
            "Every value here was produced by running the test suite, the evaluation and the "
            "experiments. Nothing in this file is typed by hand, and tools/check_numbers.py "
            "fails the build when a document quotes a figure this file no longer holds."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-measure every published figure.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="read an existing junit and coverage report instead of running pytest",
    )
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--out", default=str(METRICS_PATH))
    args = parser.parse_args(argv)

    payload = collect(args.skip_tests, Path(args.report_dir))
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {target} with {len(payload['metrics'])} measured figures and "
        f"{len(payload['rows'])} guarded table rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
