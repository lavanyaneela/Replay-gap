"""The scripts under tools/ and the experiment runner, which coverage of src/ cannot see.

An audit found that ``make experiments`` had been broken the whole time: the runner imports the
experiments package, nothing put the repository root on sys.path, and the target that
regenerates most of the published figures failed with ModuleNotFoundError. The suite was at a
hundred percent line coverage of ``src/`` and saw none of it, because nothing under ``src/``
imports the runner.

These tests exist so that the second time is caught by a machine. They run the real entry
points in a subprocess, because importing them in process would put the repository root on
sys.path as a side effect and hide exactly the failure being tested.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def run(*args: str, cwd: Path | None = None, env: dict | None = None):
    """Run a script the way a Makefile target does, with no PYTHONPATH help."""
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        check=False,
    )


def test_the_experiment_runner_can_import_the_experiments_it_runs():
    """The break an audit found. Nothing under src/ imports this, so coverage cannot see it.

    This uses --dry-run rather than --help, and the distinction is the whole test. An earlier
    version ran --help, which argparse answers and exits before the import loop is reached, so
    it passed against the broken runner: a green check certifying a property it never looked
    at. --dry-run imports every experiment and stops. The test below proves that it regresses.
    """
    result = run(str(TOOLS / "run_experiments.py"), "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "5 experiment(s) importable" in result.stdout


def test_the_regression_test_above_actually_fails_against_the_broken_runner(tmp_path):
    """A regression test that does not fail on the regression is worse than no test at all.

    This reconstructs the pre-fix runner by removing the one line that puts the repository root
    on sys.path, then asserts that --help still passes against it, which is why the first
    version of the test above was useless, and that --dry-run does not.
    """
    source = (TOOLS / "run_experiments.py").read_text(encoding="utf-8")
    marker = "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))"
    assert marker in source, "the import path fix is gone, so this test is checking nothing"
    broken = tmp_path / "run_experiments_without_the_fix.py"
    broken.write_text(source.replace(marker, "pass"), encoding="utf-8")

    helped = run(str(broken), "--help")
    assert helped.returncode == 0, "argparse answers --help before any import, as expected"
    assert "ModuleNotFoundError" not in helped.stderr

    dry = run(str(broken), "--dry-run")
    assert dry.returncode != 0, "--dry-run must reach the import and fail without the fix"
    assert "ModuleNotFoundError" in dry.stderr


def test_the_experiment_runner_lists_every_experiment_the_repository_publishes():
    result = run(str(TOOLS / "run_experiments.py"), "--dry-run")
    for name in (
        "e1_sign_agreement",
        "e2_overlap_curve",
        "e3_estimator_comparison",
        "e4_detectable_lift",
        "e5_propensity_misspecification",
    ):
        assert f"{name}: imported from" in result.stdout


def test_the_experiment_runner_rejects_an_experiment_that_does_not_exist():
    result = run(str(TOOLS / "run_experiments.py"), "--only", "e9_invented")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_the_number_checker_runs_from_the_repository_root():
    result = run(str(TOOLS / "check_numbers.py"), "--strict-reverse")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "every published figure traces to code that ran" in result.stdout


def test_the_number_checker_reports_a_missing_metrics_file_rather_than_tracing_back(tmp_path):
    result = run(str(TOOLS / "check_numbers.py"), "--metrics", str(tmp_path / "absent.json"))
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "is missing" in result.stderr


def test_the_metrics_collector_refuses_to_read_reports_that_are_not_there(tmp_path):
    result = run(
        str(TOOLS / "collect_metrics.py"),
        "--skip-tests",
        "--report-dir",
        str(tmp_path),
        "--out",
        str(tmp_path / "metrics.json"),
    )
    assert result.returncode != 0
    assert "Run 'make test' first" in result.stderr


def test_every_metric_the_specification_names_is_present_in_the_committed_payload():
    sys.path.insert(0, str(TOOLS))
    try:
        from metrics_spec import SPEC
    finally:
        sys.path.pop(0)
    payload = json.loads((ROOT / "docs" / "metrics.json").read_text(encoding="utf-8"))
    missing = sorted({metric.key for metric in SPEC.metrics} - set(payload["metrics"]))
    assert not missing, f"specified but never measured: {missing}"
    missing_rows = sorted({row.key for row in SPEC.rows} - set(payload["rows"]))
    assert not missing_rows, f"rows specified but never rendered: {missing_rows}"


def test_every_checked_document_the_payload_names_exists():
    payload = json.loads((ROOT / "docs" / "metrics.json").read_text(encoding="utf-8"))
    for name in payload["checked_documents"]:
        assert (ROOT / name).exists(), name


def test_no_exemption_in_the_checker_hides_a_specific_measured_value():
    """An exemption covering a measured figure would excuse it from the reverse sweep."""
    sys.path.insert(0, str(TOOLS))
    try:
        from check_numbers import assert_no_exemption_hides_a_metric
    finally:
        sys.path.pop(0)
    payload = json.loads((ROOT / "docs" / "metrics.json").read_text(encoding="utf-8"))
    assert assert_no_exemption_hides_a_metric(payload) == []


def test_the_committed_receipts_proof_shows_the_checker_failing_every_corruption():
    """The transcript is only worth committing if every case in it is a failure."""
    text = (ROOT / "docs" / "receipts-proof.md").read_text(encoding="utf-8")
    cases = [line for line in text.splitlines() if line.startswith("=== ")]
    exits = [line for line in text.splitlines() if line.startswith("exit ")]
    assert len(cases) == len(exits), text
    corruptions = [
        (case, code) for case, code in zip(cases, exits, strict=True) if "Restored" not in case
    ]
    corruptions = corruptions[1:]  # the first case is the uncorrupted baseline
    assert corruptions, text
    for case, code in corruptions:
        assert code == "exit 1", f"{case} is published as {code}, so the checker did not catch it"


@pytest.mark.parametrize(
    "script",
    ["run_experiments.py", "collect_metrics.py", "check_numbers.py", "metrics_spec.py"],
)
def test_every_tool_script_compiles(script):
    result = run("-m", "py_compile", str(TOOLS / script))
    assert result.returncode == 0, result.stderr
