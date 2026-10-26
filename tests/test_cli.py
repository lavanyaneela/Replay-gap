"""The command line, which is where the four exit codes become the product.

Every test here asserts an exit code, not a string. A tool whose output changes is annoying; a
tool whose exit code changes breaks whatever was calling it.
"""

from __future__ import annotations

import json

import pytest

from replay_gap.cli import build_parser, main
from replay_gap.fixtures import build_fixture, write_fixture
from replay_gap.pipeline import (
    EXIT_NOT_POSITIVE,
    EXIT_POSITIVE,
    EXIT_REFUSED,
    EXIT_TOOL_ERROR,
)


@pytest.fixture(scope="module")
def cli_workspace(tmp_path_factory, small_config):
    """A written out small fixture plus the configuration that produced it.

    The overlap floor is raised to five percent here. The floor is a configuration value rather
    than a constant precisely so it can be set against the log in front of it, and in a twelve
    hundred session log a greedily served policy keeps about four percent of the sample where in
    the fifty thousand session log it keeps under two. The refusal the tests exercise is the
    same refusal either way.
    """
    import yaml

    root = tmp_path_factory.mktemp("cli")
    fixture_dir = root / "fixtures"
    write_fixture(build_fixture(small_config), small_config, fixture_dir)
    payload = {
        "seed": small_config.seed,
        "world": dict(small_config.world.__dict__),
        "slate": {
            "n_candidates": small_config.slate.n_candidates,
            "n_positions": small_config.slate.n_positions,
            "examination": list(small_config.slate.examination),
        },
        "logging_policy": dict(small_config.logging_policy.__dict__),
        "log": dict(small_config.log.__dict__),
        "policies": dict(small_config.policies.__dict__),
        "estimators": dict(small_config.estimators.__dict__),
        "diagnostics": dict(small_config.diagnostics.__dict__) | {"min_ess_fraction": 0.05},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return root, str(config_path), str(fixture_dir)


def run_cli(workspace, *args) -> int:
    _, config_path, fixture_dir = workspace
    return main(["--config", config_path, "--fixture-dir", fixture_dir, *args])


def test_the_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_the_parser_documents_the_exit_codes_in_its_description():
    description = build_parser().description
    for fragment in ("0 positive", "1 no positive", "2 refused", "3 tool error"):
        assert fragment in description


def test_listing_the_policies_succeeds_and_includes_the_ranker(cli_workspace, capsys):
    assert run_cli(cli_workspace, "policies") == EXIT_POSITIVE
    printed = capsys.readouterr().out.split()
    assert "incumbent" in printed
    assert "lightgbm_ranker" in printed


def test_diagnosing_the_log_prints_its_shape_and_its_integrity_check(cli_workspace, capsys):
    assert run_cli(cli_workspace, "diagnose") == EXIT_POSITIVE
    out = capsys.readouterr().out
    assert "content hash" in out
    assert "0 session(s) disagree" in out
    assert "click through rate by position" in out


def test_evaluating_the_incumbent_answers_no_rather_than_refusing(cli_workspace, capsys):
    """True lift is zero by construction, so the right answer is 1, not 0 and not 2."""
    assert run_cli(cli_workspace, "evaluate", "incumbent", "--no-ranker") == EXIT_NOT_POSITIVE
    assert "true lift (referee)" in capsys.readouterr().out


def test_evaluating_a_genuinely_better_policy_returns_the_positive_code(cli_workspace):
    assert run_cli(cli_workspace, "evaluate", "oracle", "--no-ranker") == EXIT_POSITIVE


def test_evaluating_a_sharply_served_policy_refuses(cli_workspace, capsys):
    """The same ranker, served greedily. Nothing about the model changed."""
    assert run_cli(cli_workspace, "evaluate", "contrarian_sharp", "--no-ranker") == EXIT_REFUSED
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "online experiment" in out


def test_evaluating_an_unknown_policy_is_a_tool_error_not_a_negative_result(cli_workspace, capsys):
    assert run_cli(cli_workspace, "evaluate", "nonsense", "--no-ranker") == EXIT_TOOL_ERROR
    assert "unknown policy" in capsys.readouterr().err


def test_comparing_every_policy_writes_both_report_formats(cli_workspace, tmp_path, capsys):
    report_dir = tmp_path / "reports"
    code = run_cli(cli_workspace, "compare", "--no-ranker", "--report-dir", str(report_dir))
    assert code == EXIT_REFUSED
    capsys.readouterr()
    payload = json.loads((report_dir / "evaluation.json").read_text())
    assert payload["results"]
    assert (report_dir / "evaluation.html").read_text().startswith("<!doctype html>")


def test_comparing_reports_the_refusal_rather_than_hiding_it(cli_workspace, tmp_path, capsys):
    report_dir = tmp_path / "reports-refusal"
    run_cli(cli_workspace, "compare", "--no-ranker", "--report-dir", str(report_dir))
    assert "refused for 1 policy" in capsys.readouterr().out


def test_the_gap_view_prints_error_against_the_referee(cli_workspace, tmp_path, capsys):
    report_dir = tmp_path / "reports-gap"
    run_cli(cli_workspace, "compare", "--no-ranker", "--gap", "--report-dir", str(report_dir))
    assert "error" in capsys.readouterr().out.splitlines()[0]


@pytest.mark.slow
def test_tracking_records_a_run_and_registers_the_policy(cli_workspace, tmp_path, monkeypatch):
    """The --track path writes to a local SQLite store, so this exercises it for real."""
    monkeypatch.chdir(tmp_path)
    assert run_cli(cli_workspace, "evaluate", "oracle", "--no-ranker", "--track") == EXIT_POSITIVE
    from replay_gap.tracking import LocalTracking

    summary = LocalTracking().summary()
    assert summary["runs"] >= 1
    assert summary["registered_models"] == 1


def test_generating_refuses_to_overwrite_an_existing_fixture_without_force(cli_workspace, capsys):
    assert run_cli(cli_workspace, "generate") == EXIT_NOT_POSITIVE
    assert "pass --force" in capsys.readouterr().out


def test_generating_with_force_rewrites_the_fixture_and_prints_its_manifest(cli_workspace, capsys):
    assert run_cli(cli_workspace, "generate", "--force") == EXIT_POSITIVE
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["log_content_hash"]


def test_generating_into_an_empty_directory_succeeds(tmp_path, cli_workspace):
    _, config_path, _ = cli_workspace
    code = main(
        [
            "--config",
            config_path,
            "--fixture-dir",
            str(tmp_path / "fresh"),
            "generate",
            "--sessions",
            "120",
        ]
    )
    assert code == EXIT_POSITIVE


def test_a_configuration_error_becomes_a_tool_error_rather_than_a_traceback(tmp_path, capsys):
    code = main(["--config", str(tmp_path / "absent.yaml"), "policies"])
    assert code == EXIT_TOOL_ERROR
    assert "replay-gap:" in capsys.readouterr().err


def test_a_refusal_raised_by_the_library_prints_its_remedy(monkeypatch, capsys):
    """RefusalError carries a remedy, and the CLI must print it rather than swallow it."""
    from replay_gap.errors import RefusalError

    def refuse(_args):
        raise RefusalError("overlap is too poor", remedy="run an online experiment")

    monkeypatch.setitem(__import__("replay_gap.cli", fromlist=["x"]).COMMANDS, "policies", refuse)
    assert main(["policies"]) == EXIT_TOOL_ERROR
    err = capsys.readouterr().err
    assert "overlap is too poor" in err
    assert "remedy: run an online experiment" in err


def test_a_corrupt_fixture_becomes_a_tool_error_with_the_reason_printed(
    cli_workspace, capsys, tmp_path
):
    from pathlib import Path

    import pandas as pd

    _, config_path, fixture_dir = cli_workspace
    broken = tmp_path / "broken"
    broken.mkdir()
    for name in ("manifest.json", "world.npz", "log.parquet"):
        (broken / name).write_bytes((Path(fixture_dir) / name).read_bytes())
    frame = pd.read_parquet(broken / "log.parquet")
    frame.loc[0, "click_0"] = 1 - int(frame.loc[0, "click_0"])
    frame.to_parquet(broken / "log.parquet", index=False)
    code = main(["--config", config_path, "--fixture-dir", str(broken), "diagnose"])
    assert code == EXIT_TOOL_ERROR
    assert "manifest hash" in capsys.readouterr().err
