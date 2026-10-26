"""Rendering, including the check that stops a verdict badge from rendering invisibly.

Two earlier projects shipped a report where a badge class and a table cell class collided at
equal CSS specificity, the later rule won, and the headline verdict rendered in one colour on
the same colour. The Python was right, the HTML was right, and the tests passed. The test here
is the cheap half of the fix: every class the markup uses has a rule in the stylesheet. The
screenshot tool reads the computed style back, which is the half only a browser can do.
"""

from __future__ import annotations

import re

import pytest

from replay_gap.pipeline import build_policies, run_evaluation
from replay_gap.report import (
    COLUMN_LABELS,
    DECISION_CLASS,
    DECISION_LABELS,
    STYLESHEET,
    estimator_columns,
    format_percent,
    render_gap_table,
    render_html,
    render_json,
    render_policy_detail,
    render_table,
)


@pytest.fixture(scope="module")
def small_run(small_config, small_fixture):
    policies = build_policies(small_config, small_fixture, with_ranker=False)
    return run_evaluation(small_config, small_fixture, policies)


def test_a_percentage_carries_its_sign():
    assert format_percent(0.1234) == "+12.34%"
    assert format_percent(-0.05) == "-5.00%"


def test_a_percentage_honours_the_requested_precision():
    assert format_percent(0.123456, digits=1) == "+12.3%"


def test_a_missing_estimate_renders_as_text_rather_than_as_nan():
    assert format_percent(float("nan")) == "n/a"


def test_the_estimator_columns_follow_the_documented_reading_order(small_run):
    columns = estimator_columns(small_run)
    assert columns[0] == "naive"
    assert columns[-1] == "dr"
    assert all(name in COLUMN_LABELS for name in columns)


def test_the_table_opens_exactly_one_row_per_policy(small_run):
    rows = [line.split()[0] for line in render_table(small_run).splitlines()[2:]]
    assert rows == [result.policy for result in small_run.results]


def test_the_table_puts_the_referee_before_any_estimator(small_run):
    header = render_table(small_run).splitlines()[0]
    assert header.index("true lift") < header.index("naive replay")


def test_the_table_shows_a_verdict_for_every_row(small_run):
    text = render_table(small_run)
    assert sum(text.count(label) for label in DECISION_LABELS.values()) >= len(small_run.results)


def test_the_gap_table_reports_error_against_the_referee(small_run):
    text = render_gap_table(small_run)
    assert "error" in text.splitlines()[0]
    assert len(text.splitlines()) == len(small_run.results) + 2


def test_the_gap_table_prints_a_placeholder_for_an_estimator_a_policy_lacks(small_run):
    from dataclasses import replace as dataclass_replace

    first, *rest = small_run.results
    without_dr = dataclass_replace(first, lifts={k: v for k, v in first.lifts.items() if k != "dr"})
    trimmed = dataclass_replace(small_run, results=[without_dr, *rest])
    assert "doubly robust error" in render_gap_table(trimmed)
    assert "n/a" in render_gap_table(trimmed)


def test_the_policy_detail_block_names_the_matched_session_count(small_run):
    text = render_policy_detail(small_run.by_name("oracle"))
    assert "matched sessions" in text
    assert "effective sample size" in text
    assert "exit code" in text


def test_the_policy_detail_block_reports_the_referee_next_to_the_estimates(small_run):
    text = render_policy_detail(small_run.by_name("incumbent"))
    assert "naive replay lift" in text
    assert "true lift (referee)" in text


def test_the_json_payload_parses_and_carries_every_policy(small_run):
    import json

    payload = json.loads(render_json(small_run))
    assert len(payload["results"]) == len(small_run.results)
    assert payload["n_sessions"] == small_run.n_sessions


def test_the_html_report_is_self_contained(small_run):
    """No external stylesheet and no scripts, so a headless capture sees what a browser sees."""
    html = render_html(small_run)
    assert "<script" not in html
    assert 'rel="stylesheet"' not in html
    assert "<style>" in html


def test_every_css_class_the_markup_uses_has_a_rule_in_the_stylesheet(small_run):
    html = render_html(small_run)
    body = html.split("</style>", 1)[1]
    used = {name for group in re.findall(r'class="([^"]+)"', body) for name in group.split()}
    declared = set(re.findall(r"\.([a-zA-Z][\w-]*)", STYLESHEET))
    assert used <= declared, f"classes with no rule: {sorted(used - declared)}"


def test_the_verdict_badge_rule_is_more_specific_than_the_table_cell_rule():
    """The collision that made a headline verdict invisible. Pinned by specificity."""
    for css_class in DECISION_CLASS.values():
        assert f"td .verdict.{css_class}" in STYLESHEET


def test_every_decision_the_pipeline_can_return_has_a_label_and_a_class():
    assert set(DECISION_LABELS) == set(DECISION_CLASS)
    assert set(DECISION_LABELS) == {"positive", "not_positive", "refused"}


def test_the_html_report_names_every_policy(small_run):
    html = render_html(small_run)
    for result in small_run.results:
        assert result.policy in html


def test_the_html_report_leads_with_the_incumbent_replaying_against_itself(small_run):
    lede = render_html(small_run).split('class="lede"', 1)[1].split("</div>", 1)[0]
    assert "over itself" in lede
    assert "because it is itself" in lede


def test_the_html_report_marks_an_overclaim_in_the_overclaim_class(small_run):
    assert 'class="overclaim"' in render_html(small_run)


def test_the_html_report_states_how_many_slates_the_referee_enumerated(small_run):
    expected = small_run.n_candidates * (small_run.n_candidates - 1) * (small_run.n_candidates - 2)
    assert str(expected) in render_html(small_run)


def test_the_html_title_is_carried_into_the_document(small_run):
    assert "<title>custom title</title>" in render_html(small_run, title="custom title")
