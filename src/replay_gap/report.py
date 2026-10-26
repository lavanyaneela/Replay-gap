"""Rendering: the same evaluation as a terminal table, as JSON, and as an HTML page.

The column order is an argument rather than a layout choice. The referee comes first, naive
replay second, and the corrected estimators after that, so the eye crosses the gap between what
replay said and what was true before it reaches anything else. A table that led with SNIPS would
be a table about estimator quality, and this repository is about the gap.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass

from replay_gap.pipeline import EvaluationRun, PolicyResult

COLUMN_ORDER: tuple[str, ...] = ("naive", "naive_first_slot", "ips", "snips", "dr")

COLUMN_LABELS: dict[str, str] = {
    "naive": "naive replay",
    "naive_first_slot": "naive, first slot",
    "ips": "IPS",
    "snips": "SNIPS",
    "dr": "doubly robust",
}

DECISION_LABELS: dict[str, str] = {
    "positive": "SHIP",
    "not_positive": "NO",
    "refused": "REFUSED",
}

DECISION_CLASS: dict[str, str] = {
    "positive": "verdict-positive",
    "not_positive": "verdict-negative",
    "refused": "verdict-refused",
}


@dataclass(frozen=True)
class Column:
    """One column of the terminal table: a header, a width, and how to format a cell."""

    header: str
    width: int
    align: str = ">"


def format_percent(value: float, digits: int = 2) -> str:
    """A signed percentage, or a dash when the estimator did not produce one."""
    if value != value:  # NaN
        return "n/a"
    return f"{value * 100:+.{digits}f}%"


def estimator_columns(run: EvaluationRun) -> list[str]:
    """Estimator names present in this run, in the documented reading order."""
    present = {name for result in run.results for name in result.lifts}
    return [name for name in COLUMN_ORDER if name in present]


def render_table(run: EvaluationRun) -> str:
    """The terminal table. Fixed width, because a reviewer reads it in a screenshot."""
    columns = estimator_columns(run)
    header = (
        f"{'policy':<18}{'true lift':>12}"
        + "".join(f"{COLUMN_LABELS[name]:>19}" for name in columns)
        + f"{'ESS':>10}{'ESS %':>8}{'verdict':>10}"
    )
    lines = [header, "-" * len(header)]
    for result in run.results:
        overlap = result.overlap
        row = f"{result.policy:<18}{format_percent(result.true_lift):>12}"
        for name in columns:
            row += f"{format_percent(result.lifts.get(name, float('nan'))):>19}"
        row += f"{overlap.effective_sample_size:>10,.0f}" if overlap else f"{'n/a':>10}"
        row += f"{overlap.ess_fraction * 100:>7.2f}%" if overlap else f"{'n/a':>8}"
        row += f"{DECISION_LABELS.get(result.decision, '?'):>10}"
        lines.append(row)
    return "\n".join(lines)


def render_gap_table(run: EvaluationRun) -> str:
    """The table the README leads with: how far each estimator was from the truth."""
    columns = estimator_columns(run)
    header = f"{'policy':<18}{'true lift':>12}" + "".join(
        f"{COLUMN_LABELS[name] + ' error':>25}" for name in columns
    )
    lines = [header, "-" * len(header)]
    for result in run.results:
        row = f"{result.policy:<18}{format_percent(result.true_lift):>12}"
        for name in columns:
            if name not in result.lifts:
                row += f"{'n/a':>25}"
                continue
            row += f"{format_percent(result.error(name)):>25}"
        lines.append(row)
    return "\n".join(lines)


def render_policy_detail(result: PolicyResult) -> str:
    """Everything the CLI prints for the single policy it was asked about."""
    overlap = result.overlap
    low, high = result.intervals.get("snips", (float("nan"), float("nan")))
    lines = [
        f"policy                  {result.policy}",
        f"sessions in the log     {result.n_sessions:,}",
        f"logged mean reward      {result.logged_value:.5f}",
        "",
        f"naive replay lift       {format_percent(result.lifts['naive'])}"
        f"   (on {result.naive_matched_sessions:,} matched sessions, "
        f"{result.naive_match_rate:.1%} of the log)",
        f"corrected lift (SNIPS)  {format_percent(result.lifts['snips'])}"
        f"   95% interval [{format_percent(low)}, {format_percent(high)}]",
        f"true lift (referee)     {format_percent(result.true_lift)}",
        "",
        f"effective sample size   {overlap.effective_sample_size:,.0f}"
        f" of {result.n_sessions:,} ({overlap.ess_fraction:.2%})",
        f"largest single weight   {overlap.max_weight_share:.3%} of the total mass",
        f"weights hitting the cap {result.clipped_share:.3%} of sessions",
        f"minimum detectable lift {format_percent(result.minimum_detectable_lift)}",
        "",
        f"verdict                 {DECISION_LABELS.get(result.decision, '?')}"
        f" (exit code {result.exit_code})",
        f"reason                  {result.reason}",
    ]
    return "\n".join(lines)


def render_json(run: EvaluationRun) -> str:
    """The machine readable form. Everything the HTML page shows comes from this."""
    return json.dumps(run.as_dict(), indent=2, default=float) + "\n"


STYLESHEET = """
:root { color-scheme: light; }
body { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
       background: #ffffff; color: #15181d; margin: 0; padding: 32px; }
.page { max-width: 1120px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px 0; letter-spacing: -0.2px; }
h2 { font-size: 16px; margin: 32px 0 8px 0; border-bottom: 1px solid #d8dde3;
     padding-bottom: 6px; }
.subtitle { color: #5a6471; font-size: 13px; margin: 0 0 24px 0; }
.lede { background: #f4f6f8; border-left: 3px solid #15181d; padding: 12px 16px;
        font-size: 13px; line-height: 1.55; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 4px; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid #e4e8ec; }
th:first-child, td.policy { text-align: left; }
thead th { background: #eef1f4; font-weight: 600; color: #2b313a; }
td.policy { font-weight: 600; }
tbody tr:nth-child(even) { background: #fbfcfd; }
.verdict { display: inline-block; padding: 2px 9px; border-radius: 3px; font-size: 11px;
           font-weight: 700; letter-spacing: 0.4px; }
td .verdict.verdict-positive { background: #16653a; color: #ffffff; }
td .verdict.verdict-negative { background: #5a6471; color: #ffffff; }
td .verdict.verdict-refused { background: #8a2b14; color: #ffffff; }
.overclaim { color: #8a2b14; font-weight: 600; }
.footnote { color: #5a6471; font-size: 11.5px; margin-top: 10px; line-height: 1.5; }
"""


def _cell(value: float, flag_overclaim: float | None = None) -> str:
    text = html.escape(format_percent(value))
    if flag_overclaim is not None and value == value and value - flag_overclaim > 0.01:
        return f'<td class="overclaim">{text}</td>'
    return f"<td>{text}</td>"


def render_html(run: EvaluationRun, title: str = "replay-gap evaluation report") -> str:
    """A self contained HTML report, which is what the screenshots in the README are of.

    No external stylesheet and no scripts, so the page renders the same in a browser, in a
    headless capture and from a file URL. The verdict badge rules are namespaced under ``td``
    deliberately: an earlier version of this page had the badge class and the table cell class
    at equal specificity, the later rule won, and the verdict rendered in the cell's own colour
    on its own background. The screenshot tool now reads the computed style back and fails.
    """
    columns = estimator_columns(run)
    head_cells = "".join(f"<th>{html.escape(COLUMN_LABELS[c])}</th>" for c in columns)
    rows = []
    for result in run.results:
        overlap = result.overlap
        cells = "".join(_cell(result.lifts.get(c, float("nan")), result.true_lift) for c in columns)
        badge_class = DECISION_CLASS.get(result.decision, "verdict-negative")
        badge = html.escape(DECISION_LABELS.get(result.decision, "?"))
        rows.append(
            f'<tr><td class="policy">{html.escape(result.policy)}</td>'
            f"<td>{html.escape(format_percent(result.true_lift))}</td>"
            f"{cells}"
            f"<td>{overlap.effective_sample_size:,.0f}</td>"
            f"<td>{overlap.ess_fraction * 100:.2f}%</td>"
            f'<td><span class="verdict {badge_class}">{badge}</span></td></tr>'
        )
    incumbent = next((r for r in run.results if r.policy == "incumbent"), None)
    lede = (
        "Replaying the deployed ranker against the log it produced reports a lift of "
        f"{format_percent(incumbent.lifts['naive'])} over itself. Its true lift over itself is "
        f"{format_percent(incumbent.true_lift)}, because it is itself."
        if incumbent
        else "The referee's lift is exact. Every other column is an estimate of it."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{STYLESHEET}</style></head>
<body><div class="page">
<h1>{html.escape(title)}</h1>
<p class="subtitle">{run.n_sessions:,} logged sessions, {run.n_candidates} candidates per
session, {run.n_positions} slate positions. Configuration digest
{html.escape(run.config_digest)}.</p>
<div class="lede">{html.escape(lede)}</div>
<h2>Estimated lift over the deployed ranker</h2>
<table><thead><tr><th>policy</th><th>true lift</th>{head_cells}
<th>ESS</th><th>ESS share</th><th>verdict</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="footnote">The true lift column is computed by enumerating all
{run.n_candidates * (run.n_candidates - 1) * (run.n_candidates - 2)} ordered slates per session
and integrating the click model analytically, so it carries no sampling error. Figures shown in
red overstate the true lift by more than one point. REFUSED means the tool declined: overlap
between the candidate policy and the log is too poor for any offline estimate to be honest.</p>
</div></body></html>
"""
