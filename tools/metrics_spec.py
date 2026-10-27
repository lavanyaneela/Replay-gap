"""The contract between the measurements and the prose: every published figure, and its anchor.

A metric here has three parts. Where it comes from, which is always a JSON file some code wrote
rather than a number somebody typed. How it is formatted, because a document writes 27.1% where
the JSON holds 0.2710843373. And the anchor phrases, which are the exact wording a document has
to use around it.

The anchors are the part that matters. A checker that searches a document for the digits of a
metric passes happily while the sentence around them has gone stale: "45 disagreements" is found
in a document that now says "45 scenarios". Anchors name the number together with two or three
words of its context, so the check fails when the claim changes and not when the paragraph gets
rewrapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Metric:
    """One published figure, its source, its formatting, and the phrases that guard it."""

    key: str
    source: str
    path: tuple[str, ...]
    description: str
    unit: str = "raw"
    digits: int = 2
    anchors: tuple[str, ...] = ()
    scale: float = 1.0
    checked: bool = True

    def format(self, value: float) -> str:
        """Render the value exactly as a document is required to write it."""
        value = value * self.scale
        if self.unit == "percent":
            return f"{value * 100:.{self.digits}f}%"
        if self.unit == "signed_percent":
            return f"{value * 100:+.{self.digits}f}%"
        if self.unit == "integer":
            return f"{int(round(value))}"
        if self.unit == "thousands":
            return f"{int(round(value)):,}"
        if self.unit in {"ratio", "float"}:
            return f"{value:.{self.digits}f}"
        return f"{value:.{self.digits}f}"


@dataclass(frozen=True)
class Row:
    """A whole published table row, guarded as one string rather than cell by cell.

    A table cell has no words around it, so an anchor phrase cannot be built for it the way it
    can for a sentence. Guarding the rendered row instead is stronger than guarding each cell:
    any value changing, any column moving, and any cell being edited by hand all break the exact
    string, and one entry covers the row.
    """

    key: str
    source: str
    description: str
    template: str
    cells: tuple[tuple[str, tuple[str, ...], str, int], ...]

    def render(self, payload: dict, dig) -> str:
        """Format the row exactly as the document must contain it."""
        values = []
        for _name, path, unit, digits in self.cells:
            metric = Metric(
                key=_name, source=self.source, path=path, description="", unit=unit, digits=digits
            )
            values.append(metric.format(dig(payload, path, self.source)))
        return self.template.format(*values)


@dataclass(frozen=True)
class Spec:
    """Every metric the repository publishes, plus the documents that are allowed to quote them."""

    metrics: tuple[Metric, ...]
    rows: tuple[Row, ...] = field(default=())
    documents: tuple[str, ...] = field(default=())

    def by_key(self, key: str) -> Metric:
        for metric in self.metrics:
            if metric.key == key:
                return metric
        raise KeyError(f"no metric named {key}")


DOCUMENTS: tuple[str, ...] = (
    "README.md",
    "docs/defense-guide.md",
    "docs/adr/ADR-001-synthetic-world-over-public-dataset.md",
    "docs/adr/ADR-002-exact-ground-truth-by-slate-enumeration.md",
    "docs/adr/ADR-003-duckdb-default-postgres-parity.md",
    "docs/adr/ADR-004-snips-default-and-the-cost-of-clipping.md",
    "docs/adr/ADR-005-what-this-repository-cannot-establish.md",
)


METRICS: tuple[Metric, ...] = (
    # The suite itself.
    Metric(
        key="tests_total",
        source="suite",
        path=("tests_total",),
        description="tests in the suite",
        unit="integer",
        anchors=("{} tests",),
    ),
    Metric(
        key="tests_executed",
        source="suite",
        path=("tests_executed",),
        description="tests that ran on this machine, excluding the PostgreSQL parity module",
        unit="integer",
        anchors=("{} of them run", "{} tests run"),
    ),
    Metric(
        key="coverage_line_pct",
        source="suite",
        path=("coverage_line_pct",),
        description="line coverage of the package",
        unit="percent",
        digits=0,
        anchors=("{} line coverage", "coverage at {}"),
    ),
    # The headline: replaying the deployed ranker against its own log.
    Metric(
        key="incumbent_naive_lift",
        source="evaluation",
        path=("incumbent", "lifts", "naive"),
        description="naive replay's verdict on the deployed ranker, against itself",
        unit="signed_percent",
        anchors=("{} better than itself", "log reports {}", "replay lift {}"),
    ),
    Metric(
        key="incumbent_naive_matched",
        source="evaluation",
        path=("incumbent", "naive_matched_sessions"),
        description="sessions naive replay keeps when replaying the deployed ranker",
        unit="thousands",
        anchors=("{} sessions", "on {} of"),
    ),
    Metric(
        key="incumbent_match_rate",
        source="evaluation",
        path=("incumbent", "naive_match_rate"),
        description="share of the log naive replay keeps for the deployed ranker",
        unit="percent",
        digits=1,
        anchors=("{} of the log",),
    ),
    Metric(
        key="log_sessions",
        source="evaluation",
        path=("n_sessions",),
        description="sessions in the committed log",
        unit="thousands",
        anchors=("{} session", "{} logged sessions"),
    ),
    # The refusal.
    Metric(
        key="sharp_ess",
        source="evaluation",
        path=("contrarian_sharp", "overlap", "effective_sample_size"),
        description="effective sample size for the greedily served ranker",
        unit="thousands",
        anchors=("{} of", "worth {} sessions"),
    ),
    Metric(
        key="sharp_ess_fraction",
        source="evaluation",
        path=("contrarian_sharp", "overlap", "ess_fraction"),
        description="effective sample size as a share of the log, greedily served",
        unit="percent",
        anchors=("sample size is {}", "ends at {}", "{} of the log"),
    ),
    Metric(
        key="sharp_snips_lift",
        source="evaluation",
        path=("contrarian_sharp", "lifts", "snips"),
        description="what SNIPS would have reported for the refused policy",
        unit="signed_percent",
        anchors=("would have reported {}", "SNIPS says {}"),
    ),
    Metric(
        key="sharp_ips_lift",
        source="evaluation",
        path=("contrarian_sharp", "lifts", "ips"),
        description="what plain IPS reports for the refused policy",
        unit="signed_percent",
        anchors=("IPS reports {}", "IPS says {}", "a lift of {}"),
    ),
    Metric(
        key="sharp_true_lift",
        source="evaluation",
        path=("contrarian_sharp", "true_lift"),
        description="the true lift of the refused policy",
        unit="signed_percent",
        anchors=("true lift is {}", "truth is {}"),
    ),
    # Where the corrected estimator itself is wrong.
    Metric(
        key="clone_snips_lift",
        source="evaluation",
        path=("incumbent_clone", "lifts", "snips"),
        description="SNIPS on a policy that is a noisy copy of the incumbent",
        unit="signed_percent",
        anchors=("ship at {}", "calls it {}", "SNIPS says {}"),
    ),
    Metric(
        key="clone_true_lift",
        source="evaluation",
        path=("incumbent_clone", "true_lift"),
        description="the true lift of the noisy copy",
        unit="signed_percent",
        anchors=("true lift of {}", "true lift is {}", "a {} policy", "actually {}"),
    ),
    Metric(
        key="clone_ess_fraction",
        source="evaluation",
        path=("incumbent_clone", "overlap", "ess_fraction"),
        description="effective sample size for the noisy copy",
        unit="percent",
        digits=0,
        anchors=("{} of the log", "at {} effective"),
    ),
    Metric(
        key="units_error_lift",
        source="evaluation",
        path=("units_error_lift",),
        description="what the units error in the war story reported for the deployed ranker",
        unit="signed_percent",
        anchors=("reporting {} for", "plausible looking {}"),
    ),
    Metric(
        key="sharp_match_rate",
        source="evaluation",
        path=("contrarian_sharp", "naive_match_rate"),
        description="share of the log a replay keeps for the greedily served ranker",
        unit="percent",
        digits=1,
        anchors=("first slot in {} of", "{} of the log"),
    ),
    Metric(
        key="sharp_naive_lift",
        source="evaluation",
        path=("contrarian_sharp", "lifts", "naive"),
        description="what a replay reports for the greedily served ranker",
        unit="signed_percent",
        anchors=("replay lift {}", "replay reports {}"),
    ),
    Metric(
        key="min_ess_fraction",
        source="evaluation",
        path=("thresholds", "min_ess_fraction"),
        description="the overlap floor below which the tool refuses",
        unit="percent",
        anchors=("below the {} floor", "{} floor", "{} overlap floor"),
    ),
    Metric(
        key="alpha",
        source="evaluation",
        path=("thresholds", "alpha"),
        description="the two sided significance level the intervals use",
        unit="ratio",
        digits=2,
        anchors=("alpha {}", "alpha of {}"),
    ),
    Metric(
        key="power",
        source="evaluation",
        path=("thresholds", "power"),
        description="the power the minimum detectable lift is computed at",
        unit="percent",
        digits=0,
        anchors=("{} power", "power {}"),
    ),
    # E1.
    Metric(
        key="e1_scenarios",
        source="e1_sign_agreement",
        path=("n_scenarios",),
        description="candidate rankers sampled in E1",
        unit="integer",
        anchors=("{} candidate rankers", "{} sampled rankers"),
    ),
    Metric(
        key="e1_decisive",
        source="e1_sign_agreement",
        path=("n_decisive",),
        description="E1 scenarios whose true lift is large enough to act on",
        unit="integer",
        anchors=("of {} decisive", "{} decisive"),
    ),
    Metric(
        key="e1_naive_sign_disagreement_rate",
        source="e1_sign_agreement",
        path=("naive_sign_disagreement_rate",),
        description="how often naive replay disagrees with the referee about the sign",
        unit="percent",
        digits=1,
        anchors=("{} of decisive", "sign in {}", "wrong in {}"),
    ),
    Metric(
        key="e1_naive_sign_disagreements",
        source="e1_sign_agreement",
        path=("naive_sign_disagreements",),
        description="count of sign disagreements in E1",
        unit="integer",
        anchors=("{} of", "in {} of"),
    ),
    Metric(
        key="e1_snips_sign_disagreement_rate",
        source="e1_sign_agreement",
        path=("snips_sign_disagreement_rate",),
        description="how often SNIPS disagrees with the referee about the sign",
        unit="percent",
        digits=1,
        anchors=("SNIPS {}", "to {}"),
    ),
    Metric(
        key="e1_naive_overclaim_rate",
        source="e1_sign_agreement",
        path=("naive_overclaim_rate",),
        description="share of E1 scenarios where naive replay came in above the truth",
        unit="percent",
        digits=0,
        anchors=("{} of scenarios", "above the truth in {}"),
    ),
    Metric(
        key="e1_worst_true_lift",
        source="e1_sign_agreement",
        path=("worst_overclaim_true_lift",),
        description="true lift of the worst overclaimed ranker",
        unit="signed_percent",
        anchors=("true lift is {}", "worth {}"),
    ),
    Metric(
        key="e1_worst_naive_lift",
        source="e1_sign_agreement",
        path=("worst_overclaim_naive_lift",),
        description="what naive replay said about the worst overclaimed ranker",
        unit="signed_percent",
        anchors=("read as {}", "replay said {}"),
    ),
    Metric(
        key="e1_worst_gap",
        source="e1_sign_agreement",
        path=("worst_overclaim_gap",),
        description="the gap on the worst overclaim",
        unit="percent",
        anchors=("gap of {}", "{} apart"),
    ),
    # E2.
    Metric(
        key="e2_end_ess_fraction",
        source="e2_overlap_curve",
        path=("end_ess_fraction",),
        description="what the log is worth at the end of the optimistic sweep",
        unit="percent",
        anchors=("falls to {}", "{} of its nominal"),
    ),
    Metric(
        key="e2_collapse_factor",
        source="e2_overlap_curve",
        path=("ess_collapse_factor",),
        description="how far the log falls on the model axis alone",
        unit="integer",
        anchors=("factor of {}", "{} times smaller"),
    ),
    Metric(
        key="e2_end_ess_fraction_sharpened",
        source="e2_overlap_curve",
        path=("end_ess_fraction_sharpened",),
        description="what the log is worth once serving is sharpened too",
        unit="percent",
        anchors=("sweep ends at {}", "sharpened to {}"),
    ),
    Metric(
        key="e2_naive_lift_at_end",
        source="e2_overlap_curve",
        path=("naive_lift_at_end",),
        description="what naive replay still reports at the end of the sweep",
        unit="signed_percent",
        anchors=("still reports {}", "reports {} at"),
    ),
    Metric(
        key="e2_true_lift_at_end",
        source="e2_overlap_curve",
        path=("true_lift_at_end",),
        description="the truth at the end of the sweep",
        unit="signed_percent",
        anchors=("truth is {}", "true lift is {}"),
    ),
    # E3.
    Metric(
        key="e3_replicates",
        source="e3_estimator_comparison",
        path=("replicates",),
        description="independent replicates in E3",
        unit="integer",
        anchors=("{} replicates", "all {} replicates", "{} independent"),
    ),
    Metric(
        key="e3_popularity_true_lift",
        source="e3_estimator_comparison",
        path=("summary", "popularity", "mean_true_lift"),
        description="true lift of the popularity baseline",
        unit="signed_percent",
        anchors=("true lift is {}", "worth {} to"),
    ),
    Metric(
        key="e3_popularity_naive_sign_errors",
        source="e3_estimator_comparison",
        path=("summary", "popularity", "naive_sign_errors"),
        description="replicates where naive replay got the popularity baseline's sign wrong",
        unit="integer",
        anchors=("{} of", "all {} "),
    ),
    Metric(
        key="e3_serving_true_gap",
        source="e3_estimator_comparison",
        path=("widest_serving_blindness", "true_lift_gap"),
        description="how far apart two serving temperatures of one ranker truly are",
        unit="percent",
        anchors=("differ by {}", "{} apart"),
    ),
    Metric(
        key="e3_oracle_sharp_snips_rmse",
        source="e3_estimator_comparison",
        path=("summary", "oracle_sharp", "snips_rmse"),
        description="SNIPS error on the greedily served oracle",
        unit="percent",
        anchors=("SNIPS is at {}", "SNIPS at {}"),
    ),
    Metric(
        key="e3_oracle_sharp_dr_rmse",
        source="e3_estimator_comparison",
        path=("summary", "oracle_sharp", "dr_rmse"),
        description="doubly robust error on the greedily served oracle",
        unit="percent",
        anchors=("{} for the doubly robust", "doubly robust at {}"),
    ),
    # E4.
    Metric(
        key="e4_worst_detectable",
        source="e4_detectable_lift",
        path=("worst_minimum_detectable_lift",),
        description="the largest minimum detectable lift across the policy set",
        unit="percent",
        anchors=("resolve {}", "below {}", "worst case is {}"),
    ),
    Metric(
        key="e4_median_detectable",
        source="e4_detectable_lift",
        path=("median_minimum_detectable_lift",),
        description="median minimum detectable lift across the policy set",
        unit="percent",
        anchors=("median of {}", "typically {}"),
    ),
    Metric(
        key="e4_policies_above_target",
        source="e4_detectable_lift",
        path=("policies_above_target",),
        description="policies whose minimum detectable lift exceeds one percent",
        unit="integer",
        anchors=("{} of", "for {} of"),
    ),
    Metric(
        key="e4_informative_policies",
        source="e4_detectable_lift",
        path=("n_informative_policies",),
        description="policies with a usable resolution figure",
        unit="integer",
        anchors=("of {} policies", "of the {} "),
    ),
    Metric(
        key="e4_worst_sessions",
        source="e4_detectable_lift",
        path=("worst_sessions_for_one_percent",),
        description="sessions needed to resolve a one percent lift in the worst case",
        unit="thousands",
        anchors=("{} sessions", "about {} "),
    ),
    # E5.
    Metric(
        key="e5_crossover_factor",
        source="e5_propensity_misspecification",
        path=("earliest_crossover_factor",),
        description="corruption factor at which the correction stops being worth having",
        unit="ratio",
        digits=1,
        anchors=("factor of {}", "at {} the"),
    ),
    Metric(
        key="e5_ess_at_crossover",
        source="e5_propensity_misspecification",
        path=("ess_fraction_at_earliest_crossover",),
        description="what the diagnostic reads at that crossover",
        unit="percent",
        digits=1,
        anchors=("reads {}", "still {}"),
    ),
    Metric(
        key="e5_crossings_where_ess_rose",
        source="e5_propensity_misspecification",
        path=("n_crossings_where_ess_rose",),
        description="crossovers where the diagnostic improved as the estimate decayed",
        unit="integer",
        anchors=("{} of", "in all {} "),
    ),
    Metric(
        key="e5_crossed",
        source="e5_propensity_misspecification",
        path=("n_crossed",),
        description="policies whose correction stopped being worth having",
        unit="integer",
        anchors=("of {} ", "all {} "),
    ),
)


def _sweep_row(step: int, share: str) -> Row:
    """One row of the E2 overlap sweep, as the README prints it."""
    return Row(
        key=f"e2_row_{step:02d}",
        source="e2_overlap_curve",
        description=f"E2 sweep row at {share} toward true relevance",
        template="| " + share + " | {} | {} | {} | {} | {} |",
        cells=(
            ("true_lift", ("model_only", str(step), "true_lift"), "signed_percent", 2),
            ("naive_lift", ("model_only", str(step), "naive_lift"), "signed_percent", 2),
            ("snips_lift", ("model_only", str(step), "snips_lift"), "signed_percent", 2),
            (
                "effective_sample_size",
                ("model_only", str(step), "effective_sample_size"),
                "thousands",
                0,
            ),
            ("max_raw_weight", ("model_only", str(step), "max_raw_weight"), "thousands", 0),
        ),
    )


def _resolution_row(policy: str) -> Row:
    """One row of the E4 resolution table."""
    return Row(
        key=f"e4_row_{policy}",
        source="e4_detectable_lift",
        description=f"E4 resolution row for {policy}",
        template="| " + policy + " | {} | {} | {} |",
        cells=(
            ("true_lift", ("by_policy", policy, "true_lift"), "signed_percent", 2),
            ("snips_lift", ("by_policy", policy, "snips_lift"), "signed_percent", 2),
            (
                "minimum_detectable_lift",
                ("by_policy", policy, "minimum_detectable_lift"),
                "percent",
                2,
            ),
        ),
    )


def _cli_rows(policy: str, prefix: str) -> tuple[Row, ...]:
    """The lines of a captured CLI block in the README, guarded exactly as printed.

    These are not example invocations. They are the output of a command that ran, they carry
    the document's headline and its refusal, and an audit found that excluding fenced blocks
    left them as the only unchecked numbers on the first screen.
    """
    return (
        Row(
            key=f"cli_{prefix}_naive",
            source="evaluation",
            description=f"README CLI block for {policy}: the naive replay line",
            template="naive replay lift       {}   (on {} matched sessions, {} of the log)",
            cells=(
                ("lift", (policy, "lifts", "naive"), "signed_percent", 2),
                ("matched", (policy, "naive_matched_sessions"), "thousands", 0),
                ("rate", (policy, "naive_match_rate"), "percent", 1),
            ),
        ),
        Row(
            key=f"cli_{prefix}_true",
            source="evaluation",
            description=f"README CLI block for {policy}: the referee line",
            template="true lift (referee)     {}",
            cells=(("true_lift", (policy, "true_lift"), "signed_percent", 2),),
        ),
        Row(
            key=f"cli_{prefix}_ess",
            source="evaluation",
            description=f"README CLI block for {policy}: the effective sample size line",
            template="effective sample size   {} of 50,000 ({})",
            cells=(
                (
                    "ess",
                    (policy, "overlap", "effective_sample_size"),
                    "thousands",
                    0,
                ),
                ("fraction", (policy, "overlap", "ess_fraction"), "percent", 2),
            ),
        ),
    )


def _rmse_row(policy: str, label: str) -> Row:
    """One row of the E3 estimator comparison table."""
    return Row(
        key=f"e3_row_{policy}",
        source="e3_estimator_comparison",
        description=f"E3 estimator comparison row for {policy}",
        template="| " + label + " | {} | {} | {} | {} | {} |",
        cells=(
            ("mean_true_lift", ("summary", policy, "mean_true_lift"), "signed_percent", 2),
            ("naive_rmse", ("summary", policy, "naive_rmse"), "percent", 2),
            ("ips_rmse", ("summary", policy, "ips_rmse"), "percent", 2),
            ("snips_rmse", ("summary", policy, "snips_rmse"), "percent", 2),
            ("dr_rmse", ("summary", policy, "dr_rmse"), "percent", 2),
        ),
    )


ROWS: tuple[Row, ...] = (
    Row(
        key="e1_error_row_naive",
        source="e1_sign_agreement",
        description="E1 mean error row for naive replay",
        template="| mean signed error | {} | {} |",
        cells=(
            ("naive_mean_signed_error", ("naive_mean_signed_error",), "signed_percent", 2),
            ("snips_mean_signed_error", ("snips_mean_signed_error",), "signed_percent", 2),
        ),
    ),
    Row(
        key="e1_error_row_absolute",
        source="e1_sign_agreement",
        description="E1 mean absolute error row",
        template="| mean absolute error | {} | {} |",
        cells=(
            ("naive_mean_absolute_error", ("naive_mean_absolute_error",), "percent", 2),
            ("snips_mean_absolute_error", ("snips_mean_absolute_error",), "percent", 2),
        ),
    ),
    Row(
        key="e3_serving_row_mild",
        source="e3_estimator_comparison",
        description="E3 serving blindness row, stochastic",
        template="| oracle, served stochastically | {} | {} |",
        cells=(
            ("true", ("widest_serving_blindness", "true_lift_mild"), "signed_percent", 2),
            ("naive", ("widest_serving_blindness", "naive_lift_mild"), "signed_percent", 2),
        ),
    ),
    Row(
        key="e3_serving_row_sharp",
        source="e3_estimator_comparison",
        description="E3 serving blindness row, greedy",
        template="| oracle, served greedily | {} | {} |",
        cells=(
            ("true", ("widest_serving_blindness", "true_lift_sharp"), "signed_percent", 2),
            ("naive", ("widest_serving_blindness", "naive_lift_sharp"), "signed_percent", 2),
        ),
    ),
    _sweep_row(0, "0%"),
    _sweep_row(3, "25%"),
    _sweep_row(6, "50%"),
    _sweep_row(9, "75%"),
    _sweep_row(12, "100%"),
    _resolution_row("incumbent_clone"),
    _resolution_row("popularity"),
    _resolution_row("uniform"),
    _resolution_row("contrarian_lift"),
    _resolution_row("contrarian_sharp"),
    _resolution_row("oracle"),
    _rmse_row("popularity", "popularity"),
    _rmse_row("oracle", "oracle"),
    _rmse_row("oracle_sharp", "oracle, greedily served"),
    _rmse_row("contrarian_lift", "contrarian_lift"),
    _rmse_row("contrarian_sharp", "contrarian_sharp"),
    *_cli_rows("incumbent", "incumbent"),
    *_cli_rows("contrarian_sharp", "sharp"),
    Row(
        key="cli_incumbent_snips",
        source="evaluation",
        description="README CLI block for the incumbent: the corrected estimate line",
        template="corrected lift (SNIPS)  {}   95% interval [{}, {}]",
        cells=(
            ("lift", ("incumbent", "lifts", "snips"), "signed_percent", 2),
            ("low", ("incumbent", "intervals", "snips", "0"), "signed_percent", 2),
            ("high", ("incumbent", "intervals", "snips", "1"), "signed_percent", 2),
        ),
    ),
    Row(
        key="cli_sharp_snips",
        source="evaluation",
        description="README CLI block for the refused policy: the corrected estimate line",
        template="corrected lift (SNIPS)  {}   95% interval [{}, {}]",
        cells=(
            ("lift", ("contrarian_sharp", "lifts", "snips"), "signed_percent", 2),
            ("low", ("contrarian_sharp", "intervals", "snips", "0"), "signed_percent", 2),
            ("high", ("contrarian_sharp", "intervals", "snips", "1"), "signed_percent", 2),
        ),
    ),
    Row(
        key="cli_incumbent_weights",
        source="evaluation",
        description="README CLI block for the incumbent: the weight diagnostics lines",
        template=(
            "largest single weight   {} of the total mass weights hitting the cap {} of sessions"
        ),
        cells=(
            ("share", ("incumbent", "overlap", "max_weight_share"), "percent", 3),
            ("clipped", ("incumbent", "clipped_share"), "percent", 3),
        ),
    ),
    Row(
        key="cli_sharp_reason",
        source="evaluation",
        description="README CLI block for the refused policy: the refusal reason",
        template="effective sample size is {} of the log, below the {} floor.",
        cells=(
            ("fraction", ("contrarian_sharp", "overlap", "ess_fraction"), "percent", 2),
            ("floor", ("thresholds", "min_ess_fraction"), "percent", 2),
        ),
    ),
    Row(
        key="cli_logged_mean_reward",
        source="evaluation",
        description="README CLI block: the logged mean reward line",
        template="logged mean reward      {}",
        cells=(("reward", ("logged_mean_reward",), "float", 5),),
    ),
)

SPEC = Spec(metrics=METRICS, rows=ROWS, documents=DOCUMENTS)
