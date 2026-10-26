"""The pipeline, and the exit code it produces for each kind of answer.

The four outcomes are the product. A test suite that checked the numbers and not the exit codes
would be testing the library and not the tool.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from replay_gap.diagnostics.overlap import OverlapReport
from replay_gap.estimators.bootstrap import BootstrapInterval
from replay_gap.pipeline import (
    EXIT_NOT_POSITIVE,
    EXIT_POSITIVE,
    EXIT_REFUSED,
    _decide,
    build_policies,
    compare_exit_code,
    evaluate_policy,
    run_evaluation,
)
from replay_gap.policies.baselines import ContrarianPolicy


@pytest.fixture(scope="module")
def small_run(small_config, small_fixture):
    policies = build_policies(small_config, small_fixture, with_ranker=False)
    return run_evaluation(small_config, small_fixture, policies)


def usable_overlap(**overrides) -> OverlapReport:
    base = {
        "policy": "p",
        "n_sessions": 100,
        "effective_sample_size": 90.0,
        "ess_fraction": 0.9,
        "max_weight_share": 0.01,
        "weight_p99": 2.0,
        "top1_agreement": 0.3,
        "unseen_exposure": 0.0,
        "usable": True,
        "reason": "overlap is sufficient for an offline estimate",
    }
    base.update(overrides)
    return OverlapReport(**base)


def test_poor_overlap_refuses_before_any_estimate_is_considered():
    """The order matters: a number printed next to a warning gets read, the warning does not."""
    overlap = usable_overlap(usable=False, reason="effective sample size is 1% of the log")
    decision, code, reason = _decide(overlap, BootstrapInterval(0.5, 0.4, 0.6, 0.05, 100))
    assert (decision, code) == ("refused", EXIT_REFUSED)
    assert "online experiment" in reason


def test_an_interval_entirely_above_zero_is_a_positive_result():
    decision, code, _ = _decide(usable_overlap(), BootstrapInterval(0.1, 0.02, 0.18, 0.04, 100))
    assert (decision, code) == ("positive", EXIT_POSITIVE)


def test_an_interval_straddling_zero_is_the_tool_answering_no():
    decision, code, reason = _decide(
        usable_overlap(), BootstrapInterval(0.01, -0.04, 0.06, 0.02, 100)
    )
    assert (decision, code) == ("not_positive", EXIT_NOT_POSITIVE)
    assert "not distinguishable" in reason


def test_an_interval_entirely_below_zero_is_also_the_tool_answering_no():
    decision, code, reason = _decide(
        usable_overlap(), BootstrapInterval(-0.1, -0.18, -0.02, 0.04, 100)
    )
    assert (decision, code) == ("not_positive", EXIT_NOT_POSITIVE)
    assert "below zero" in reason


def test_an_interval_that_is_not_finite_is_a_refusal_rather_than_a_negative():
    decision, code, _ = _decide(
        usable_overlap(), BootstrapInterval(float("nan"), float("nan"), float("nan"), 0.0, 10)
    )
    assert (decision, code) == ("refused", EXIT_REFUSED)


def result_with(decision: str) -> object:
    from replay_gap.pipeline import PolicyResult

    return PolicyResult(
        policy=decision,
        n_sessions=10,
        logged_value=1.0,
        true_value=1.0,
        true_lift=0.0,
        naive_match_rate=0.3,
        naive_matched_sessions=3,
        clipped_share=0.0,
        decision=decision,
    )


def test_a_refusal_anywhere_outranks_a_win_anywhere():
    """Somebody will otherwise ship the unanswered question on a naive replay."""
    results = [result_with("positive"), result_with("refused"), result_with("not_positive")]
    assert compare_exit_code(results) == EXIT_REFUSED


def test_a_win_with_no_refusal_is_a_positive_comparison():
    assert compare_exit_code([result_with("positive"), result_with("not_positive")]) == (
        EXIT_POSITIVE
    )


def test_a_comparison_where_nothing_wins_answers_no():
    assert compare_exit_code([result_with("not_positive")]) == EXIT_NOT_POSITIVE


def test_an_empty_comparison_answers_no_rather_than_claiming_a_win():
    assert compare_exit_code([]) == EXIT_NOT_POSITIVE


def test_every_policy_in_the_run_gets_every_estimator(small_run):
    for result in small_run.results:
        assert set(result.lifts) >= {"naive", "naive_first_slot", "ips", "snips", "dr"}
        assert set(result.intervals) == set(result.lifts)
        assert set(result.standard_errors) == set(result.lifts)


def test_the_incumbent_is_refereed_at_exactly_zero_lift(small_run):
    assert small_run.by_name("incumbent").true_lift == pytest.approx(0.0, abs=1e-12)


def test_naive_replay_overstates_the_incumbent_by_several_points(small_run):
    """The headline of the whole repository, as an assertion about the measured run."""
    incumbent = small_run.by_name("incumbent")
    assert incumbent.lifts["naive"] > incumbent.true_lift + 0.02


def test_the_corrected_estimator_is_closer_to_the_truth_than_naive_replay_on_average(small_run):
    naive = np.mean([abs(r.error("naive")) for r in small_run.results])
    snips = np.mean([abs(r.error("snips")) for r in small_run.results])
    assert snips < naive


def test_the_uniform_policy_is_judged_worse_than_the_incumbent(small_run):
    assert small_run.by_name("uniform").true_lift < -0.02


def test_a_sharply_served_policy_loses_most_of_the_effective_sample(small_run):
    mild = small_run.by_name("contrarian_lift").overlap.ess_fraction
    sharp = small_run.by_name("contrarian_sharp").overlap.ess_fraction
    assert sharp < mild


def test_serving_a_policy_greedily_does_not_change_which_sessions_naive_replay_keeps(small_run):
    assert small_run.by_name("contrarian_sharp").naive_match_rate == pytest.approx(
        small_run.by_name("contrarian_lift").naive_match_rate
    )


def test_the_run_can_be_looked_up_by_policy_name(small_run):
    assert small_run.by_name("oracle").policy == "oracle"
    with pytest.raises(KeyError, match="no result for policy"):
        small_run.by_name("absent")


def test_the_run_reports_the_policies_it_covered(small_run):
    assert "incumbent" in small_run.policy_names
    assert len(small_run.policy_names) == len(small_run.results)


def test_the_run_serialises_to_plain_data(small_run):
    payload = small_run.as_dict()
    assert payload["n_sessions"] == small_run.n_sessions
    assert payload["results"][0]["overlap"]["policy"] == payload["results"][0]["policy"]


def test_the_error_of_an_estimator_is_its_distance_from_the_referee(small_run):
    result = small_run.by_name("oracle")
    assert result.error("snips") == pytest.approx(result.lifts["snips"] - result.true_lift)


def test_a_policy_evaluated_without_an_outcome_model_omits_the_doubly_robust_column(
    small_config, small_fixture
):
    policies = build_policies(small_config, small_fixture, with_ranker=False)
    result = evaluate_policy(small_config, small_fixture, policies["oracle"], None)
    assert "dr" not in result.lifts
    assert "snips" in result.lifts


def test_an_evaluation_run_is_reproducible(small_config, small_fixture):
    policies = {"oracle": build_policies(small_config, small_fixture, with_ranker=False)["oracle"]}
    first = run_evaluation(small_config, small_fixture, policies, with_reward_model=False)
    second = run_evaluation(small_config, small_fixture, policies, with_reward_model=False)
    assert first.by_name("oracle").lifts == second.by_name("oracle").lifts
    assert first.by_name("oracle").intervals == second.by_name("oracle").intervals


def test_tightening_the_overlap_floor_turns_an_answer_into_a_refusal(small_config, small_fixture):
    """The refusal is a policy decision in configuration, not a hardcoded judgement."""
    strict = replace(
        small_config,
        diagnostics=replace(small_config.diagnostics, min_ess_fraction=0.999),
    )
    policy = ContrarianPolicy(small_fixture.world, small_fixture.logging_policy)
    result = evaluate_policy(strict, small_fixture, policy, None)
    assert result.exit_code == EXIT_REFUSED
    assert result.decision == "refused"


def test_the_pipeline_computes_the_baseline_itself_when_none_is_supplied(
    small_config, small_fixture
):
    policy = ContrarianPolicy(small_fixture.world, small_fixture.logging_policy)
    with_baseline = evaluate_policy(
        small_config, small_fixture, policy, None, baseline_true_value=None
    )
    assert np.isfinite(with_baseline.true_lift)


@pytest.mark.slow
def test_building_policies_with_the_ranker_adds_exactly_one_policy(small_config, small_fixture):
    without = build_policies(small_config, small_fixture, with_ranker=False)
    with_ranker = build_policies(small_config, small_fixture, with_ranker=True)
    assert set(with_ranker) - set(without) == {"lightgbm_ranker"}
