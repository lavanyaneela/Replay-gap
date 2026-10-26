"""Candidate policies, including the two adversarial baselines the experiments rest on.

The point of the baseline set is not coverage of ranker families. It is that two of them are
built to break naive replay in opposite directions: one is the deployed policy wearing a hat,
which has a true lift of zero by construction, and one disagrees with the deployed policy almost
everywhere while being genuinely better. Any evaluation method that cannot separate those two is
not measuring model quality.
"""

from replay_gap.policies.base import Policy, ScorePolicy
from replay_gap.policies.baselines import (
    BlendPolicy,
    ContrarianPolicy,
    IncumbentPolicy,
    OraclePolicy,
    PopularityPolicy,
    UniformPolicy,
    build_baseline_policies,
    click_counts_from_log,
)
from replay_gap.policies.lightgbm_ranker import LightGBMRankerPolicy, train_lightgbm_ranker

__all__ = [
    "BlendPolicy",
    "ContrarianPolicy",
    "IncumbentPolicy",
    "LightGBMRankerPolicy",
    "OraclePolicy",
    "Policy",
    "PopularityPolicy",
    "ScorePolicy",
    "UniformPolicy",
    "build_baseline_policies",
    "click_counts_from_log",
    "train_lightgbm_ranker",
]
