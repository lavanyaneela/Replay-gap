"""The four estimators, and the inputs they share.

Every estimator is expressed as a function of a session index array over precomputed per session
quantities. That shape is not an optimisation: it is what makes the bootstrap honest. SNIPS is a
ratio of two sums, so resampling has to recompute the ratio rather than resample a mean, and the
lift of a candidate over the logging policy has to be resampled with both arms drawn on the same
sessions or the interval is wrong in a direction that flatters the result.
"""

from replay_gap.estimators.bootstrap import (
    BootstrapInterval,
    RatioEstimator,
    bootstrap_ratio_lifts,
)
from replay_gap.estimators.doubly_robust import (
    dr_value,
    fit_reward_model,
    predict_candidate_rewards,
    reward_model_terms,
)
from replay_gap.estimators.inputs import EstimatorInputs, build_estimator_inputs
from replay_gap.estimators.ips import importance_weights, ips_value
from replay_gap.estimators.naive import naive_match_mask, naive_match_rate, naive_replay_value
from replay_gap.estimators.snips import snips_value

__all__ = [
    "BootstrapInterval",
    "EstimatorInputs",
    "RatioEstimator",
    "bootstrap_ratio_lifts",
    "build_estimator_inputs",
    "dr_value",
    "fit_reward_model",
    "importance_weights",
    "ips_value",
    "naive_match_mask",
    "naive_match_rate",
    "naive_replay_value",
    "predict_candidate_rewards",
    "reward_model_terms",
    "snips_value",
]
