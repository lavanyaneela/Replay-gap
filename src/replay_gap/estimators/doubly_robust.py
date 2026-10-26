"""The doubly robust estimator, and the regime where it is the worst of the four.

DR adds an outcome model to IPS: predict the reward of every slate the candidate policy might
have served, then correct that prediction on the slate that was actually served. When the
outcome model is good it cuts variance sharply. The catch, and the reason this estimator gets an
experiment rather than a sentence, is that the outcome model here is fitted on the same
confounded log. It is accurate on the slates the logging policy serves often and extrapolating
everywhere else, so for a candidate that disagrees with the logging policy the direct part of DR
is evaluated almost entirely out of distribution. E3 measures where that crossover sits.
"""

from __future__ import annotations

import numpy as np

from replay_gap.config import GBDTConfig
from replay_gap.evaluation import PolicyTrace
from replay_gap.policies.lightgbm_ranker import build_features
from replay_gap.simulator.generate import InteractionLog
from replay_gap.simulator.user_model import World

_PREDICT_CHUNK = 200_000


def _impression_features(
    world: World, context_ids: np.ndarray, items: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Context, item, their product, and the position, one row per impression."""
    base = build_features(world, context_ids, items)
    return np.concatenate([base, positions.reshape(-1, 1).astype(np.float64)], axis=1)


def fit_reward_model(world: World, log: InteractionLog, cfg: GBDTConfig):
    """Fit a click model on the impressions the log contains, and only those.

    Pinned to a single thread and LightGBM's deterministic mode for the same reason the ranker
    is: a published number that moves with the thread count is a fact about the runner.
    """
    import lightgbm as lgb

    shown = log.slate_items()
    n_sessions, n_positions = shown.shape
    positions = np.tile(np.arange(n_positions), n_sessions)
    context_ids = np.repeat(log.context_id, n_positions)
    items = shown.reshape(-1, 1)
    features = _impression_features(world, context_ids, items, positions)
    target = log.clicks.reshape(-1).astype(np.float64)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        random_state=cfg.seed,
        n_jobs=1,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(features, target)
    return model


def predict_candidate_rewards(world: World, log: InteractionLog, model) -> np.ndarray:
    """Predicted click probability for every candidate in every position.

    Shape (n_sessions, n_positions, n_candidates). Chunked, because at fifty thousand sessions
    this is one and a half million rows and LightGBM will happily allocate the lot.
    """
    n_sessions, n_candidates = log.candidates.shape
    n_positions = log.n_positions
    out = np.empty((n_sessions, n_positions, n_candidates), dtype=np.float64)
    rows_per_session = n_positions * n_candidates
    chunk_sessions = max(1, _PREDICT_CHUNK // rows_per_session)
    for start in range(0, n_sessions, chunk_sessions):
        stop = min(start + chunk_sessions, n_sessions)
        block = stop - start
        context_ids = np.repeat(log.context_id[start:stop], rows_per_session)
        items = np.tile(log.candidates[start:stop], (1, n_positions)).reshape(-1, 1)
        positions = np.tile(np.repeat(np.arange(n_positions), n_candidates), block)
        features = _impression_features(world, context_ids, items, positions)
        predicted = model.predict(features, num_threads=1)
        out[start:stop] = np.asarray(predicted).reshape(block, n_positions, n_candidates)
    return out


def reward_model_terms(
    log: InteractionLog,
    trace: PolicyTrace,
    candidate_rewards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The two outcome model terms DR needs, per session.

    The first is the model's expected reward for the candidate policy, taken over the policy's
    whole slate distribution rather than over a sampled slate. The second is the model's
    prediction for the slate that was actually served, which is what the importance weighted
    residual corrects.
    """
    marginals = trace.marginals
    direct = (marginals * candidate_rewards).sum(axis=(1, 2))
    positions = np.arange(log.n_positions)
    logged = candidate_rewards[
        np.arange(log.n_sessions)[:, None], positions[None, :], log.slate
    ].sum(axis=1)
    return direct, logged


def dr_value(
    reward: np.ndarray,
    weight: np.ndarray,
    direct: np.ndarray,
    logged: np.ndarray,
    index: np.ndarray | None = None,
) -> float:
    """Direct model expectation plus the importance weighted residual on the served slate."""
    if index is not None:
        reward = reward[index]
        weight = weight[index]
        direct = direct[index]
        logged = logged[index]
    if reward.shape[0] == 0:
        return float("nan")
    return float(np.mean(direct + weight * (reward - logged)))
