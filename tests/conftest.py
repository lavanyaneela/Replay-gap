"""Shared fixtures. Small by default, because a slow suite is a suite nobody runs.

The small world is not a toy version of the real one: it is the same generator with smaller
numbers, so a property that holds here holds there. Two tests deliberately use the committed
full size fixture, and they are the ones about the committed fixture itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from replay_gap.config import (
    Config,
    DiagnosticsConfig,
    EstimatorsConfig,
    ExperimentsConfig,
    GBDTConfig,
    LogConfig,
    LoggingPolicyConfig,
    PoliciesConfig,
    SlateConfig,
    WorldConfig,
    load_config,
)
from replay_gap.estimators.doubly_robust import fit_reward_model, predict_candidate_rewards
from replay_gap.evaluation import trace_policy
from replay_gap.fixtures import build_fixture, load_fixture
from replay_gap.policies.baselines import ContrarianPolicy, IncumbentPolicy, OraclePolicy
from replay_gap.ranking import permutation_index


@pytest.fixture(scope="session")
def small_config() -> Config:
    """A world small enough that the whole suite runs in seconds and large enough to be real."""
    return Config(
        seed=4242,
        world=WorldConfig(
            n_items=40,
            n_item_features=6,
            n_segments=3,
            n_contexts=250,
            relevance_scale=2.4,
            quality_spread=0.55,
        ),
        slate=SlateConfig(n_candidates=6, n_positions=3, examination=(1.0, 0.55, 0.32)),
        logging_policy=LoggingPolicyConfig(
            item_embedding_noise=0.75, quality_noise=0.45, temperature=1.0
        ),
        log=LogConfig(n_sessions=1200),
        policies=PoliciesConfig(temperature=1.0, sharp_temperature=0.25),
        ranker=GBDTConfig(n_estimators=20, num_leaves=7, min_child_samples=20, seed=11),
        reward_model=GBDTConfig(n_estimators=20, num_leaves=7, min_child_samples=20, seed=12),
        estimators=EstimatorsConfig(clip_weight=20.0, bootstrap_resamples=40, bootstrap_seed=7),
        diagnostics=DiagnosticsConfig(
            min_ess_fraction=0.02, max_weight_share=0.05, alpha=0.05, power=0.8
        ),
        experiments=ExperimentsConfig(
            e1_scenarios=6,
            e1_sessions=400,
            e2_divergence_steps=3,
            e3_replicates=2,
            e3_sessions=400,
            e5_factors=(1.0, 2.0),
            e5_sessions=400,
        ),
    )


@pytest.fixture(scope="session")
def small_fixture(small_config: Config):
    """World, logging policy and log, generated once for the whole session."""
    return build_fixture(small_config)


@pytest.fixture(scope="session")
def small_world(small_fixture):
    return small_fixture.world


@pytest.fixture(scope="session")
def small_log(small_fixture):
    return small_fixture.log


@pytest.fixture(scope="session")
def small_perms(small_config: Config) -> np.ndarray:
    return permutation_index(small_config.slate.n_candidates, small_config.slate.n_positions)


@pytest.fixture(scope="session")
def incumbent_policy(small_fixture) -> IncumbentPolicy:
    return IncumbentPolicy(small_fixture.world, small_fixture.logging_policy)


@pytest.fixture(scope="session")
def oracle_policy(small_fixture) -> OraclePolicy:
    return OraclePolicy(small_fixture.world)


@pytest.fixture(scope="session")
def contrarian_policy(small_fixture) -> ContrarianPolicy:
    return ContrarianPolicy(small_fixture.world, small_fixture.logging_policy)


@pytest.fixture(scope="session")
def oracle_trace(small_fixture, oracle_policy, small_perms):
    return trace_policy(oracle_policy, small_fixture.log, small_perms)


@pytest.fixture(scope="session")
def incumbent_trace(small_fixture, incumbent_policy, small_perms):
    return trace_policy(incumbent_policy, small_fixture.log, small_perms)


@pytest.fixture(scope="session")
def candidate_rewards(small_fixture, small_config):
    """Outcome model predictions, fitted once because LightGBM is the slow part of the suite."""
    model = fit_reward_model(small_fixture.world, small_fixture.log, small_config.reward_model)
    return predict_candidate_rewards(small_fixture.world, small_fixture.log, model)


@pytest.fixture(scope="session")
def repository_config() -> Config:
    """The configuration the repository actually publishes numbers from."""
    return load_config()


@pytest.fixture(scope="session")
def committed_fixture(repository_config):
    """The committed log, loaded from disk, hash checked on the way in."""
    return load_fixture(repository_config)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)
