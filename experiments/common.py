"""Shared machinery for the five experiments: output paths, timing, and the scenario family.

Each experiment writes one JSON file under ``docs/experiments`` and prints one table. The JSON
is the only thing the receipts script reads, so an experiment that prints a number it did not
write to JSON is a number nothing re-measures.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from replay_gap.config import Config
from replay_gap.fixtures import Fixture, build_fixture, config_digest
from replay_gap.policies.base import Policy
from replay_gap.policies.baselines import BlendPolicy

EXPERIMENT_DIR = Path("docs/experiments")


def write_result(name: str, payload: dict, directory: Path = EXPERIMENT_DIR) -> Path:
    """Write one experiment's JSON, stamped with enough context to interpret it later."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.json"
    payload = dict(payload)
    payload["environment"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.system(),
    }
    target.write_text(json.dumps(payload, indent=2, default=_encode) + "\n", encoding="utf-8")
    return target


def _encode(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__}")


@dataclass
class Timer:
    """Wall time for the record, never for a published claim.

    Durations go in the JSON so a reader knows what the experiment cost, and never into the
    prose: a published millisecond is a fact about whichever machine ran it.
    """

    label: str
    seconds: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start


def experiment_fixture(config: Config, n_sessions: int) -> Fixture:
    """A log of the requested size from the repository's own seed and world.

    Experiments use their own session counts rather than the committed fifty thousand, because
    several of them sweep over scenarios and a fifty thousand session log per scenario would
    make the suite too slow to re-run in CI. The world and the logging policy are identical.
    """
    return build_fixture(config, n_sessions=n_sessions)


def scenario_policies(
    fixture: Fixture, n_scenarios: int, seed: int, temperature: float = 1.0
) -> list[Policy]:
    """A family of candidate rankers drawn uniformly over the three way simplex.

    A candidate ranker in this world can only be made of three things: something true, something
    inherited from the model that produced its training data, and some arbitrary direction in
    which it is wrong. Drawing the mixing weights uniformly over the simplex is the least rigged
    way to sample that space, and it is what makes E1 a measurement rather than a demonstration.
    """
    rng = np.random.default_rng(seed)
    policies: list[Policy] = []
    for index in range(n_scenarios):
        truth, incumbent, noise = rng.dirichlet(np.ones(3))
        policies.append(
            BlendPolicy(
                fixture.world,
                fixture.logging_policy,
                truth_weight=float(truth),
                incumbent_weight=float(incumbent),
                noise_weight=float(noise),
                noise_seed=10_000 + index,
                temperature=temperature,
                name=f"blend_{index:03d}",
            )
        )
    return policies


def header(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def provenance(config: Config, n_sessions: int) -> dict:
    """The context every experiment JSON carries, so a stale file is recognisable."""
    return {
        "config_digest": config_digest(config),
        "seed": config.seed,
        "n_sessions": n_sessions,
        "n_candidates": config.slate.n_candidates,
        "n_positions": config.slate.n_positions,
    }
