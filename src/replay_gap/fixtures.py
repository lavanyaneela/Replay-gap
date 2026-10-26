"""The committed fixture, and the hash that stops it drifting from the generator that made it.

Two artefacts are committed: the world, which carries the true relevance the referee needs, and
the log, which carries only what a production log would. Both are also regenerable from the seed
in the configuration, and a test regenerates them and asserts the content hash in the manifest
still matches.

That pairing is the point. A committed fixture alone can rot: somebody changes the generator and
the fixture keeps the old numbers alive. A generator alone means a reviewer cannot look at the
data without running code. Committing both and hashing one against the other is the version
where neither failure is possible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from replay_gap.config import Config
from replay_gap.errors import FixtureError
from replay_gap.simulator.generate import InteractionLog, generate_log
from replay_gap.simulator.logging_policy import LoggingPolicy, build_logging_policy
from replay_gap.simulator.user_model import World, build_world

FIXTURE_DIR = Path("fixtures/interactions")
WORLD_FILE = "world.npz"
LOG_FILE = "log.parquet"
MANIFEST_FILE = "manifest.json"

LOG_COLUMNS: tuple[str, ...] = (
    "session_id",
    "context_id",
    "candidate_slot",
    "candidate_item",
    "slate_slot_0",
    "slate_slot_1",
    "slate_slot_2",
    "log_propensity",
    "click_0",
    "click_1",
    "click_2",
    "reward",
)


@dataclass(frozen=True)
class Fixture:
    """A world, the policy that was deployed in it, and the log that policy produced."""

    world: World
    logging_policy: LoggingPolicy
    log: InteractionLog


def config_digest(config: Config) -> str:
    """A short hash over the configuration values the fixture depends on.

    Only the generative section is hashed. Changing the bootstrap resample count should not
    invalidate a fixture, and if it did, nobody would trust the invalidation signal.
    """
    payload = {
        "seed": config.seed,
        "world": config.world.__dict__,
        "slate": {
            "n_candidates": config.slate.n_candidates,
            "n_positions": config.slate.n_positions,
            "examination": list(config.slate.examination),
        },
        "logging_policy": config.logging_policy.__dict__,
        "log": config.log.__dict__,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_fixture(config: Config, n_sessions: int | None = None) -> Fixture:
    """Generate the world, the logging policy and the log from one seeded generator.

    One generator rather than three, drawn in a fixed order, so the whole fixture is a function
    of a single integer. Three independently seeded generators would be reproducible too and
    would quietly let somebody reseed one of them.
    """
    rng = np.random.default_rng(config.seed)
    world = build_world(
        n_items=config.world.n_items,
        n_item_features=config.world.n_item_features,
        n_segments=config.world.n_segments,
        n_contexts=config.world.n_contexts,
        relevance_scale=config.world.relevance_scale,
        quality_spread=config.world.quality_spread,
        examination=np.asarray(config.slate.examination, dtype=np.float64),
        rng=rng,
    )
    policy = build_logging_policy(
        world=world,
        item_embedding_noise=config.logging_policy.item_embedding_noise,
        quality_noise=config.logging_policy.quality_noise,
        temperature=config.logging_policy.temperature,
        rng=rng,
    )
    log = generate_log(
        world=world,
        policy=policy,
        n_sessions=n_sessions if n_sessions is not None else config.log.n_sessions,
        n_candidates=config.slate.n_candidates,
        rng=rng,
    )
    return Fixture(world=world, logging_policy=policy, log=log)


def log_to_frame(log: InteractionLog) -> pd.DataFrame:
    """The log as one long frame, which is the shape Parquet stores compactly.

    One row per retrieved candidate, with the session level columns repeated. Wider than the
    arrays and considerably smaller on disk, because the repeated columns compress to almost
    nothing and the integer columns dictionary encode.
    """
    n_sessions, n_candidates = log.candidates.shape
    frame = pd.DataFrame(
        {
            "session_id": np.repeat(log.session_id, n_candidates).astype(np.int32),
            "context_id": np.repeat(log.context_id, n_candidates).astype(np.int32),
            "candidate_slot": np.tile(np.arange(n_candidates), n_sessions).astype(np.int16),
            "candidate_item": log.candidates.reshape(-1).astype(np.int16),
        }
    )
    for position in range(log.n_positions):
        frame[f"slate_slot_{position}"] = np.repeat(log.slate[:, position], n_candidates).astype(
            np.int16
        )
        frame[f"click_{position}"] = np.repeat(log.clicks[:, position], n_candidates).astype(
            np.int8
        )
    frame["log_propensity"] = np.repeat(log.log_propensity, n_candidates)
    frame["reward"] = np.repeat(log.reward, n_candidates).astype(np.int8)
    return frame


def frame_to_log(frame: pd.DataFrame, n_positions: int) -> InteractionLog:
    """Rebuild the array log from the stored frame."""
    required = {"session_id", "context_id", "candidate_slot", "candidate_item", "log_propensity"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FixtureError(f"stored log is missing column(s): {', '.join(missing)}")
    n_candidates = int(frame["candidate_slot"].max()) + 1
    if len(frame) % n_candidates != 0:
        raise FixtureError("stored log row count is not a multiple of the candidate set size")
    n_sessions = len(frame) // n_candidates
    candidates = frame["candidate_item"].to_numpy().reshape(n_sessions, n_candidates)
    first = frame["candidate_slot"].to_numpy().reshape(n_sessions, n_candidates)[:, 0]
    if np.any(first != 0):
        raise FixtureError("stored log rows are not grouped by session in candidate slot order")
    head = slice(None, None, n_candidates)
    slate = np.stack(
        [frame[f"slate_slot_{k}"].to_numpy()[head] for k in range(n_positions)], axis=1
    )
    clicks = np.stack([frame[f"click_{k}"].to_numpy()[head] for k in range(n_positions)], axis=1)
    return InteractionLog(
        session_id=frame["session_id"].to_numpy()[head].astype(np.int64),
        context_id=frame["context_id"].to_numpy()[head].astype(np.int64),
        candidates=candidates.astype(np.int64),
        slate=slate.astype(np.int64),
        log_propensity=frame["log_propensity"].to_numpy()[head].astype(np.float64),
        clicks=clicks.astype(np.int8),
        reward=frame["reward"].to_numpy()[head].astype(np.int64),
    )


def write_fixture(fixture: Fixture, config: Config, directory: Path | str = FIXTURE_DIR) -> dict:
    """Write the world, the log and the manifest. Returns the manifest."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path / WORLD_FILE,
        item_features=fixture.world.item_features,
        item_quality=fixture.world.item_quality,
        context_features=fixture.world.context_features,
        context_segment=fixture.world.context_segment,
        examination=fixture.world.examination,
        relevance_scale=np.array([fixture.world.relevance_scale]),
        policy_item_features=fixture.logging_policy.item_features,
        policy_item_quality=fixture.logging_policy.item_quality,
        policy_temperature=np.array([fixture.logging_policy.temperature]),
    )
    frame = log_to_frame(fixture.log)
    frame.to_parquet(path / LOG_FILE, index=False, compression="zstd")
    manifest = {
        "seed": config.seed,
        "config_digest": config_digest(config),
        "n_sessions": fixture.log.n_sessions,
        "n_candidates": fixture.log.n_candidates,
        "n_positions": fixture.log.n_positions,
        "n_items": fixture.world.n_items,
        "n_contexts": fixture.world.n_contexts,
        "log_content_hash": fixture.log.content_hash(),
        "log_rows": int(len(frame)),
        "columns": list(LOG_COLUMNS),
        "note": (
            "Regenerated by 'make fixtures'. The log content hash is re-derived from the seed "
            "by the test suite, so the committed file and the generator cannot drift apart."
        ),
    }
    (path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def read_manifest(directory: Path | str = FIXTURE_DIR) -> dict:
    """Load the manifest, or say clearly that the fixture has not been generated."""
    path = Path(directory) / MANIFEST_FILE
    if not path.exists():
        raise FixtureError(f"no fixture manifest at {path}; run 'make fixtures' to create one")
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixture(config: Config, directory: Path | str = FIXTURE_DIR) -> Fixture:
    """Load the committed fixture and refuse it if it does not match the configuration."""
    path = Path(directory)
    manifest = read_manifest(path)
    if manifest.get("config_digest") != config_digest(config):
        raise FixtureError(
            "the committed fixture was generated from a different configuration "
            f"(manifest {manifest.get('config_digest')}, current {config_digest(config)}); "
            "run 'make fixtures' to regenerate it"
        )
    world_path = path / WORLD_FILE
    log_path = path / LOG_FILE
    for required in (world_path, log_path):
        if not required.exists():
            raise FixtureError(f"fixture file missing: {required}")

    with np.load(world_path) as data:
        world = World(
            item_features=data["item_features"],
            item_quality=data["item_quality"],
            context_features=data["context_features"],
            context_segment=data["context_segment"],
            examination=data["examination"],
            relevance_scale=float(data["relevance_scale"][0]),
        )
        policy = LoggingPolicy(
            item_features=data["policy_item_features"],
            item_quality=data["policy_item_quality"],
            relevance_scale=world.relevance_scale,
            temperature=float(data["policy_temperature"][0]),
        )
    log = frame_to_log(pd.read_parquet(log_path), int(manifest["n_positions"]))
    actual = log.content_hash()
    if actual != manifest["log_content_hash"]:
        raise FixtureError(
            f"the committed log does not match its manifest hash (expected "
            f"{manifest['log_content_hash']}, found {actual})"
        )
    return Fixture(world=world, logging_policy=policy, log=log)


def load_or_build(config: Config, directory: Path | str = FIXTURE_DIR) -> Fixture:
    """Prefer the committed fixture, and generate one only when none has been written yet.

    A missing fixture is a fresh checkout that has not run ``make fixtures``, which is a state
    worth recovering from silently. A fixture whose hash or configuration digest does not match
    is a different thing entirely, and swallowing that would defeat the reason the hash exists,
    so it propagates.
    """
    if not (Path(directory) / MANIFEST_FILE).exists():
        return build_fixture(config)
    return load_fixture(config, directory)
