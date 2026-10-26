"""Produce a logged interaction stream, deterministically, from one seed.

The log carries exactly what a production log carries: which items were retrieved, which slate
was served, the propensity of that slate, and what the user did. It does not carry relevance,
and nothing downstream of this module is allowed to ask the world for it except the referee.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from replay_gap.errors import FixtureError
from replay_gap.simulator.logging_policy import LoggingPolicy
from replay_gap.simulator.user_model import World, draw_clicks

_CANDIDATE_CHUNK = 4096


@dataclass(frozen=True)
class InteractionLog:
    """One row per session. Arrays rather than a frame, because every consumer is numerical.

    ``slate`` holds indices into ``candidates``, not item ids. The conversion happens once, in
    :meth:`slate_items`, so an estimator cannot accidentally index the catalogue with a slot.
    """

    session_id: np.ndarray
    context_id: np.ndarray
    candidates: np.ndarray
    slate: np.ndarray
    log_propensity: np.ndarray
    clicks: np.ndarray
    reward: np.ndarray

    def __post_init__(self) -> None:
        n = self.session_id.shape[0]
        for name in ("context_id", "candidates", "slate", "log_propensity", "clicks", "reward"):
            array = getattr(self, name)
            if array.shape[0] != n:
                raise FixtureError(
                    f"log column {name} has {array.shape[0]} rows but session_id has {n}"
                )
        if self.slate.shape[1] != self.clicks.shape[1]:
            raise FixtureError("slate and clicks must have the same number of positions")
        if self.slate.max(initial=0) >= self.candidates.shape[1]:
            raise FixtureError("slate holds a slot outside the candidate set")

    @property
    def n_sessions(self) -> int:
        return int(self.session_id.shape[0])

    @property
    def n_candidates(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def n_positions(self) -> int:
        return int(self.slate.shape[1])

    @property
    def propensity(self) -> np.ndarray:
        """The probability the logging policy assigned to the slate it served."""
        return np.exp(self.log_propensity)

    def slate_items(self) -> np.ndarray:
        """The item ids actually shown, shape (n_sessions, n_positions)."""
        return np.take_along_axis(self.candidates, self.slate, axis=1)

    def subset(self, index: np.ndarray) -> InteractionLog:
        """A view of a subset of sessions, used by the experiments that resample the log."""
        index = np.asarray(index)
        return InteractionLog(
            session_id=self.session_id[index],
            context_id=self.context_id[index],
            candidates=self.candidates[index],
            slate=self.slate[index],
            log_propensity=self.log_propensity[index],
            clicks=self.clicks[index],
            reward=self.reward[index],
        )

    def content_hash(self) -> str:
        """A hash over every column, so a committed fixture and a regenerated one cannot drift.

        Columns are hashed in a fixed order with their dtype normalised, because a float32 column
        that becomes float64 on a round trip through Parquet is the same data and must hash the
        same way.
        """
        digest = hashlib.sha256()
        for name in (
            "session_id",
            "context_id",
            "candidates",
            "slate",
            "clicks",
        ):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(getattr(self, name), dtype=np.int64).tobytes())
        digest.update(b"log_propensity")
        digest.update(np.round(self.log_propensity, 10).astype(np.float64).tobytes())
        return digest.hexdigest()


def draw_candidate_sets(
    n_sessions: int, n_items: int, n_candidates: int, rng: np.random.Generator
) -> np.ndarray:
    """Retrieve a candidate set per session, uniformly and without replacement.

    Retrieval is deliberately out of scope: a biased retriever would add a second confounder on
    top of the one this repository is about. ADR-001 records that choice and what it costs.

    Chunked because the obvious one shot version allocates an ``n_sessions`` by ``n_items`` float
    array, which at fifty thousand sessions and a two hundred item catalogue is eighty megabytes
    held for the duration of a single argpartition.
    """
    if n_candidates > n_items:
        raise ValueError("cannot retrieve more candidates than the catalogue holds")
    out = np.empty((n_sessions, n_candidates), dtype=np.int64)
    for start in range(0, n_sessions, _CANDIDATE_CHUNK):
        stop = min(start + _CANDIDATE_CHUNK, n_sessions)
        keys = rng.random(size=(stop - start, n_items))
        picked = np.argpartition(keys, n_candidates - 1, axis=1)[:, :n_candidates]
        out[start:stop] = np.sort(picked, axis=1)
    return out


def generate_log(
    world: World,
    policy: LoggingPolicy,
    n_sessions: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> InteractionLog:
    """Run the deployed policy against the world and record what a production log would record."""
    if n_sessions < 1:
        raise ValueError("n_sessions must be at least 1")
    context_id = rng.integers(0, world.n_contexts, size=n_sessions).astype(np.int64)
    candidates = draw_candidate_sets(n_sessions, world.n_items, n_candidates, rng)
    slate, log_propensity = policy.serve(world, context_id, candidates, rng)
    shown = np.take_along_axis(candidates, slate, axis=1)
    relevance = world.relevance(context_id, shown)
    clicks = draw_clicks(relevance, world.examination, rng)
    return InteractionLog(
        session_id=np.arange(n_sessions, dtype=np.int64),
        context_id=context_id,
        candidates=candidates,
        slate=slate,
        log_propensity=log_propensity,
        clicks=clicks,
        reward=clicks.sum(axis=1).astype(np.int64),
    )
