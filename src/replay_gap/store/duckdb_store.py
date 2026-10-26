"""The embedded execution engine, which is what keeps the SQL claim and the offline rule together.

DuckDB runs in process against the frames the log becomes, so every query in ``sql/duckdb`` is a
query that actually executed when ``make verify`` ran. The PostgreSQL files next door are the
same six queries written for a server, and an optional CI job runs them against a real one. The
split is honest: this repository proves the DuckDB path on every run and the PostgreSQL path on
every run that has a container available, and ADR-003 says so rather than implying both.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType

import duckdb
import pandas as pd

from replay_gap.errors import ConfigError
from replay_gap.simulator.generate import InteractionLog
from replay_gap.store.dialect import Dialect, read_query
from replay_gap.store.schema import TABLES, log_to_frames


class DuckDBStore(AbstractContextManager):
    """An in-memory DuckDB holding one interaction log, queried through the committed SQL files."""

    def __init__(self, database: str = ":memory:") -> None:
        self._connection = duckdb.connect(database)
        self._loaded = False

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def create_schema(self) -> None:
        """Run the committed DDL, so the schema in the repository is the schema in the engine."""
        self._connection.execute(read_query("001_schema", Dialect.DUCKDB))

    def load(self, log: InteractionLog) -> None:
        """Create the schema and insert the log. Idempotent: re-loading replaces the contents."""
        self.create_schema()
        frames = log_to_frames(log)
        for table in TABLES:
            self._connection.register(f"_incoming_{table}", frames[table])
            self._connection.execute(f"DELETE FROM {table}")
            self._connection.execute(f"INSERT INTO {table} SELECT * FROM _incoming_{table}")
            self._connection.unregister(f"_incoming_{table}")
        self._loaded = True

    def run(self, name: str) -> pd.DataFrame:
        """Execute one named query from ``sql/duckdb`` and return the result as a frame."""
        if not self._loaded:
            raise ConfigError(f"query {name} requested before any log was loaded")
        return self._connection.execute(read_query(name, Dialect.DUCKDB)).df()

    def row_counts(self) -> dict[str, int]:
        """Rows per table, used by the fixture integrity check and printed by the report."""
        if not self._loaded:
            raise ConfigError("row counts requested before any log was loaded")
        return {
            table: int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in TABLES
        }
