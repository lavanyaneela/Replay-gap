"""The log as a database, in two dialects, one of which runs offline.

DuckDB is the default because it is embedded: the log is queried with real SQL and no server, so
``make verify`` holds its promise of needing no network. PostgreSQL is compiled alongside and run
by an optional CI job against a service container, so the dialect claim is not theoretical.
ADR-003 is about why both exist and what having only one would cost.
"""

from replay_gap.store.dialect import Dialect, available_dialects, list_queries, read_query
from replay_gap.store.duckdb_store import DuckDBStore
from replay_gap.store.schema import TABLES, log_to_frames

__all__ = [
    "TABLES",
    "Dialect",
    "DuckDBStore",
    "available_dialects",
    "list_queries",
    "log_to_frames",
    "read_query",
]
