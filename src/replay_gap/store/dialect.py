"""Which SQL file to run, and the rule that the two dialects must stay in step.

The queries are hand written per dialect rather than transpiled, because the differences that
matter are not syntactic. PostgreSQL truncates integer division and DuckDB does not, so the same
text produces a click through rate of zero on one engine and the right answer on the other,
with no error on either. A transpiler preserves that bug faithfully. Writing both by hand and
pinning them with a test that every query name exists in both directories is the version that
actually catches it.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from replay_gap.errors import ConfigError

SQL_ROOT_ENV = "REPLAY_GAP_SQL_ROOT"
_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "sql"


def sql_root() -> Path:
    """Locate the SQL directory, and say all three places it looked if it is missing.

    The queries live at the repository root rather than inside the package, so a non editable
    install would not carry them. Rather than silently returning no queries, this names the
    environment variable that fixes it.
    """
    override = os.environ.get(SQL_ROOT_ENV)
    candidates = [Path(override)] if override else []
    candidates += [_SOURCE_ROOT, Path.cwd() / "sql"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    looked = ", ".join(str(c) for c in candidates)
    raise ConfigError(
        f"cannot find the sql directory (looked in {looked}); set {SQL_ROOT_ENV} or run from "
        f"a checkout of the repository"
    )


class Dialect(StrEnum):
    """The SQL engines this repository compiles for."""

    DUCKDB = "duckdb"
    POSTGRES = "postgres"


def dialect_root(dialect: Dialect | str) -> Path:
    """Directory holding the query files for one dialect."""
    name = Dialect(dialect).value
    path = sql_root() / name
    if not path.is_dir():
        raise ConfigError(f"no SQL directory for dialect {name}: {path}")
    return path


def available_dialects() -> list[str]:
    """Dialect names that actually have a directory on disk."""
    root = sql_root()
    return sorted(d.value for d in Dialect if (root / d.value).is_dir())


def list_queries(dialect: Dialect | str = Dialect.DUCKDB) -> list[str]:
    """Query names, in the numeric order the filenames impose, without the extension."""
    return sorted(p.stem for p in dialect_root(dialect).glob("*.sql"))


def read_query(name: str, dialect: Dialect | str = Dialect.DUCKDB) -> str:
    """Load one query by name. Raises rather than returning empty text for a missing file."""
    path = dialect_root(dialect) / f"{name}.sql"
    if not path.exists():
        known = ", ".join(list_queries(dialect))
        raise ConfigError(f"no query named {name} for dialect {Dialect(dialect).value}: {known}")
    return path.read_text(encoding="utf-8")
