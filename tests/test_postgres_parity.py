"""The committed PostgreSQL queries, executed against a real server.

These are the tests that stop the dialect claim from being theoretical. They are marked
``postgres`` and skipped wherever no server is reachable, which is every local run and every CI
job except the one with a service container. ADR-003 states that split rather than implying both
engines are exercised everywhere.

The assertions deliberately compare PostgreSQL's answers against DuckDB's on the same data. A
test that only checked the PostgreSQL query ran would pass on the integer division bug that
motivated writing two dialects by hand, because that bug returns zero without erroring.
"""

from __future__ import annotations

import os

import pytest

from replay_gap.store.dialect import Dialect, list_queries, read_query
from replay_gap.store.duckdb_store import DuckDBStore
from replay_gap.store.schema import TABLES, log_to_frames

pytestmark = pytest.mark.postgres

psycopg = pytest.importorskip("psycopg", reason="the postgres extra is not installed")


def connection_string() -> str:
    return (
        f"host={os.environ.get('PGHOST', 'localhost')} "
        f"port={os.environ.get('PGPORT', '5432')} "
        f"user={os.environ.get('PGUSER', 'postgres')} "
        f"password={os.environ.get('PGPASSWORD', '')} "
        f"dbname={os.environ.get('PGDATABASE', 'postgres')}"
    )


@pytest.fixture(scope="module")
def postgres(small_log):
    """A live server loaded with the same log DuckDB holds, or a skip."""
    try:
        connection = psycopg.connect(connection_string(), connect_timeout=5)
    except Exception as error:  # pragma: no cover - only on a machine with no server
        pytest.skip(f"no PostgreSQL server reachable: {error}")

    with connection:
        with connection.cursor() as cursor:
            for table in TABLES:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
            cursor.execute(read_query("001_schema", Dialect.POSTGRES))
            frames = log_to_frames(small_log)
            for table in TABLES:
                frame = frames[table]
                columns = ", ".join(frame.columns)
                with cursor.copy(f"COPY {table} ({columns}) FROM STDIN") as copy:
                    for row in frame.itertuples(index=False, name=None):
                        copy.write_row(row)
        connection.commit()
        yield connection


def fetch(connection, name: str) -> list[tuple]:
    with connection.cursor() as cursor:
        cursor.execute(read_query(name, Dialect.POSTGRES))
        return cursor.fetchall()


@pytest.fixture(scope="module")
def duck(small_log):
    with DuckDBStore() as store:
        store.load(small_log)
        yield store


def test_the_server_holds_the_same_row_counts_as_duckdb(postgres, duck):
    with postgres.cursor() as cursor:
        for table, expected in duck.row_counts().items():
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            assert cursor.fetchone()[0] == expected, table


@pytest.mark.parametrize("name", [q for q in list_queries("postgres") if q != "001_schema"])
def test_every_committed_query_executes_on_a_real_server(postgres, name):
    fetch(postgres, name)


def test_the_integrity_query_finds_nothing_wrong_on_either_engine(postgres, duck):
    assert fetch(postgres, "002_session_rewards") == []
    assert len(duck.run("002_session_rewards")) == 0


def test_click_through_rate_is_not_truncated_to_zero_by_integer_division(postgres):
    """The bug that motivated writing both dialects by hand instead of transpiling one.

    Without the numeric cast this query returns exactly 0 for every position on PostgreSQL and
    the right answer on DuckDB, with no error on either.
    """
    rows = fetch(postgres, "003_position_ctr")
    assert rows
    assert all(float(row[3]) > 0.0 for row in rows), rows


def test_click_through_rate_matches_duckdb_position_for_position(postgres, duck):
    expected = duck.run("003_position_ctr").sort_values("position")
    actual = sorted(fetch(postgres, "003_position_ctr"), key=lambda row: row[0])
    assert len(actual) == len(expected)
    for row, (_, reference) in zip(actual, expected.iterrows(), strict=True):
        assert row[0] == reference["position"]
        assert row[1] == reference["impressions"]
        assert float(row[3]) == pytest.approx(float(reference["ctr"]), rel=1e-9)


def test_the_propensity_deciles_match_duckdb(postgres, duck):
    """PostgreSQL has no MEDIAN aggregate, so this is the ordered set aggregate agreeing."""
    expected = duck.run("004_propensity_deciles").sort_values("decile")
    actual = sorted(fetch(postgres, "004_propensity_deciles"), key=lambda row: row[0])
    assert len(actual) == len(expected)
    for row, (_, reference) in zip(actual, expected.iterrows(), strict=True):
        assert row[1] == reference["sessions"]
        assert float(row[3]) == pytest.approx(float(reference["median_propensity"]), rel=1e-6)
        assert float(row[5]) == pytest.approx(float(reference["mean_reward"]), rel=1e-9)


def test_exposure_concentration_reaches_one_on_a_real_server(postgres):
    rows = fetch(postgres, "005_exposure_concentration")
    assert float(rows[-1][2]) == pytest.approx(1.0)


def test_catalogue_coverage_matches_duckdb(postgres, duck):
    actual = fetch(postgres, "006_catalogue_coverage")[0]
    expected = duck.run("006_catalogue_coverage").iloc[0]
    assert actual[0] == expected["retrieved_items"]
    assert actual[2] == expected["first_slot_items"]
