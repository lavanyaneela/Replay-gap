"""The SQL layer, in both dialects.

The DuckDB queries execute here. The PostgreSQL queries cannot, because there is no server and
``make verify`` must pass with no network, so they are checked structurally instead: both
dialects parse, both expose the same query names, and both produce the same output columns.
An optional CI job runs the PostgreSQL files against a real server. ADR-003 states that split
rather than implying both are exercised everywhere.
"""

from __future__ import annotations

import re

import pytest
import sqlglot

from replay_gap.errors import ConfigError
from replay_gap.store.dialect import (
    SQL_ROOT_ENV,
    Dialect,
    available_dialects,
    dialect_root,
    list_queries,
    read_query,
    sql_root,
)
from replay_gap.store.duckdb_store import DuckDBStore
from replay_gap.store.schema import TABLES, log_to_frames


@pytest.fixture(scope="module")
def store(small_log):
    with DuckDBStore() as db:
        db.load(small_log)
        yield db


def test_both_dialects_are_present_on_disk():
    assert available_dialects() == ["duckdb", "postgres"]


def test_the_two_dialects_expose_exactly_the_same_query_names():
    """A query that exists in one dialect and not the other is a claim that quietly rots."""
    assert list_queries(Dialect.DUCKDB) == list_queries(Dialect.POSTGRES)


@pytest.mark.parametrize("dialect", ["duckdb", "postgres"])
def test_every_query_parses_under_its_own_dialect(dialect):
    for name in list_queries(dialect):
        sqlglot.parse(read_query(name, dialect), read=dialect)


@pytest.mark.parametrize("name", [q for q in list_queries("duckdb") if q != "001_schema"])
def test_the_two_dialects_select_the_same_output_columns(name):
    """The engines differ; the contract the Python side reads must not."""
    columns = {}
    for dialect in ("duckdb", "postgres"):
        statement = sqlglot.parse_one(read_query(name, dialect), read=dialect)
        columns[dialect] = [expression.alias_or_name for expression in statement.selects]
    assert columns["duckdb"] == columns["postgres"]


def test_the_two_schemas_declare_the_same_tables_and_columns():
    declared = {}
    for dialect in ("duckdb", "postgres"):
        text = read_query("001_schema", dialect)
        declared[dialect] = sorted(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", text))
    assert declared["duckdb"] == declared["postgres"] == sorted(TABLES)


def test_the_postgres_rate_queries_cast_away_integer_division():
    """PostgreSQL truncates integer division and DuckDB does not, so the same text gives a click
    through rate of zero on one engine and the right answer on the other, silently."""
    text = read_query("003_position_ctr", "postgres")
    assert "::numeric" in text
    assert "::numeric" not in read_query("003_position_ctr", "duckdb")


def test_the_postgres_median_uses_the_ordered_set_aggregate():
    assert "PERCENTILE_CONT" in read_query("004_propensity_deciles", "postgres")
    assert "MEDIAN" in read_query("004_propensity_deciles", "duckdb")


def test_an_unknown_query_name_lists_the_ones_that_exist():
    with pytest.raises(ConfigError, match="no query named"):
        read_query("999_absent", "duckdb")


def test_a_dialect_with_no_directory_on_disk_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv(SQL_ROOT_ENV, str(tmp_path))
    (tmp_path / "duckdb").mkdir()
    with pytest.raises(ConfigError, match="no SQL directory for dialect postgres"):
        dialect_root("postgres")


def test_an_unknown_dialect_is_rejected():
    with pytest.raises(ValueError):
        dialect_root("mysql")


def test_the_sql_root_can_be_overridden_by_an_environment_variable(tmp_path, monkeypatch):
    (tmp_path / "duckdb").mkdir()
    monkeypatch.setenv(SQL_ROOT_ENV, str(tmp_path))
    assert sql_root() == tmp_path


def test_a_missing_sql_root_names_every_place_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv(SQL_ROOT_ENV, str(tmp_path / "absent"))
    monkeypatch.setattr("replay_gap.store.dialect._SOURCE_ROOT", tmp_path / "also-absent")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="cannot find the sql directory"):
        sql_root()


def test_the_log_becomes_three_tables_at_three_different_grains(small_log):
    frames = log_to_frames(small_log)
    assert len(frames["sessions"]) == small_log.n_sessions
    assert len(frames["candidates"]) == small_log.n_sessions * small_log.n_candidates
    assert len(frames["impressions"]) == small_log.n_sessions * small_log.n_positions


def test_the_impressions_table_holds_the_items_that_were_actually_shown(small_log):
    frames = log_to_frames(small_log)
    shown = small_log.slate_items()
    assert frames["impressions"]["item_id"].to_numpy().tolist() == shown.reshape(-1).tolist()


def test_the_store_reports_the_row_count_of_every_table(store, small_log):
    counts = store.row_counts()
    assert counts["sessions"] == small_log.n_sessions
    assert counts["impressions"] == small_log.n_sessions * small_log.n_positions


def test_the_reward_column_agrees_with_the_impressions_it_was_summed_from(store):
    """A committed fixture nobody cross checks is a committed fixture nobody should trust."""
    assert len(store.run("002_session_rewards")) == 0


def test_click_through_rate_falls_down_the_slate(store):
    frame = store.run("003_position_ctr").sort_values("position")
    rates = frame["ctr"].tolist()
    assert rates == sorted(rates, reverse=True)


def test_the_position_shares_of_clicks_sum_to_one(store):
    assert store.run("003_position_ctr")["share_of_clicks"].sum() == pytest.approx(1.0)


def test_reward_rises_with_the_logging_policys_own_confidence(store):
    """The confounder in one table: a replay keeps the top deciles, and the top deciles pay more."""
    frame = store.run("004_propensity_deciles").sort_values("decile")
    assert frame["mean_reward"].iloc[-1] > frame["mean_reward"].iloc[0]


def test_the_propensity_deciles_split_the_log_evenly(store, small_log):
    frame = store.run("004_propensity_deciles")
    assert frame["sessions"].sum() == small_log.n_sessions
    assert frame["sessions"].max() - frame["sessions"].min() <= 1


def test_exposure_concentration_reaches_a_cumulative_share_of_one(store):
    frame = store.run("005_exposure_concentration")
    assert frame["cumulative_share"].iloc[-1] == pytest.approx(1.0)
    assert frame["cumulative_share"].is_monotonic_increasing


def test_catalogue_coverage_never_exceeds_the_catalogue(store, small_world):
    row = store.run("006_catalogue_coverage").iloc[0]
    assert row["first_slot_items"] <= row["impressed_items"] <= row["retrieved_items"]
    assert row["retrieved_items"] <= small_world.n_items


def test_a_query_before_any_log_is_loaded_is_refused():
    with DuckDBStore() as db:
        with pytest.raises(ConfigError, match="before any log was loaded"):
            db.run("003_position_ctr")
        with pytest.raises(ConfigError, match="before any log was loaded"):
            db.row_counts()


def test_loading_twice_replaces_the_contents_rather_than_appending(small_log):
    with DuckDBStore() as db:
        db.load(small_log)
        first = db.row_counts()
        db.load(small_log)
        assert db.row_counts() == first
        assert db.loaded


def test_the_store_closes_cleanly_as_a_context_manager(small_log):
    with DuckDBStore() as db:
        db.load(small_log)
    assert db is not None
