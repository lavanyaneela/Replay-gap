"""The committed fixture, and the guarantee that it still matches the generator that wrote it.

Two of these tests read the real 50,000 session fixture from disk. They are the reason the
repository can claim that a reviewer who regenerates the log gets the same log, rather than
claiming it in a sentence.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from replay_gap.errors import FixtureError
from replay_gap.fixtures import (
    MANIFEST_FILE,
    build_fixture,
    config_digest,
    frame_to_log,
    load_fixture,
    load_or_build,
    log_to_frame,
    read_manifest,
    write_fixture,
)


def test_the_committed_log_matches_the_hash_in_its_manifest(committed_fixture):
    """Loading already checks this; asserting it separately is what makes the check visible."""
    manifest = read_manifest()
    assert committed_fixture.log.content_hash() == manifest["log_content_hash"]


def test_regenerating_the_committed_log_from_its_seed_reproduces_it_exactly(
    repository_config, committed_fixture
):
    """The committed fixture and the generator are the same artefact, verified rather than
    asserted. This is the test that stops a committed file from quietly outliving its code."""
    regenerated = build_fixture(repository_config)
    assert regenerated.log.content_hash() == committed_fixture.log.content_hash()
    assert np.allclose(regenerated.world.item_features, committed_fixture.world.item_features)
    assert np.array_equal(regenerated.log.reward, committed_fixture.log.reward)


def test_the_committed_manifest_describes_the_log_it_sits_next_to(committed_fixture):
    manifest = read_manifest()
    assert manifest["n_sessions"] == committed_fixture.log.n_sessions
    assert manifest["n_candidates"] == committed_fixture.log.n_candidates
    assert manifest["n_positions"] == committed_fixture.log.n_positions
    assert manifest["log_rows"] == (
        committed_fixture.log.n_sessions * committed_fixture.log.n_candidates
    )


def test_the_configuration_digest_covers_the_generative_settings(repository_config):
    changed = replace(
        repository_config,
        world=replace(repository_config.world, n_items=repository_config.world.n_items + 1),
    )
    assert config_digest(changed) != config_digest(repository_config)


def test_the_configuration_digest_ignores_settings_the_fixture_does_not_depend_on(
    repository_config,
):
    """Changing the bootstrap resample count must not invalidate a fixture, or nobody will
    believe the invalidation signal when it fires for a real reason."""
    changed = replace(
        repository_config,
        estimators=replace(repository_config.estimators, bootstrap_resamples=11),
    )
    assert config_digest(changed) == config_digest(repository_config)


def test_a_log_survives_a_round_trip_through_the_stored_frame(small_log):
    restored = frame_to_log(log_to_frame(small_log), small_log.n_positions)
    assert restored.content_hash() == small_log.content_hash()
    assert np.array_equal(restored.candidates, small_log.candidates)
    assert np.array_equal(restored.clicks, small_log.clicks)


def test_the_content_hash_survives_a_change_of_dtype(small_log):
    """A column that comes back from Parquet as a wider integer is the same data."""
    widened = replace(small_log, candidates=small_log.candidates.astype(np.int32))
    assert widened.content_hash() == small_log.content_hash()


def test_the_content_hash_changes_when_a_single_click_changes(small_log):
    clicks = small_log.clicks.copy()
    clicks[0, 0] = 1 - clicks[0, 0]
    assert replace(small_log, clicks=clicks).content_hash() != small_log.content_hash()


def test_writing_and_loading_a_fixture_round_trips(tmp_path, small_config):
    fixture = build_fixture(small_config)
    manifest = write_fixture(fixture, small_config, tmp_path)
    loaded = load_fixture(small_config, tmp_path)
    assert manifest["log_content_hash"] == loaded.log.content_hash()
    assert loaded.logging_policy.temperature == fixture.logging_policy.temperature
    assert np.allclose(loaded.world.item_quality, fixture.world.item_quality)


def test_loading_a_fixture_written_under_a_different_configuration_is_refused(
    tmp_path, small_config
):
    write_fixture(build_fixture(small_config), small_config, tmp_path)
    changed = replace(small_config, seed=small_config.seed + 1)
    with pytest.raises(FixtureError, match="different configuration"):
        load_fixture(changed, tmp_path)


def test_loading_a_fixture_whose_log_has_been_edited_is_refused(tmp_path, small_config):
    """The hash is the point: an edited committed file must not pass silently."""
    import pandas as pd

    write_fixture(build_fixture(small_config), small_config, tmp_path)
    frame = pd.read_parquet(tmp_path / "log.parquet")
    frame.loc[0, "click_0"] = 1 - int(frame.loc[0, "click_0"])
    frame.to_parquet(tmp_path / "log.parquet", index=False)
    with pytest.raises(FixtureError, match="does not match its manifest hash"):
        load_fixture(small_config, tmp_path)


def test_loading_from_a_directory_with_no_manifest_is_refused(tmp_path, small_config):
    with pytest.raises(FixtureError, match="no fixture manifest"):
        load_fixture(small_config, tmp_path)


def test_loading_with_a_manifest_but_no_log_names_the_missing_file(tmp_path, small_config):
    write_fixture(build_fixture(small_config), small_config, tmp_path)
    (tmp_path / "log.parquet").unlink()
    with pytest.raises(FixtureError, match="fixture file missing"):
        load_fixture(small_config, tmp_path)


def test_a_fresh_checkout_with_no_fixture_generates_one(tmp_path, small_config):
    fixture = load_or_build(small_config, tmp_path)
    assert fixture.log.n_sessions == small_config.log.n_sessions


def test_a_corrupt_fixture_is_not_silently_regenerated(tmp_path, small_config):
    """Falling back to generation here would defeat the reason the hash exists."""
    write_fixture(build_fixture(small_config), small_config, tmp_path)
    changed = replace(small_config, seed=small_config.seed + 1)
    with pytest.raises(FixtureError):
        load_or_build(changed, tmp_path)


def test_a_stored_frame_missing_a_column_is_refused(small_log):
    frame = log_to_frame(small_log).drop(columns=["log_propensity"])
    with pytest.raises(FixtureError, match="missing column"):
        frame_to_log(frame, small_log.n_positions)


def test_a_stored_frame_with_a_truncated_session_is_refused(small_log):
    frame = log_to_frame(small_log).iloc[:-1]
    with pytest.raises(FixtureError, match="not a multiple"):
        frame_to_log(frame, small_log.n_positions)


def test_a_stored_frame_whose_rows_are_shuffled_is_refused(small_log):
    frame = log_to_frame(small_log).sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(FixtureError, match="not grouped by session"):
        frame_to_log(frame, small_log.n_positions)


def test_generating_a_fixture_with_an_overridden_session_count(small_config):
    fixture = build_fixture(small_config, n_sessions=77)
    assert fixture.log.n_sessions == 77


def test_the_manifest_file_is_named_consistently(tmp_path, small_config):
    write_fixture(build_fixture(small_config), small_config, tmp_path)
    assert (tmp_path / MANIFEST_FILE).exists()
