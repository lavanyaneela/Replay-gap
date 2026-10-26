"""Configuration loading and validation. An unnoticed typo here changes every published number."""

from __future__ import annotations

import pytest
import yaml

from replay_gap.config import Config, SlateConfig, WorldConfig, load_config
from replay_gap.errors import ConfigError


def write_config(path, payload) -> str:
    target = path / "config.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(target)


def test_the_shipped_configuration_loads_and_validates():
    config = load_config()
    assert config.slate.n_candidates > config.slate.n_positions
    assert config.log.n_sessions > 0


def test_the_number_of_ordered_slates_matches_the_falling_factorial():
    config = Config(slate=SlateConfig(n_candidates=6, n_positions=3, examination=(1.0, 0.5, 0.2)))
    assert config.n_permutations == 6 * 5 * 4


def test_a_slate_longer_than_the_candidate_set_is_rejected():
    with pytest.raises(ConfigError, match="must exceed"):
        Config(slate=SlateConfig(n_candidates=3, n_positions=3, examination=(1.0, 0.5, 0.2)))


def test_a_slate_with_no_positions_is_rejected():
    with pytest.raises(ConfigError, match="at least 1"):
        Config(slate=SlateConfig(n_candidates=5, n_positions=0, examination=()))


def test_an_examination_curve_of_the_wrong_length_is_rejected():
    with pytest.raises(ConfigError, match="examination"):
        Config(slate=SlateConfig(n_candidates=5, n_positions=3, examination=(1.0, 0.5)))


def test_an_examination_probability_outside_the_unit_interval_is_rejected():
    with pytest.raises(ConfigError, match="must lie in"):
        Config(slate=SlateConfig(n_candidates=5, n_positions=2, examination=(1.4, 0.5)))


def test_an_examination_curve_that_rises_down_the_slate_is_rejected():
    """Position bias that increases with depth would quietly invert every result."""
    with pytest.raises(ConfigError, match="non-increasing"):
        Config(slate=SlateConfig(n_candidates=5, n_positions=2, examination=(0.4, 0.9)))


def test_a_catalogue_smaller_than_the_candidate_set_is_rejected():
    with pytest.raises(ConfigError, match="n_items"):
        Config(
            world=WorldConfig(n_items=4),
            slate=SlateConfig(n_candidates=6, n_positions=2, examination=(1.0, 0.5)),
        )


def test_a_non_positive_temperature_is_rejected():
    config = load_config()
    with pytest.raises(ConfigError, match="temperature"):
        Config(logging_policy=type(config.logging_policy)(temperature=0.0))


def test_a_sharp_temperature_that_is_not_sharper_is_rejected():
    config = load_config()
    with pytest.raises(ConfigError, match="sharp_temperature"):
        Config(policies=type(config.policies)(temperature=1.0, sharp_temperature=1.0))


def test_a_clip_at_or_below_one_is_rejected_because_it_would_be_a_floor():
    config = load_config()
    with pytest.raises(ConfigError, match="clip_weight"):
        Config(estimators=type(config.estimators)(clip_weight=1.0))


def test_fewer_than_two_bootstrap_resamples_is_rejected():
    config = load_config()
    with pytest.raises(ConfigError, match="bootstrap_resamples"):
        Config(estimators=type(config.estimators)(bootstrap_resamples=1))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_ess_fraction", 0.0, "min_ess_fraction"),
        ("min_ess_fraction", 1.0, "min_ess_fraction"),
        ("max_weight_share", 0.0, "max_weight_share"),
        ("max_weight_share", 1.5, "max_weight_share"),
        ("alpha", 0.0, "alpha"),
        ("power", 1.0, "power"),
    ],
)
def test_a_diagnostic_threshold_outside_its_range_is_rejected(field, value, message):
    config = load_config()
    kwargs = dict(config.diagnostics.__dict__)
    kwargs[field] = value
    with pytest.raises(ConfigError, match=message):
        Config(diagnostics=type(config.diagnostics)(**kwargs))


def test_a_log_with_no_sessions_is_rejected():
    config = load_config()
    with pytest.raises(ConfigError, match="n_sessions"):
        Config(log=type(config.log)(n_sessions=0))


def test_a_missing_configuration_file_is_reported_by_path(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


def test_a_configuration_file_that_is_not_a_mapping_is_rejected(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        load_config(target)


def test_an_empty_configuration_file_falls_back_to_every_default(tmp_path):
    target = tmp_path / "config.yaml"
    target.write_text("", encoding="utf-8")
    assert load_config(target) == Config()


def test_an_unknown_top_level_section_is_rejected_rather_than_ignored(tmp_path):
    path = write_config(tmp_path, {"wolrd": {"n_items": 10}})
    with pytest.raises(ConfigError, match="unknown top level key"):
        load_config(path)


def test_an_unknown_key_inside_a_section_is_rejected_rather_than_ignored(tmp_path):
    """A typo that is silently ignored means the published run is not the configured run."""
    path = write_config(tmp_path, {"world": {"n_itmes": 10}})
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_a_section_that_is_not_a_mapping_is_rejected(tmp_path):
    path = write_config(tmp_path, {"world": 12})
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(path)


def test_a_boolean_where_an_integer_belongs_is_rejected(tmp_path):
    path = write_config(tmp_path, {"log": {"n_sessions": True}})
    with pytest.raises(ConfigError, match="must be an int"):
        load_config(path)


def test_a_list_in_the_file_becomes_a_tuple_so_the_config_stays_hashable(tmp_path):
    path = write_config(tmp_path, {"slate": {"examination": [1.0, 0.5, 0.2]}})
    assert load_config(path).slate.examination == (1.0, 0.5, 0.2)


def test_a_dotted_override_replaces_exactly_one_value():
    config = load_config(overrides={"log.n_sessions": 321})
    assert config.log.n_sessions == 321
    assert config.world.n_items == load_config().world.n_items


def test_an_override_naming_a_key_that_does_not_exist_is_rejected():
    with pytest.raises(ConfigError, match="does not name a key"):
        load_config(overrides={"log.sessions": 10})


def test_an_override_naming_a_section_that_does_not_exist_is_rejected():
    with pytest.raises(ConfigError, match="does not name a section"):
        load_config(overrides={"logg.n_sessions": 10})


def test_the_configuration_is_frozen_so_an_experiment_cannot_mutate_it_halfway_through():
    config = load_config()
    with pytest.raises(Exception, match="cannot assign|frozen"):
        config.seed = 1
