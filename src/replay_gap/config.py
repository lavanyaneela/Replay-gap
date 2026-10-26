"""Typed configuration, loaded from YAML, validated once at the boundary.

The reason this file exists rather than a dict passed around: every published number in the
repository is a function of these values, so they have to be in one place a reviewer can read,
and a typo in a nested key has to fail loudly at load time rather than silently producing a
different experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from replay_gap.errors import ConfigError

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(frozen=True)
class WorldConfig:
    """The latent world: a catalogue, a set of users, and the true relevance that links them."""

    n_items: int = 200
    n_item_features: int = 8
    n_segments: int = 6
    n_contexts: int = 5000
    relevance_scale: float = 2.4
    quality_spread: float = 0.55


@dataclass(frozen=True)
class SlateConfig:
    """The action space. Small on purpose: 720 ordered slates can be enumerated exactly."""

    n_candidates: int = 10
    n_positions: int = 3
    examination: tuple[float, ...] = (1.0, 0.55, 0.32)


@dataclass(frozen=True)
class LoggingPolicyConfig:
    """How far the deployed ranker's view of the world is from the truth, and how random it is."""

    item_embedding_noise: float = 0.75
    quality_noise: float = 0.45
    temperature: float = 1.0


@dataclass(frozen=True)
class LogConfig:
    n_sessions: int = 50000


@dataclass(frozen=True)
class PoliciesConfig:
    temperature: float = 1.0
    sharp_temperature: float = 0.25


@dataclass(frozen=True)
class GBDTConfig:
    """Shared shape for the two gradient boosted models, which differ only in their numbers."""

    n_estimators: int = 160
    learning_rate: float = 0.08
    num_leaves: int = 15
    min_child_samples: int = 40
    seed: int = 11


@dataclass(frozen=True)
class EstimatorsConfig:
    clip_weight: float = 20.0
    bootstrap_resamples: int = 400
    bootstrap_seed: int = 7


@dataclass(frozen=True)
class DiagnosticsConfig:
    """The thresholds that decide whether the tool answers or refuses."""

    min_ess_fraction: float = 0.02
    max_weight_share: float = 0.05
    alpha: float = 0.05
    power: float = 0.8


@dataclass(frozen=True)
class ExperimentsConfig:
    e1_scenarios: int = 180
    e1_sessions: int = 8000
    e1_decisive_lift_pct: float = 1.0
    e2_divergence_steps: int = 12
    e3_replicates: int = 24
    e3_sessions: int = 8000
    e5_factors: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
    e5_sessions: int = 8000


@dataclass(frozen=True)
class Config:
    """The whole configuration. Frozen, so an experiment cannot mutate it halfway through."""

    seed: int = 20260301
    world: WorldConfig = field(default_factory=WorldConfig)
    slate: SlateConfig = field(default_factory=SlateConfig)
    logging_policy: LoggingPolicyConfig = field(default_factory=LoggingPolicyConfig)
    log: LogConfig = field(default_factory=LogConfig)
    policies: PoliciesConfig = field(default_factory=PoliciesConfig)
    ranker: GBDTConfig = field(default_factory=GBDTConfig)
    reward_model: GBDTConfig = field(default_factory=lambda: GBDTConfig(seed=12))
    estimators: EstimatorsConfig = field(default_factory=EstimatorsConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Refuse a configuration the rest of the code would silently misinterpret."""
        s = self.slate
        if s.n_positions < 1:
            raise ConfigError("slate.n_positions must be at least 1")
        if s.n_candidates <= s.n_positions:
            raise ConfigError(
                f"slate.n_candidates ({s.n_candidates}) must exceed slate.n_positions "
                f"({s.n_positions}), or there is no ranking decision to make"
            )
        if len(s.examination) != s.n_positions:
            raise ConfigError(
                f"slate.examination has {len(s.examination)} entries but slate.n_positions is "
                f"{s.n_positions}"
            )
        if any(not 0.0 < p <= 1.0 for p in s.examination):
            raise ConfigError("every slate.examination entry must lie in (0, 1]")
        if list(s.examination) != sorted(s.examination, reverse=True):
            raise ConfigError("slate.examination must be non-increasing down the slate")
        if self.world.n_items < s.n_candidates:
            raise ConfigError("world.n_items must be at least slate.n_candidates")
        temperatures = (
            self.logging_policy.temperature,
            self.policies.temperature,
            self.policies.sharp_temperature,
        )
        if any(t <= 0 for t in temperatures):
            raise ConfigError("a Plackett-Luce temperature must be strictly positive")
        if self.policies.sharp_temperature >= self.policies.temperature:
            raise ConfigError(
                "policies.sharp_temperature must be below policies.temperature, or the sharp "
                "variant is not sharper"
            )
        if self.estimators.clip_weight <= 1.0:
            raise ConfigError(
                "estimators.clip_weight must exceed 1.0 to be a cap rather than a floor"
            )
        if self.estimators.bootstrap_resamples < 2:
            raise ConfigError("estimators.bootstrap_resamples must be at least 2")
        d = self.diagnostics
        if not 0.0 < d.min_ess_fraction < 1.0:
            raise ConfigError("diagnostics.min_ess_fraction must lie strictly between 0 and 1")
        if not 0.0 < d.max_weight_share <= 1.0:
            raise ConfigError("diagnostics.max_weight_share must lie in (0, 1]")
        if not 0.0 < d.alpha < 1.0:
            raise ConfigError("diagnostics.alpha must lie strictly between 0 and 1")
        if not 0.0 < d.power < 1.0:
            raise ConfigError("diagnostics.power must lie strictly between 0 and 1")
        if self.log.n_sessions < 1:
            raise ConfigError("log.n_sessions must be at least 1")

    @property
    def n_permutations(self) -> int:
        """How many ordered slates exist. The whole exact-ground-truth design rests on this
        staying small enough to enumerate."""
        total = 1
        for k in range(self.slate.n_positions):
            total *= self.slate.n_candidates - k
        return total


def _coerce(cls: type, raw: Any, path: str) -> Any:
    """Build one dataclass from a plain mapping, rejecting unknown and mistyped keys.

    An unknown key is almost always a typo in the YAML, and silently ignoring it means the
    experiment that gets published is not the experiment that was configured.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping, got {type(raw).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise ConfigError(f"unknown key(s) under {path}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        target = known[name].type
        if isinstance(value, list):
            value = tuple(value)
        if target in ("int", int) and isinstance(value, bool):
            raise ConfigError(f"{path}.{name} must be an int, got a bool")
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load the YAML configuration, apply dotted overrides, and validate the result.

    Overrides use dotted keys ("log.n_sessions") because that is what a CLI flag can carry and it
    keeps the override surface identical to the file's own structure.
    """
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not source.exists():
        raise ConfigError(f"configuration file not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must contain a mapping at the top level")

    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".")
        cursor: dict[str, Any] = raw
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {dotted} does not name a section in {source.name}")
            cursor = nxt
        if parts[-1] not in cursor:
            raise ConfigError(f"override {dotted} does not name a key in {source.name}")
        cursor[parts[-1]] = value

    top = {f.name: f for f in fields(Config)}
    unknown = sorted(set(raw) - set(top))
    if unknown:
        raise ConfigError(f"unknown top level key(s): {', '.join(unknown)}")

    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        declared = Config.__dataclass_fields__[name].default_factory  # type: ignore[union-attr]
        if callable(declared) and is_dataclass(declared()):
            kwargs[name] = _coerce(type(declared()), value, name)
        else:
            kwargs[name] = value
    return Config(**kwargs)
