"""The session bootstrap, written around the one thing all four estimators have in common.

Every estimator here is a ratio of two session level sums. IPS divides a weighted reward sum by
the session count, SNIPS divides it by the weight sum, naive replay divides a masked reward sum
by the mask's own sum, and the doubly robust estimator divides a corrected sum by the session
count. Writing that structure down explicitly buys three things.

The bootstrap becomes one matrix product. Resampling sessions with replacement is the same as
drawing a count per session, so the whole resample is a multiply of a (resamples, sessions)
count matrix against the handful of per session vectors the estimators are built from. On a
fifty thousand session log with four hundred resamples that replaced several hundred million
fancy indexing gathers with one call into BLAS.

The draws are shared, so the four intervals differ only by the estimators and not by bootstrap
noise, which is what E3 needs in order to compare them.

And the lift is paired: the candidate arm and the baseline arm are evaluated on the same
resampled sessions. Resampling them independently inflates the interval and holding the baseline
fixed shrinks it, and the second mistake is the one that flatters a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_COUNT_CHUNK = 64


@dataclass(frozen=True)
class RatioEstimator:
    """An estimator expressed as the ratio of two per session sums."""

    name: str
    numerator: np.ndarray
    denominator: np.ndarray

    def __post_init__(self) -> None:
        if self.numerator.shape != self.denominator.shape:
            raise ValueError(
                f"{self.name}: numerator shape {self.numerator.shape} does not match "
                f"denominator shape {self.denominator.shape}"
            )
        if self.numerator.ndim != 1:
            raise ValueError(f"{self.name}: estimator terms must be one dimensional")

    @property
    def n_sessions(self) -> int:
        return int(self.numerator.shape[0])

    def value(self, index: np.ndarray | None = None) -> float:
        """The point estimate, optionally over a subset of sessions."""
        numerator = self.numerator if index is None else self.numerator[index]
        denominator = self.denominator if index is None else self.denominator[index]
        total = float(denominator.sum())
        if total == 0.0:
            return float("nan")
        return float(numerator.sum() / total)


@dataclass(frozen=True)
class BootstrapInterval:
    """A point estimate with a percentile interval and the spread behind it."""

    point: float
    low: float
    high: float
    standard_error: float
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        """True when the whole interval sits on one side of zero."""
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            return False
        return (self.low > 0.0 and self.high > 0.0) or (self.low < 0.0 and self.high < 0.0)


def _count_matrix(n_sessions: int, resamples: int, rng: np.random.Generator) -> np.ndarray:
    """How many times each session appears in each resample, shape (resamples, sessions)."""
    counts = np.empty((resamples, n_sessions), dtype=np.float64)
    for r in range(resamples):
        draw = rng.integers(0, n_sessions, size=n_sessions)
        counts[r] = np.bincount(draw, minlength=n_sessions)
    return counts


def bootstrap_ratio_lifts(
    estimators: dict[str, RatioEstimator],
    baselines: dict[str, RatioEstimator],
    resamples: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, BootstrapInterval]:
    """Percentile intervals for ``estimator / baseline - 1``, over one shared set of resamples.

    Each estimator may have its own baseline, because they are not all denominated in the same
    thing: a replay that scores only the first slot has to be compared against the log's first
    slot click rate, not against its mean session reward. Dividing by the wrong baseline is a
    mistake that produces a plausible looking number, which is the worst kind.
    """
    if not estimators:
        raise ValueError("at least one estimator is required")
    if resamples < 2:
        raise ValueError("resamples must be at least 2")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    missing = sorted(set(estimators) - set(baselines))
    if missing:
        raise ValueError(f"no baseline supplied for: {', '.join(missing)}")

    names = list(estimators)
    n_sessions = estimators[names[0]].n_sessions
    terms: list[np.ndarray] = []
    slots: dict[str, tuple[int, int, int, int]] = {}
    for name in names:
        estimator, baseline = estimators[name], baselines[name]
        if estimator.n_sessions != n_sessions or baseline.n_sessions != n_sessions:
            raise ValueError(f"{name}: estimator and baseline must cover the same sessions")
        start = len(terms)
        terms.extend(
            [estimator.numerator, estimator.denominator, baseline.numerator, baseline.denominator]
        )
        slots[name] = (start, start + 1, start + 2, start + 3)
    matrix = np.ascontiguousarray(np.stack(terms, axis=1))

    draws = np.empty((resamples, len(names)), dtype=np.float64)
    for start in range(0, resamples, _COUNT_CHUNK):
        stop = min(start + _COUNT_CHUNK, resamples)
        counts = _count_matrix(n_sessions, stop - start, rng)
        sums = counts @ matrix
        for position, name in enumerate(names):
            a, b, c, d = slots[name]
            draws[start:stop, position] = _lift_from_sums(
                sums[:, a], sums[:, b], sums[:, c], sums[:, d]
            )

    out: dict[str, BootstrapInterval] = {}
    for position, name in enumerate(names):
        point = _lift(estimators[name].value(), baselines[name].value())
        finite = draws[:, position][np.isfinite(draws[:, position])]
        if finite.size < 2:
            out[name] = BootstrapInterval(
                point, float("nan"), float("nan"), float("nan"), resamples
            )
            continue
        low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        out[name] = BootstrapInterval(
            point=point,
            low=float(low),
            high=float(high),
            standard_error=float(finite.std(ddof=1)),
            resamples=resamples,
        )
    return out


def _lift_from_sums(
    numerator: np.ndarray,
    denominator: np.ndarray,
    base_numerator: np.ndarray,
    base_denominator: np.ndarray,
) -> np.ndarray:
    """Vectorised lift over a block of resamples, with degenerate draws returned as NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.where(denominator != 0, numerator / denominator, np.nan)
        base = np.where(base_denominator != 0, base_numerator / base_denominator, np.nan)
        return np.where(base > 0, value / base - 1.0, np.nan)


def _lift(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return value / baseline - 1.0
