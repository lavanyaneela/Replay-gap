"""How small a true lift this log could have detected, and the floor below which it cannot.

Two functions, and the second is the one that keeps the repository honest.

``minimum_detectable_lift`` converts the bootstrap spread of the corrected estimate into the
smallest true lift a two sided test at the configured alpha would reject the null for, at the
configured power. It is almost always larger than the lift teams actually ship on, which is the
point.

``resolution_floor`` is receipts rule four in a function. A measured zero over n trials is not
zero, it is below one in n, and any claim built on a measured zero has to be priced at that
rate instead.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def minimum_detectable_lift(
    standard_error: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest detectable relative lift, given the standard error of the lift estimate.

    The standard error is already expressed as a lift, so the conversion is the usual sum of two
    normal quantiles and nothing else. Written as its own function anyway, because the constant
    2.8 appearing inline in a report is how a power calculation becomes folklore.
    """
    if standard_error < 0:
        raise ValueError("standard error cannot be negative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between 0 and 1")
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_power = float(norm.ppf(power))
    return (z_alpha + z_power) * standard_error


def resolution_floor(n_trials: int) -> float:
    """One over the sample size: the smallest non-zero rate this sample could have shown.

    A measured zero is reported as "below one in n", never as zero, and a claim that needs a
    rate smaller than this floor needs a bigger sample, not a better sentence.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    return 1.0 / n_trials


def sessions_needed_for_rate(rate: float, n_events: int = 1) -> int:
    """How many trials a sample would need before the given rate is resolvable at all."""
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must lie in (0, 1]")
    if n_events < 1:
        raise ValueError("n_events must be at least 1")
    return int(math.ceil(n_events / rate))
