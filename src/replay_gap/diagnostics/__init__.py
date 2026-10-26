"""The numbers that decide whether an estimate is worth printing at all.

A corrected estimate with no diagnostic beside it is the second most dangerous artefact in
offline evaluation, after an uncorrected one. The correction is exact here, so it cannot be
wrong, but it can be worthless: importance weighting a log where the candidate policy barely
overlaps the logging policy produces an estimate that is unbiased and governed by a handful of
sessions. These functions are how the tool knows to refuse.
"""

from replay_gap.diagnostics.detectable_lift import minimum_detectable_lift, resolution_floor
from replay_gap.diagnostics.ess import effective_sample_size, ess_fraction
from replay_gap.diagnostics.overlap import OverlapReport, assess_overlap

__all__ = [
    "OverlapReport",
    "assess_overlap",
    "effective_sample_size",
    "ess_fraction",
    "minimum_detectable_lift",
    "resolution_floor",
]
