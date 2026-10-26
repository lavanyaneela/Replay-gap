"""replay-gap: offline ranker evaluation that reports when it cannot answer.

The log was produced by the ranker that is already deployed. Replaying a candidate against it
cannot separate a genuinely better model from one that merely agrees with the logging policy more
often, so the package publishes the correction, the effective sample size the correction leaves,
and a refusal path for the case where no correction is honest.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
