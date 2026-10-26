"""One exception type per reason the tool can fail, because the CLI maps them to exit codes.

A caller needs to tell "the tool worked and the answer is no" apart from "the tool could not
answer", and that distinction is only cheap to make if the failure modes are separate types at
the point they are raised rather than string-matched later.
"""

from __future__ import annotations


class ReplayGapError(Exception):
    """Base class, so a caller can catch everything this package raises and nothing else."""


class ConfigError(ReplayGapError):
    """A configuration value is missing, the wrong type, or outside the range the code assumes."""


class FixtureError(ReplayGapError):
    """A fixture is absent, truncated, or does not match the hash recorded in its manifest."""


class RefusalError(ReplayGapError):
    """The question cannot be answered honestly from this log.

    This is not a bug and not a negative result. It is the tool declining, and it is the reason
    exit code 2 exists. The message must say what would need to change for the question to become
    answerable, because a refusal with no remedy is just a failure.
    """

    def __init__(self, message: str, remedy: str) -> None:
        super().__init__(message)
        self.remedy = remedy
