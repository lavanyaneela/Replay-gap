"""Enforce the promise that no absolute duration is published as a claim in this repository.

The README and ADR-005 both say that no absolute duration appears anywhere here, because a
published millisecond is a fact about whichever machine ran it rather than about the code. That
promise covers the whole repository, and the number checker only covers seven documents. An
audit pointed out that a fabricated performance figure in a source docstring passed everything,
which made the broadest claim in the repository the one nothing enforced.

So this scans every tracked text file for a duration shaped numeral and fails on one. Durations
still go into the experiment JSON for the record, which is why that directory is excluded: the
rule is that a duration must not be published as a claim, not that nothing may measure one.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERN = re.compile(
    r"\b(\d[\d,.]*)\s*(ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?)\b", re.IGNORECASE
)

TRACKED = ("*.py", "*.md", "*.yaml", "*.yml", "*.sql", "*.toml", "Makefile")

# Paths excluded, each for a stated reason rather than for convenience.
EXCLUDED_PREFIXES = (
    "docs/experiments/",  # measured durations belong in the JSON; that is the point of the rule
    "docs/video/",  # the demo manifest records how long each captured command took
    "docs/evidence/",  # the screenshot manifest is machine written
    "tools/check_durations.py",  # this file names the units it looks for
)

# Phrases that are not measurements of this repository's behaviour.
ALLOWED = (
    "30 second version",  # a heading about how long to talk, not about how long code takes
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", *TRACKED], capture_output=True, text=True, check=True
    )
    return [Path(name) for name in result.stdout.split() if name]


def scan(paths: list[Path]) -> list[tuple[Path, int, str, str]]:
    findings = []
    for path in paths:
        if any(str(path).startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, FileNotFoundError):  # pragma: no cover - binary or removed
            continue
        for number, line in enumerate(lines, start=1):
            for match in PATTERN.finditer(line):
                if any(phrase in line for phrase in ALLOWED):
                    continue
                findings.append((path, number, match.group(0), line.strip()))
    return findings


def main() -> int:
    paths = tracked_files()
    findings = scan(paths)
    print(f"scanned {len(paths)} tracked file(s) for published durations")
    for path, number, text, line in findings:
        print(f"  {path}:{number}  {text!r}  in: {line[:90]}")
    if findings:
        print(
            f"\n{len(findings)} absolute duration(s) published as a claim. A millisecond is a "
            "fact about the machine that ran it, so publish a ratio or a count instead, or add "
            "the phrase to ALLOWED with a reason."
        )
        return 1
    print("no absolute duration is published as a claim anywhere in this repository")
    return 0


if __name__ == "__main__":
    sys.exit(main())
