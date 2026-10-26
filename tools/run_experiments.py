"""Run every experiment in order and write its JSON. One command, so CI cannot skip one.

Each experiment is independent and deterministic, so the order here is for reading rather than
for correctness: E1 establishes that the gap exists, E2 shows how fast the log stops being able
to answer, E3 compares the corrections, E4 says what the log could resolve at all, and E5
attacks the assumption the other four rest on.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

EXPERIMENTS = (
    "e1_sign_agreement",
    "e2_overlap_curve",
    "e3_estimator_comparison",
    "e4_detectable_lift",
    "e5_propensity_misspecification",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the experiments and write their JSON.")
    parser.add_argument(
        "--only", nargs="*", choices=EXPERIMENTS, help="run a subset, for iterating"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="write the JSON without printing the tables"
    )
    args = parser.parse_args(argv)

    selected = tuple(args.only) if args.only else EXPERIMENTS
    started = time.perf_counter()
    for name in selected:
        module = importlib.import_module(f"experiments.{name}")
        payload = module.run()
        if not args.quiet:
            module.report(payload)
    if not args.quiet:
        print()
        print(
            f"{len(selected)} experiment(s) in {time.perf_counter() - started:.1f}s. "
            f"JSON written to docs/experiments/."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
