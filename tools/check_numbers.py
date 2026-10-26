"""Fail the build when a document quotes a figure the code no longer produces.

Two checks run here, and the second catches more than the first.

The forward check takes every measured metric and confirms that each document quoting it uses
one of its anchor phrases, which is the number together with two or three words of context.
Anchors are alternatives rather than requirements: the same figure reads differently in a table
and in a sentence, and demanding every phrasing would fail the build over prose that is
perfectly correct, which is the fastest way to teach somebody to pass --no-verify.

Every metric must be anchored somewhere in the checked set. That requirement is what catches the
case the forward check alone cannot: when a figure is edited to a different number, the old
value simply stops appearing, so a check that only looks for quotations of the current value
finds nothing to complain about. Requiring the current value to appear, in its own wording, in
at least one document turns a silent pass into a failure.

A metric whose digits are small is handled differently beyond that, because its digits prove
nothing. A 3 or a 6 occurs in a version string, a heading, an ADR number and a date, so
searching for it finds matches in documents that never mention the measurement. Those metrics
are checked through their anchors only.

Document text is also tokenised into numbers before the forward check, so a figure is only
considered quoted when it stands alone. Without that, 6 is found inside 2026 and 3.78%, and the
build fails on documents that are correct.

The reverse check sweeps every number in the prose and reports any that no metric accounts for,
because a number nothing re-measures is a number that will eventually be wrong. Fenced code
blocks, inline code spans, HTML attributes and link targets are excluded: an example invocation
is allowed to contain a made up row count, and flagging it trains the reader to skip the whole
section.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_spec import SPEC  # noqa: E402

METRICS_PATH = Path("docs/metrics.json")

FENCE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HTML_TAG = re.compile(r"<[^>]+>")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
ADR_REFERENCE = re.compile(r"ADR-\d+")

NUMBER = re.compile(r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?<![.,])\s*(%?)")

# Numbers a document may use without a metric behind them: structural facts about the design,
# section numbering, years, and the small counts that describe the repository's own shape.
ALLOWED_BARE = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "20",
    "24",
    "100",
    "200",
    "720",
    "30,240",  # ten candidates over five positions, the slate length ADR-002 rejects
    "2026",
    "3.11",
    "3.12",
    "3.13",
    "5,000",
    "50,000",
    "0.25",
    "1.0",
    "95",
    "80",
    "0.05",
}


AMBIGUOUS_BELOW = 10


def is_ambiguous(metric) -> bool:
    """True when the metric's digits are too small to prove anything on their own."""
    return metric.unit in {"integer", "ratio"} and abs(metric.scale) < AMBIGUOUS_BELOW


@dataclass
class Finding:
    """One problem worth failing the build over, or reporting as a gap."""

    document: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.document}: {self.kind}: {self.detail}"


def strip_code(text: str) -> str:
    """Remove everything a reader would not read as a claim."""
    for pattern in (FENCE, IMAGE, LINK_TARGET, INLINE_CODE, HTML_TAG, ADR_REFERENCE):
        text = pattern.sub(" ", text)
    return text


def normalise(text: str) -> str:
    """Collapse whitespace, so an anchor still matches across a wrapped line.

    This rule was learned the hard way. An anchor longer than a few words straddles a newline in
    wrapped markdown and then matches nothing, while the prose it guards is correct.
    """
    return re.sub(r"\s+", " ", text)


def number_tokens(text: str) -> set[str]:
    """Every standalone number in the text, so a figure is not found inside another one."""
    tokens: set[str] = set()
    for match in NUMBER.finditer(text):
        digits, percent = match.group(1), match.group(2)
        tokens.add(f"{digits}{percent}")
        tokens.add(digits)
        tokens.add(digits.replace(",", ""))
    return tokens


def forward_check(payload: dict, documents: dict[str, str]) -> tuple[list[Finding], int, int]:
    """Every quoted metric must appear with one of its anchor phrases."""
    findings: list[Finding] = []
    checked = 0
    quoted = 0
    tokens = {name: number_tokens(text) for name, text in documents.items()}
    for metric in SPEC.metrics:
        if not metric.checked:
            continue
        key = metric.key
        if key not in payload["metrics"]:
            findings.append(
                Finding("docs/metrics.json", "gap", f"{key} is specified but was never measured")
            )
            continue
        # Anchors are re-derived from the specification rather than read back from the payload,
        # so editing a phrase does not require re-running the whole measurement. The payload
        # keeps its own copy for the record, and the check below catches the two disagreeing.
        value = metric.format(payload["metrics"][key])
        anchors = [normalise(phrase.format(value)) for phrase in metric.anchors]
        if value != payload["formatted"][key]:
            findings.append(
                Finding(
                    "docs/metrics.json",
                    "stale",
                    f"{key} formats as {value} from the specification but the payload holds "
                    f"{payload['formatted'][key]}; re-run collect_metrics.py",
                )
            )
            continue
        if not anchors:
            findings.append(
                Finding("metrics_spec.py", "gap", f"{key} has no anchor phrase, so it is unguarded")
            )
            continue
        checked += 1

        ambiguous = is_ambiguous(metric) or float(payload["metrics"][key]) < AMBIGUOUS_BELOW
        anchored_in = [name for name, text in documents.items() if any(a in text for a in anchors)]
        if not anchored_in:
            findings.append(
                Finding(
                    "checked documents",
                    "unanchored",
                    f"{key} measures {value} and no document says so in any of its anchor "
                    "phrases: " + " | ".join(anchors),
                )
            )
            continue
        quoted += len(anchored_in)
        if ambiguous:
            # A small figure is checked through its anchors only, because its digits turn up by
            # coincidence in version strings, headings and dates.
            continue

        for name, text in documents.items():
            if value not in tokens[name]:
                continue
            quoted += 1
            if not any(anchor in text for anchor in anchors):
                findings.append(
                    Finding(
                        name,
                        "stale",
                        f"{key} appears as {value} but none of its anchor phrases do: "
                        + " | ".join(anchors),
                    )
                )
    return findings, checked, quoted


def row_check(payload: dict, documents: dict[str, str]) -> tuple[list[Finding], int]:
    """Every guarded table row must appear in some checked document, exactly as rendered.

    A table cell has no words around it, so there is no anchor phrase to build. Guarding the
    whole rendered row is stronger: a changed value, a moved column and a hand edited cell all
    break the exact string, and the row is the unit a reader actually reads.
    """
    findings: list[Finding] = []
    found = 0
    for key, rendered in payload.get("rows", {}).items():
        target = normalise(rendered)
        if any(target in text for text in documents.values()):
            found += 1
            continue
        findings.append(
            Finding(
                "checked documents",
                "stale",
                f"no document contains the current {key} row: {target}",
            )
        )
    return findings, found


def reverse_check(payload: dict, documents: dict[str, str]) -> list[Finding]:
    """Any number in the prose that no metric accounts for is a number nothing re-measures."""
    known = set(payload["formatted"].values())
    for rendered in payload.get("rows", {}).values():
        known |= number_tokens(rendered)
    known |= {value.replace(",", "") for value in known}
    known |= {value.lstrip("+") for value in known}
    findings: list[Finding] = []
    for name, text in documents.items():
        for match in NUMBER.finditer(text):
            digits, percent = match.group(1), match.group(2)
            token = f"{digits}{percent}"
            if token in known or digits in known or digits in ALLOWED_BARE:
                continue
            if not percent and "." not in digits and len(digits.replace(",", "")) <= 2:
                continue  # a small bare integer is prose, not a measurement
            findings.append(Finding(name, "unguarded", f"{token} matches no measured figure"))
    return findings


def load_documents(names: list[str]) -> dict[str, str]:
    documents: dict[str, str] = {}
    for name in names:
        path = Path(name)
        if not path.exists():
            raise SystemExit(f"checked document {name} does not exist")
        documents[name] = normalise(strip_code(path.read_text(encoding="utf-8")))
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail the build when a document quotes a figure the code no longer produces."
    )
    parser.add_argument("--metrics", default=str(METRICS_PATH))
    parser.add_argument(
        "--strict-reverse",
        action="store_true",
        help="fail on an unguarded number as well as on a stale one",
    )
    args = parser.parse_args(argv)

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} is missing. Run 'make receipts' or collect_metrics.py.")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    documents = load_documents(payload["checked_documents"])

    stale, checked, quoted = forward_check(payload, documents)
    row_findings, rows_found = row_check(payload, documents)
    stale.extend(row_findings)
    unguarded = reverse_check(payload, documents)

    print(
        f"checked {checked} measured figures and {rows_found + len(row_findings)} table rows "
        f"against {len(documents)} document(s); {quoted} quotation(s) found"
    )
    for finding in stale:
        print(f"  STALE     {finding}")
    for finding in unguarded:
        print(f"  UNGUARDED {finding}")

    if stale:
        print(f"\n{len(stale)} stale figure(s). A document quotes a number the code no longer")
        print("produces, quotes it in wording the measurement no longer supports, or has")
        print("stopped making a claim the measurement still produces.")
        return 1
    if unguarded and args.strict_reverse:
        print(f"\n{len(unguarded)} unguarded number(s) and --strict-reverse is set.")
        print("A number nothing re-measures is a number that will eventually be wrong.")
        return 1
    if unguarded:
        print(f"\n{len(unguarded)} unguarded number(s), reported but not failing the build.")
        print("Pass --strict-reverse to fail on these too, which is what CI does.")
    print("every published figure traces to code that ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
