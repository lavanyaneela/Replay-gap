"""Fail the build when a document quotes a figure the code no longer produces.

Two checks run here, and the second catches more than the first.

The forward check takes every measured metric and confirms that each document quoting it uses
one of its anchor phrases, which is the number together with two or three words of context.
Anchors are alternatives rather than requirements: the same figure reads differently in a table
and in a sentence, and demanding every phrasing would fail the build over prose that is
perfectly correct, which is the fastest way to teach somebody to pass --no-verify.

Every metric must be anchored somewhere in the checked set, and every document that quotes a
metric's value must itself carry one of that metric's anchors. The first rule catches a figure
edited to a different number, where the old value simply stops appearing and a check looking
only for quotations of the current value has nothing to complain about. The second rule is the
one this checker was missing: requiring the anchor anywhere in the set meant a README whose
every framing sentence had gone stale still passed, because the defense guide happened to retain
the phrasing. The proof transcript published that as a case the checker did not catch, which is
how it was found.

A metric whose rendered value is a short bare integer is exempt from the second rule only,
because its digits prove nothing: a 3 or a 6 occurs in a version string, a heading, an ADR
number and a date, so searching for it finds matches in documents that never mention the
measurement. Judging that on the rendered string rather than on the raw value matters. An
earlier version compared the raw value against a threshold of ten, which made 0.0553 "small"
and exempted almost every percentage in the repository from the per document check.

Document text is also tokenised into numbers before the forward check, so a figure is only
considered quoted when it stands alone. Without that, 6 is found inside 2026 and 3.78%, and the
build fails on documents that are correct.

The reverse check sweeps every number in the prose and reports any that no metric accounts for,
because a number nothing re-measures is a number that will eventually be wrong. Fenced code
blocks, inline code spans, HTML attributes and link targets are excluded from that sweep, since
a command line is allowed to contain an argument that is not a measurement.

Excluding fences entirely was wrong, though, and an audit caught it. The README's two most read
blocks are not examples: they are captured output, they carry the headline and the refusal, and
stripping them meant the first screen of the document was the one part nothing checked. Falsified
figures in both passed. So the row check below reads the raw document rather than the stripped
one, and those blocks are guarded line by line as exact strings.

A second audit then found that this closed mutation and left insertion open. Guarding the known
lines proves those lines are current; it says nothing about a line that was never guarded. A
fabricated "precision at 5  99.4%" inserted into the headline block passed the build, which is
the overclaim this whole mechanism exists to prevent, sitting on the first screen under a green
check. So :func:`excluded_region_check` sweeps the numerals inside everything ``strip_code``
removes and requires each one to belong to a guarded row or a measured value. That is the
inverse rule: the forward check asks whether every measurement is claimed correctly, and this
asks whether every claim is measured.
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
# Alt text is prose: a screen reader speaks it and a broken image shows it. A src or a width is
# not, so neither is swept.
IMAGE_ALT = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
HTML_ALT = re.compile(r'\balt="([^"]*)"')

NUMBER = re.compile(r"(?<![\w.])([+-]?\d[\d,]*(?:\.\d+)?)(?<![.,])\s*(%?)")

# Numbers a document may use without a measurement behind them. These are structural facts
# about the design rather than results: the slate geometry, the supported Python versions, the
# configured thresholds, and single digits that read as words.
#
# Nothing here may collide with a live metric's rendered value, because an exemption that
# covers a measured figure silently excuses it from the reverse check. The assertion below
# enforces that, and it fired on 24, which was both an exemption and the E3 replicate count.
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
    "200",
    "720",
    "30,240",  # ten candidates over five positions, the slate length ADR-002 rejects
    "0.5",  # the quantile argument in PERCENTILE_CONT(0.5), which is SQL rather than a result
    "2026",
    "3.11",
    "3.12",
    "3.13",
}

# Deliberately absent, because each is a measured value guarded by the sweep like any other:
# the log size, the context count, the significance level, the power, the coverage percentage
# and the overlap floor. An exemption covering one of those is what the assertion below exists
# to catch, and it has caught several.


AMBIGUOUS_DIGITS = 3


def is_ambiguous(formatted: str) -> bool:
    """True when the rendered value is too plain for its digits to prove anything.

    Judged on what a document actually contains. "6" and "24" turn up by coincidence; "+5.53%",
    "1.62%" and "17,187" do not, because a decimal point, a percent sign or four digits make a
    token specific enough that finding it means the document really is quoting the figure.
    """
    if "%" in formatted or "." in formatted:
        return False
    return len(formatted.lstrip("+-").replace(",", "")) <= AMBIGUOUS_DIGITS


def assert_no_exemption_hides_a_metric(payload: dict) -> list[Finding]:
    """An exemption that covers a specific measured value would excuse it from the sweep.

    Ambiguous metrics are allowed to collide and are not reported. A bare 3 cannot be guarded by
    a sweep over numerals whatever the exemption list says, which is why those metrics are
    required to be anchored by phrase instead. The collision that matters is a specific value
    such as 50,000 sitting in the exemption list, where the sweep could have guarded it and the
    exemption is what stopped it.

    The comparison covers the bare digits as well as the rendered string, because that is what
    the reverse check actually exempts on. An earlier version compared only the rendered string,
    so a metric rendering as 95% was invisible to the assertion while 95 sat in the list.
    """
    collisions = sorted(
        key
        for key, value in payload["formatted"].items()
        if not is_ambiguous(value)
        and {
            value,
            value.rstrip("%"),
            value.lstrip("+-").replace(",", ""),
            value.lstrip("+-").rstrip("%").replace(",", ""),
        }
        & ALLOWED_BARE
    )
    return [
        Finding(
            "check_numbers.py",
            "exemption",
            f"{key} renders as {payload['formatted'][key]}, which ALLOWED_BARE exempts, so the "
            "reverse check would never see it go stale",
        )
        for key in collisions
    ]


def assert_no_anchor_is_shared(payload: dict) -> list[Finding]:
    """Two metrics must not render the same anchor phrase.

    When they do, one metric's claim can be deleted outright and the other's sentence satisfies
    the check on its behalf, which is exactly the property the per document rule was added to
    create. An audit found this between the two figures that both render 1.62%, and deleting
    the sweep claim from the README passed.
    """
    rendered: list[tuple[str, str]] = [
        (normalise(anchor), key)
        for key, anchors in payload["anchors"].items()
        for anchor in anchors
    ]
    findings: list[Finding] = []
    for index, (anchor, key) in enumerate(rendered):
        for other_anchor, other_key in rendered[index + 1 :]:
            if other_key == key:
                continue
            # Containment matters as much as equality. An audit found that renaming one anchor
            # to "sweep ends at 1.62%" left the other's "ends at 1.62%" as a suffix of it, so
            # one sentence still satisfied both metrics and the property was still violated.
            if anchor == other_anchor or anchor in other_anchor or other_anchor in anchor:
                findings.append(
                    Finding(
                        "metrics_spec.py",
                        "shared anchor",
                        f"{anchor!r} and {other_anchor!r} guard {key} and {other_key}, so one "
                        "sentence satisfies both and either claim can be deleted unnoticed",
                    )
                )
    return findings


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


def split_paragraphs(text: str) -> list[str]:
    """The document as normalised paragraphs, which is the unit the anchor rule works in.

    A markdown table, a fenced block and a list are each one paragraph here, because a blank
    line is what separates them from the prose around them, and that is the same boundary a
    reader perceives.
    """
    return [normalise(block) for block in re.split(r"\n\s*\n", text) if block.strip()]


def covered_spans(text: str, phrases: list[str]) -> list[tuple[int, int]]:
    """Character ranges of the text covered by any of these phrases."""
    spans: list[tuple[int, int]] = []
    for phrase in phrases:
        if not phrase:
            continue
        start = text.find(phrase)
        while start != -1:
            spans.append((start, start + len(phrase)))
            start = text.find(phrase, start + 1)
    return spans


def value_occurrences(text: str, value: str) -> list[tuple[int, int]]:
    """Every place the rendered value stands alone as a number, with its position.

    Standing alone matters: without it, 810 is found inside 8,100 and 1.62% inside 11.62%.
    """
    spans: list[tuple[int, int]] = []
    for match in NUMBER.finditer(text):
        digits, percent = match.group(1), match.group(2)
        token = f"{digits}{percent}"
        if token != value and digits != value and digits.replace(",", "") != value.replace(",", ""):
            continue
        # The pattern allows whitespace before an optional percent sign, so match.end() can sit
        # past the number itself. Using the group boundaries keeps the span on the digits, which
        # matters because an anchor phrase ends at the number rather than after it.
        stop = match.end(2) if percent else match.end(1)
        spans.append((match.start(1), stop))
    return spans


def forward_check(payload: dict, documents: dict[str, str]) -> tuple[list[Finding], int, int]:
    """Every place a metric's value is written must sit inside one of its anchor phrases.

    Per occurrence rather than per document, and that is the third rule this check has had. The
    first required an anchor anywhere in the checked set, so a README whose every framing
    sentence had gone stale passed because another file retained the wording. The second
    required it in the quoting document, which still allowed a false sentence to sit beside a
    true one: an audit inverted the central finding of the overlap sweep into "overlap stays
    comfortable throughout, never dropping under 1.62%" and the build stayed green, because the
    correct sentence about the same figure was a hundred lines further down.

    So each occurrence has to be inside an anchor of its own, inside a guarded row, which is how
    a table cell and a line of captured output earn their place without words around them, or in
    a paragraph where the claim is already made properly.

    That last exemption is the one that keeps this rule usable. Without it a back reference, a
    parenthetical, a heading or a summary bullet fails the build while being perfectly correct,
    and a checker that fails on correct prose is a checker somebody will route around. The
    paragraph is the right unit: it is small enough that an inverted sentence a hundred lines
    away is still caught, and large enough that "That +5.53% is the estimator's bias and nothing
    else" can follow the sentence that anchored it.
    """
    findings: list[Finding] = []
    checked = 0
    quoted = 0
    tokens = {name: number_tokens(text) for name, text in documents.items()}
    rows = [normalise(row) for row in payload.get("rows", {}).values()]
    paragraphs = {name: split_paragraphs(text) for name, text in documents.items()}

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

        ambiguous = is_ambiguous(value)
        anchored_in = [
            name
            for name, blocks in paragraphs.items()
            if any(anchor in block for block in blocks for anchor in anchors)
        ]
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

        for name in documents:
            if value not in tokens[name]:
                continue
            quoted += 1
            for paragraph in paragraphs[name]:
                occurrences = value_occurrences(paragraph, value)
                if not occurrences:
                    continue
                if any(anchor in paragraph for anchor in anchors):
                    # The claim is made properly in this paragraph, so a back reference, a
                    # parenthetical or a summary line beside it is allowed to repeat the figure.
                    continue
                allowed = covered_spans(paragraph, rows)
                for start, stop in occurrences:
                    if any(low <= start and stop <= high for low, high in allowed):
                        continue
                    context = paragraph[max(0, start - 60) : stop + 60].strip()
                    findings.append(
                        Finding(
                            name,
                            "stale",
                            f"{key} is written as {value} in a paragraph that does not claim "
                            f"it: ...{context}... (anchors: " + " | ".join(anchors) + ")",
                        )
                    )
    return findings, checked, quoted


def row_check(payload: dict, documents: dict[str, str]) -> tuple[list[Finding], int]:
    """Every guarded row must appear in some checked document, exactly as rendered.

    A table cell and a column of captured terminal output both have no words around them, so
    there is no anchor phrase to build. Guarding the whole rendered line is stronger: a changed
    value, a moved column and a hand edited cell all break the exact string, and the line is the
    unit a reader actually reads.

    These are matched against the raw document, not the stripped one, because the output blocks
    this guards live inside fences.
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


def excluded_regions(text: str) -> str:
    """Everything the reverse sweep skips but a reader still reads as a claim.

    That is every fence whatever its language tag, every inline code span, and the alt text of
    every image. An audit inserted a false claim into each of the three and watched the build
    stay green, so the rule is now inclusive rather than clever: a figure a reader can see is a
    figure that has to be measured, wherever it sits.

    An earlier version swept only fences and spans that already contained a percent sign, which
    is why "a 2,400,000 session log" in the layout block and "41,200 slates per second" in a
    code span both passed. The small integer exemption further down is what keeps a command
    line argument from being treated as a measurement, and it does that job without needing a
    percent sign to decide.

    Link targets, image sources and width attributes stay out: a URL or a pixel count is not a
    claim, and sweeping them trains a reader to ignore the section.
    """
    regions = [match.group(0) for match in FENCE.finditer(text)]
    regions.extend(match.group(0) for match in INLINE_CODE.finditer(text))
    regions.extend(match.group(1) for match in IMAGE_ALT.finditer(text))
    regions.extend(match.group(1) for match in HTML_ALT.finditer(text))
    return normalise(" ".join(regions))


def excluded_region_check(payload: dict, raw_documents: dict[str, Path | str]) -> list[Finding]:
    """Numerals inside fences and code spans must belong to a guarded row or a measurement.

    The row check proves the guarded lines are current. This proves nothing else was added
    beside them, which is the other half and the one an audit found missing.
    """
    known = set(payload["formatted"].values())
    known |= {value.replace(",", "") for value in known}
    known |= {value.lstrip("+") for value in known}
    for rendered in payload.get("rows", {}).values():
        known |= number_tokens(rendered)

    findings: list[Finding] = []
    for name, text in raw_documents.items():
        for match in NUMBER.finditer(excluded_regions(str(text))):
            digits, percent = match.group(1), match.group(2)
            token = f"{digits}{percent}"
            if token in known or digits in known or digits in ALLOWED_BARE:
                continue
            if not percent and "." not in digits and len(digits.replace(",", "")) <= 2:
                continue  # a small bare integer inside a command line is an argument
            findings.append(
                Finding(
                    name,
                    "fabricated",
                    f"{token} sits inside a code block or span and matches no measured figure "
                    "and no guarded line",
                )
            )
    return findings


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


def load_documents(names: list[str]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Each checked document three ways: stripped, normalised, and exactly as written.

    The anchor and reverse checks read the stripped text, because a command line is not a claim.
    The row check reads the normalised text, because the blocks of captured output it guards are
    inside fences. The excluded region check reads the text exactly as written, newlines and
    all, because the patterns that find a fence are anchored on line breaks.

    That third form exists because of a real defect. Both other forms run through normalise,
    which collapses newlines to spaces, so a fence pattern requiring a newline could never match
    and the function documenting it was dead code. The sweep still caught insertions, by
    accident: on newline collapsed text a fence's three backticks pair up as inline spans, so
    fence bodies were being captured by the wrong pattern. It worked, and a mechanism that works
    for a reason nobody wrote down is a mechanism nobody can safely change.
    """
    stripped: dict[str, str] = {}
    normalised: dict[str, str] = {}
    verbatim: dict[str, str] = {}
    for name in names:
        path = Path(name)
        if not path.exists():
            raise SystemExit(f"checked document {name} does not exist")
        text = path.read_text(encoding="utf-8")
        stripped[name] = strip_code(text)
        normalised[name] = normalise(text)
        verbatim[name] = text
    return stripped, normalised, verbatim


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
    documents, normalised_documents, verbatim_documents = load_documents(
        payload["checked_documents"]
    )

    stale = assert_no_exemption_hides_a_metric(payload)
    stale.extend(assert_no_anchor_is_shared(payload))
    forward_findings, checked, quoted = forward_check(payload, documents)
    stale.extend(forward_findings)
    row_findings, rows_found = row_check(payload, normalised_documents)
    stale.extend(row_findings)
    stale.extend(excluded_region_check(payload, verbatim_documents))
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
