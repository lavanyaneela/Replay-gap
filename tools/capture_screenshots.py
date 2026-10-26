"""Capture the README's evidence from the real HTML report the tool produces.

Three rules here, each from a failure rather than from taste.

Frame by named heading, not by pixel offset. The shot is clipped from one h2 to the next, read
out of the live document. A hardcoded crop drifts silently the first time a paragraph gets
longer, and then the README shows half a table. A missing heading fails the run, because it
means the report changed shape and the shot is no longer of what its caption says.

Pass full_page even when a clip is given. Without it Chromium clamps the clip to the viewport
and every shot comes out exactly one viewport tall with its bottom missing.

Read the computed style of the verdict badge back and fail if it is invisible. In two earlier
projects a badge class and a table cell class collided at equal CSS specificity, the later rule
won, and the headline verdict rendered green on green. The Python was correct, the HTML was
correct, the tests passed, and the number nobody could read was the one the whole page existed
for. A screenshot tool is the only thing in the pipeline that can see pixels, so this is where
the check belongs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVIDENCE_DIR = Path("docs/evidence")
REPORT_PATH = Path("reports/evaluation.html")
MIN_CONTRAST = 3.0


def ensure_report(path: Path) -> Path:
    """Build the report if it is absent, rather than photographing whatever is on disk.

    A screenshot of a stale report is worse than no screenshot, because it carries the
    authority of a real capture and the content of an old run.
    """
    if path.exists():
        return path
    from replay_gap.cli import main

    print(f"{path} is missing; building it with 'replay-gap compare'")
    main(["compare", "--report-dir", str(path.parent)])
    if not path.exists():
        raise SystemExit(f"{path} still missing after building the report")
    return path


def _luminance(rgb: tuple[float, float, float]) -> float:
    """Relative luminance, the WCAG definition, for the contrast ratio below."""

    def channel(value: float) -> float:
        value = value / 255.0
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (channel(part) for part in rgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _parse_rgb(text: str) -> tuple[float, float, float]:
    inner = text[text.index("(") + 1 : text.index(")")]
    parts = [float(part.strip()) for part in inner.split(",")[:3]]
    return parts[0], parts[1], parts[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Contrast between two computed colours, so an invisible badge is a number not an opinion."""
    light = _luminance(_parse_rgb(foreground))
    dark = _luminance(_parse_rgb(background))
    if light < dark:
        light, dark = dark, light
    return (light + 0.05) / (dark + 0.05)


def capture(report: Path, out_dir: Path) -> dict:
    """Frame each named section and read the verdict badge's rendered colours back."""
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    shots = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1180, "height": 900}, device_scale_factor=2)
        page.goto(f"file://{report.resolve()}")
        page.wait_for_timeout(300)

        headings = page.evaluate(
            """() => Array.from(document.querySelectorAll('h1, h2')).map(node => ({
                text: node.textContent.trim(),
                top: node.getBoundingClientRect().top + window.scrollY,
            }))"""
        )
        if not headings:
            raise SystemExit("the report has no headings, so there is nothing to frame by")

        badge = page.evaluate(
            """() => {
                const node = document.querySelector('.verdict');
                if (!node) return null;
                const style = getComputedStyle(node);
                return {
                    text: node.textContent.trim(),
                    colour: style.color,
                    background: style.backgroundColor,
                    width: node.offsetWidth,
                    height: node.offsetHeight,
                };
            }"""
        )
        if badge is None:
            raise SystemExit("the report has no verdict badge, which is the headline of the page")
        if badge["width"] == 0 or badge["height"] == 0:
            raise SystemExit("the verdict badge has zero size, so it is not on the page")
        ratio = contrast_ratio(badge["colour"], badge["background"])
        if ratio < MIN_CONTRAST:
            raise SystemExit(
                f"the verdict badge reads {badge['colour']} on {badge['background']}, a contrast "
                f"ratio of {ratio:.2f}, below the {MIN_CONTRAST} floor. The headline of the "
                f"report is illegible."
            )

        full_height = page.evaluate("() => document.body.scrollHeight")
        targets = [
            ("report-table", "Estimated lift over the deployed ranker"),
        ]
        for name, heading_text in targets:
            match = next((h for h in headings if h["text"] == heading_text), None)
            if match is None:
                available = ", ".join(h["text"] for h in headings)
                raise SystemExit(
                    f"the report no longer has a heading '{heading_text}', so the shot named "
                    f"{name} would be of something else. Headings present: {available}"
                )
            start = max(0, match["top"] - 28)
            later = [h["top"] for h in headings if h["top"] > match["top"] + 4]
            stop = min(later) - 12 if later else full_height
            target = out_dir / f"{name}.png"
            page.screenshot(
                path=str(target),
                full_page=True,
                clip={"x": 0, "y": start, "width": 1180, "height": stop - start},
            )
            shots.append(
                {
                    "name": name,
                    "file": str(target),
                    "framed_by": heading_text,
                    "framed_until": "the next heading" if later else "the end of the page",
                    "height_px": round(stop - start),
                    "bytes": target.stat().st_size,
                }
            )
            print(f"wrote {target} ({target.stat().st_size:,} bytes)")

        # A second shot of the whole page, which is what a reviewer wants to see once.
        whole = out_dir / "report-full.png"
        page.screenshot(path=str(whole), full_page=True)
        shots.append(
            {
                "name": "report-full",
                "file": str(whole),
                "framed_by": "the whole document",
                "framed_until": "the end of the page",
                "height_px": full_height,
                "bytes": whole.stat().st_size,
            }
        )
        print(f"wrote {whole} ({whole.stat().st_size:,} bytes)")
        browser.close()

    return {
        "report": str(report),
        "verdict_badge": {
            "text": badge["text"],
            "colour": badge["colour"],
            "background": badge["background"],
            "contrast_ratio": round(ratio, 2),
            "minimum_required": MIN_CONTRAST,
            "legible": True,
        },
        "shots": shots,
        "note": (
            "Each shot is framed by a named heading read from the live document, never by a "
            "pixel offset, and the run fails if a named heading is gone or if the verdict badge "
            "is not legible against its own background."
        ),
    }


def main() -> int:
    report = ensure_report(REPORT_PATH)
    manifest = capture(report, EVIDENCE_DIR)
    (EVIDENCE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    badge = manifest["verdict_badge"]
    print(
        f"verdict badge reads {badge['text']} at contrast {badge['contrast_ratio']}, "
        f"floor {badge['minimum_required']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
