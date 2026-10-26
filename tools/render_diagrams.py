"""Render the README's architecture diagram to a committed SVG.

GitHub does not reliably render a mermaid fence. The failure is inside their lazily loaded
renderer, shows as "Unable to render rich display", and there is nothing fixable in the source.
So the diagram is rendered here, committed as an SVG, and embedded with an img tag, which then
renders identically on GitHub, in an editor preview, in a PDF export and offline.

Three details, each from a failure that cost real time:

htmlLabels must be off. With HTML labels on, mermaid puts markup inside a foreignObject,
including unclosed br tags, so the file is not well formed XML. It still displays when injected
into a live page and silently fails to load as an img src: naturalWidth zero, a broken image
icon, and nothing in any console explaining why.

The root element needs intrinsic dimensions. Mermaid emits a percentage width with a max-width
style, which is right for a live page and wrong for an img src: with no height and a percentage
width a browser has no aspect ratio to work from and picks a default.

The background must be painted. Transparent sounds theme neutral and is not. Node fills are
light with dark text either way, so subgraph titles and edge labels, which have nothing behind
them, come out dark grey on near black in a dark theme.

And the committed file is loaded back as an img and its naturalWidth asserted, because that is
the only check that catches the first failure above. The verification page is written next to
the SVG and opened over file://, not injected with set_content: a page built by set_content has
an about:blank origin, Chromium refuses to load a file:// subresource into it, and the img
reports naturalWidth zero. That is the same symptom as a genuinely broken SVG, so a check
written the convenient way fails on a file that is perfectly good.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

README = Path("README.md")
OUTPUT_DIR = Path("docs/diagrams")
SOURCE_FENCE = re.compile(r"```mermaid-source\n(.*?)```", re.DOTALL)
MERMAID_BUNDLE_CANDIDATES = (
    Path("node_modules/mermaid/dist/mermaid.min.js"),
    Path("/usr/lib/node_modules/mermaid/dist/mermaid.min.js"),
    Path("/opt/npm-tools/node_modules/mermaid/dist/mermaid.min.js"),
)


def find_mermaid() -> Path | None:
    """Locate a local mermaid bundle. Attached as a script tag, never inlined.

    Inlining three megabytes of JavaScript into a scratch page worked for some diagrams and
    timed out for others, which is the kind of flake that costs an hour to attribute.
    """
    for candidate in MERMAID_BUNDLE_CANDIDATES:
        if candidate.exists():
            return candidate
    override = os.environ.get("MERMAID_BUNDLE")
    if override and Path(override).exists():
        return Path(override)
    return None


def extract_sources(readme: Path) -> list[tuple[str, str]]:
    """Pull every mermaid-source fence out of the README, in order.

    The fence is labelled mermaid-source rather than mermaid so that nothing tries to render it
    and produce the broken display message next to the image that already works.
    """
    text = readme.read_text(encoding="utf-8")
    blocks = SOURCE_FENCE.findall(text)
    return [(f"architecture-{index + 1}", block.strip()) for index, block in enumerate(blocks)]


def render(source: str, mermaid_js: Path) -> str:
    """Render one diagram to SVG text through a headless browser."""
    from playwright.sync_api import sync_playwright

    page_html = """<!doctype html><html><head><meta charset="utf-8"></head>
<body><div id="target"></div></body></html>"""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.set_content(page_html)
        page.add_script_tag(path=str(mermaid_js))
        svg = page.evaluate(
            """async (source) => {
                mermaid.initialize({
                    startOnLoad: false,
                    htmlLabels: false,
                    flowchart: { htmlLabels: false, curve: 'basis' },
                    theme: 'neutral',
                    securityLevel: 'strict',
                });
                const { svg } = await mermaid.render('generated', source);
                return svg;
            }""",
            source,
        )
        browser.close()
    return svg


def fix_root(svg: str) -> str:
    """Give the SVG intrinsic dimensions from its viewBox and paint a light background."""
    viewbox = re.search(r'viewBox="([^"]+)"', svg)
    if not viewbox:
        raise SystemExit("rendered SVG has no viewBox, so it cannot be given a size")
    _, _, width, height = (float(value) for value in viewbox.group(1).split())
    svg = re.sub(r'\swidth="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sheight="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sstyle="[^"]*max-width[^"]*"', "", svg, count=1)
    svg = svg.replace("<svg ", f'<svg width="{width:.0f}" height="{height:.0f}" ', 1)
    background = f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>'
    marker = svg.index(">", svg.index("<svg")) + 1
    return svg[:marker] + background + svg[marker:]


def verify_loads(svg_path: Path) -> dict:
    """Open the committed SVG as an img and assert the browser gave it a real size.

    A broken SVG reports naturalWidth zero with nothing in any console to say why, so this is
    the only step in the pipeline that can tell a valid file from one that merely parses.
    """
    from playwright.sync_api import sync_playwright

    page_path = svg_path.with_name(f"_verify_{svg_path.stem}.html")
    page_path.write_text(
        f'<body style="margin:0"><img id="d" src="{svg_path.name}"></body>', encoding="utf-8"
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{page_path.resolve()}")
            page.wait_for_timeout(500)
            measured = page.evaluate(
                """() => {
                    const image = document.getElementById('d');
                    return {
                        naturalWidth: image.naturalWidth,
                        naturalHeight: image.naturalHeight,
                        complete: image.complete,
                    };
                }"""
            )
            browser.close()
    finally:
        page_path.unlink(missing_ok=True)
    if not measured["naturalWidth"] or not measured["naturalHeight"]:
        raise SystemExit(
            f"{svg_path} does not load as an image (naturalWidth "
            f"{measured['naturalWidth']}). Check that htmlLabels is off and that the root "
            f"element carries explicit width and height."
        )
    return measured


def main() -> int:
    mermaid_js = find_mermaid()
    if mermaid_js is None:
        print(
            "no local mermaid bundle found. Run 'npm install mermaid' in the repository "
            "root, or point MERMAID_BUNDLE at a mermaid.min.js, then rerun. The committed SVG "
            "under docs/diagrams is what the README embeds, so this step is only needed when "
            "the diagram source changes.",
            file=sys.stderr,
        )
        return 1
    sources = extract_sources(README)
    if not sources:
        print("no mermaid-source fence found in README.md", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, source in sources:
        svg = fix_root(render(source, mermaid_js))
        target = OUTPUT_DIR / f"{name}.svg"
        target.write_text(svg, encoding="utf-8")
        viewbox = re.search(r'viewBox="([^"]+)"', svg).group(1)
        measured = verify_loads(target)
        manifest.append(
            {
                "name": name,
                "file": str(target),
                "viewbox": viewbox,
                "bytes": len(svg),
                "html_labels": False,
                "background": "painted",
                "loads_as_image": True,
                "natural_width": measured["naturalWidth"],
                "natural_height": measured["naturalHeight"],
            }
        )
        print(
            f"wrote {target} ({len(svg):,} bytes, viewBox {viewbox}), loads at "
            f"{measured['naturalWidth']}x{measured['naturalHeight']}"
        )
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps({"mermaid": str(mermaid_js), "diagrams": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
