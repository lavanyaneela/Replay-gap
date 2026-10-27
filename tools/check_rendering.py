"""Render the README the way GitHub does and assert every image actually resolves.

python-markdown disagrees with GitHub about exactly this kind of file in several ways, so the
rendering here uses cmark-gfm, which is the library GitHub itself uses. The rendered page is
then loaded in a headless browser and every image is checked for a non-zero naturalWidth,
because a broken image reports nothing in any console and the first person to notice it is a
reviewer looking at the repository.

The page is written next to the README and opened over file://, not injected with set_content.
A page built by set_content has an about:blank origin and Chromium refuses to load a file://
subresource into it, so every local image reports naturalWidth zero and the check fails on a
document that is perfectly good.
"""

from __future__ import annotations

import sys
from pathlib import Path

README = Path("README.md")


def render(source: str) -> str:
    """Markdown to HTML through the library GitHub uses, with its extensions enabled."""
    import cmarkgfm
    from cmarkgfm.cmark import Options

    return cmarkgfm.markdown_to_html_with_extensions(
        source,
        options=Options.CMARK_OPT_UNSAFE,
        extensions=["table", "autolink", "strikethrough", "tagfilter"],
    )


def check(document: Path) -> int:
    from playwright.sync_api import sync_playwright

    page_path = document.parent / "_rendered_readme.html"
    page_path.write_text(
        '<!doctype html><meta charset="utf-8"><body style="margin:0;background:#0d1117;'
        'color:#d7dce3;font-family:system-ui">' + render(document.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1100, "height": 900})
            page.goto(f"file://{page_path.resolve()}")
            page.wait_for_timeout(1200)
            images = page.evaluate(
                """() => Array.from(document.querySelectorAll('img')).map(image => ({
                    src: image.getAttribute('src'),
                    naturalWidth: image.naturalWidth,
                    naturalHeight: image.naturalHeight,
                }))"""
            )
            tables = page.evaluate("() => document.querySelectorAll('table').length")
            browser.close()
    finally:
        page_path.unlink(missing_ok=True)

    local = [image for image in images if not image["src"].startswith("http")]
    broken = [image for image in local if not image["naturalWidth"]]
    print(f"{len(images)} image(s), {len(local)} local, {tables} table(s) rendered")
    for image in local:
        state = (
            f"{image['naturalWidth']}x{image['naturalHeight']}"
            if image["naturalWidth"]
            else "BROKEN"
        )
        print(f"  {state:>12}  {image['src']}")
    if broken:
        print(f"\n{len(broken)} image(s) do not resolve, which a reviewer sees first")
        return 1
    if not local:
        print("\nno local images in the document, which is itself worth knowing")
        return 1
    print("\nevery local image in the README resolves")
    return 0


def main() -> int:
    return check(README)


if __name__ == "__main__":
    sys.exit(main())
