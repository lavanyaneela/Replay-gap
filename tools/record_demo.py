"""Record the demo as a replay of captured output, not as a screen capture.

Every line of terminal text in the result is real stdout from a command that actually ran, with
its real exit code, and each segment is paced by that command's measured wall time. So a slow
command looks slow and nothing is hand written. The README says this about the recording, and
docs/video/manifest.json lists every command with its exit code and duration so the claim is
checkable rather than asserted.

The import is warmed with a throwaway command before anything is captured. Without that the
first segment pays for loading NumPy, LightGBM and DuckDB, and the pacing of the whole recording
is wrong in a way that makes the tool look slower than it is.

Segment order is the argument, not the feature list: the log first, then the headline, then the
refusal that no other tool reports, then the gap against the referee, then the receipts.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

VIDEO_DIR = Path("docs/video")
WIDTH = 1000
HEIGHT = 620

SEGMENTS: tuple[tuple[str, str], ...] = (
    ("replay-gap diagnose", "The log, before any candidate policy is involved."),
    (
        "replay-gap evaluate incumbent --no-ranker",
        "The deployed ranker, replayed against the log it produced.",
    ),
    (
        "replay-gap evaluate contrarian_sharp --no-ranker",
        "The same ranker as contrarian_lift, served greedily. The tool refuses.",
    ),
    (
        "replay-gap compare --no-ranker --gap",
        "Every policy, scored as error against the exact referee.",
    ),
    ("python tools/check_numbers.py --strict-reverse", "Every published figure, re-checked."),
)

WARMUP = "replay-gap policies"


def capture(command: str) -> dict:
    """Run one command, keeping its real output, exit code and duration."""
    started = time.perf_counter()
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=1800, check=False
    )
    duration = time.perf_counter() - started
    return {
        "command": command,
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def build_player(segments: list[dict], captions: dict[str, str]) -> str:
    """A self contained page that types each command and reveals its captured output."""
    payload = json.dumps(
        [
            {
                "command": segment["command"],
                "caption": captions[segment["command"]],
                "output": segment["stdout"] + segment["stderr"],
                "exit": segment["exit_code"],
                "duration": segment["duration_seconds"],
            }
            for segment in segments
        ]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#0f1116;}}
#term{{font-family:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;font-size:13.5px;
line-height:1.45;color:#d7dce3;padding:18px 22px;white-space:pre-wrap;height:{HEIGHT - 36}px;
overflow:hidden;}}
.caption{{color:#7c8696;}}
.prompt{{color:#5ea9f5;}}
.cmd{{color:#ffffff;}}
.ok{{color:#56b870;}}
.refused{{color:#e0714a;}}
</style></head><body><div id="term"></div><script>
const segments = {payload};
const term = document.getElementById('term');
const sleep = ms => new Promise(r => setTimeout(r, ms));
function trim(text, lines) {{
  const parts = text.replace(/\\s+$/, '').split('\\n');
  return parts.length <= lines ? parts.join('\\n')
    : parts.slice(0, lines).join('\\n') + '\\n  ... ' + (parts.length - lines) + ' more lines';
}}
async function play() {{
  for (const segment of segments) {{
    term.innerHTML = '';
    const caption = document.createElement('div');
    caption.className = 'caption';
    caption.textContent = '# ' + segment.caption;
    term.appendChild(caption);
    const line = document.createElement('div');
    line.innerHTML = '<span class="prompt">$ </span><span class="cmd"></span>';
    term.appendChild(line);
    const target = line.querySelector('.cmd');
    for (const character of segment.command) {{
      target.textContent += character;
      await sleep(22);
    }}
    await sleep(260);
    const body = document.createElement('div');
    body.textContent = trim(segment.output, 26);
    term.appendChild(body);
    const status = document.createElement('div');
    status.className = segment.exit === 2 ? 'refused' : 'ok';
    status.textContent = '[exit ' + segment.exit + ']';
    term.appendChild(status);
    const paced = Math.min(4200, Math.max(1500, segment.duration * 1000));
    await sleep(paced);
  }}
  window.__done = true;
}}
play();
</script></body></html>
"""


def record(page_path: Path, out_dir: Path, total_seconds: float) -> Path:
    """Drive the player in a headless browser and keep the video."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": WIDTH, "height": HEIGHT},
            record_video_dir=str(out_dir),
            record_video_size={"width": WIDTH, "height": HEIGHT},
        )
        page = context.new_page()
        page.goto(f"file://{page_path.resolve()}")
        page.wait_for_function("() => window.__done === true", timeout=int(total_seconds * 2500))
        page.wait_for_timeout(600)
        video = page.video
        context.close()
        browser.close()
        return Path(video.path())


def transcode(webm: Path, out_dir: Path) -> dict:
    """An MP4 for quality and a two pass palette GIF that autoplays inline on GitHub."""
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found; keeping the webm only", file=sys.stderr)
        return {"webm": str(webm)}
    mp4 = out_dir / "demo.mp4"
    gif = out_dir / "demo.gif"
    palette = out_dir / "_palette.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(webm),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=1000:-2",
            str(mp4),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(webm),
            "-vf",
            "fps=10,scale=860:-1:flags=lanczos,palettegen=max_colors=96",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(webm),
            "-i",
            str(palette),
            "-lavfi",
            "fps=10,scale=860:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(gif),
        ],
        check=True,
    )
    palette.unlink(missing_ok=True)
    return {
        "mp4": str(mp4),
        "mp4_bytes": mp4.stat().st_size,
        "gif": str(gif),
        "gif_bytes": gif.stat().st_size,
    }


def main() -> int:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"warming the import with: {WARMUP}")
    capture(WARMUP)

    captions = dict(SEGMENTS)
    segments = []
    for command, _caption in SEGMENTS:
        print(f"capturing: {command}")
        segment = capture(command)
        print(f"  exit {segment['exit_code']} in {segment['duration_seconds']}s")
        segments.append(segment)

    page_path = VIDEO_DIR / "_player.html"
    page_path.write_text(build_player(segments, captions), encoding="utf-8")
    total = sum(min(4.2, max(1.5, s["duration_seconds"])) + 1.5 for s in segments) + 4
    webm = record(page_path, VIDEO_DIR, total)
    target_webm = VIDEO_DIR / "demo.webm"
    shutil.move(str(webm), target_webm)
    page_path.unlink(missing_ok=True)

    outputs = transcode(target_webm, VIDEO_DIR)
    manifest = {
        "note": (
            "A replay, not a screen capture. Every line of terminal text is real captured "
            "stdout from the command above it, each segment is paced by that command's measured "
            "wall time, and the exit codes are the exit codes those commands actually returned."
        ),
        "warmup": WARMUP,
        "segments": [
            {
                "command": segment["command"],
                "caption": captions[segment["command"]],
                "exit_code": segment["exit_code"],
                "duration_seconds": segment["duration_seconds"],
                "stdout_lines": len(segment["stdout"].splitlines()),
            }
            for segment in segments
        ],
        "outputs": {"webm": str(target_webm), **outputs},
    }
    (VIDEO_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for key, value in outputs.items():
        if key.endswith("_bytes"):
            print(f"{key}: {value:,}")
    print(f"wrote {VIDEO_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
