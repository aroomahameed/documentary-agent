"""
Stage 8: the Documentary Agent's tools as an MCP server.

Any MCP client (Claude Desktop, other agents, our own LangGraph agent) can
discover these tools and call them, without importing our Python code:

  tools      search_images, download_image, list_voices, narrate, render_video
  resource   documentaries://videos  (read-only: the finished videos)

    python mcp_server.py          # speaks MCP over stdin/stdout ("stdio" transport)

Rules for a stdio MCP server:
  * NEVER print() to stdout: stdout IS the protocol channel; a stray print breaks it.
  * The server runs an async event loop. Slow, blocking work (network, speech,
    rendering) runs in a WORKER THREAD (anyio.to_thread.run_sync): it keeps the
    server responsive, and edge-tts's asyncio.run() is not allowed inside a running loop.
  * Clients may run it from any folder, so we work relative to THIS file.
  * Tools are callable by an AI: validate every input, and only ever read and
    write inside this project's output/ folder.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # so `doc_agent` imports work from any folder
os.chdir(ROOT)                          # relative paths (.cache, output/) live in the project

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import anyio  # noqa: E402
from mcp.server.fastmcp import Context, FastMCP  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from doc_agent.languages import LANGUAGES, get_language  # noqa: E402
from doc_agent.media import audio_duration, download, search_photos, text_to_speech  # noqa: E402
from doc_agent.styles import get_style  # noqa: E402
from doc_agent.video import render_video as render_mp4  # noqa: E402

OUTPUT = ROOT / "output"
MCP_FILES = OUTPUT / "mcp"              # everything created through MCP goes here
MAX_NARRATION_CHARS = 3000

server = FastMCP(
    "documentary-tools",
    instructions="Tools from the Documentary Agent: find openly licensed images, "
                 "narrate text with neural voices, and render documentary videos.",
)


def _safe_path(filename: str, ext: str) -> Path:
    """A file inside output/mcp, whatever name the client sends ('../../x' -> 'x')."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(filename).stem).strip("-")[:60] or "file"
    MCP_FILES.mkdir(parents=True, exist_ok=True)
    return MCP_FILES / f"{stem}{ext}"


# --- Tools ------------------------------------------------------------------
# The docstring and type hints ARE the tool's description for the AI client.
# Annotations are hints for clients (e.g. ask before running tools that write files).

@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
async def search_images(query: str, kind: Literal["photo", "artwork"] = "photo",
                        count: int = 5) -> list[dict]:
    """Search openly licensed images (Openverse: Flickr, museums and more).
    Use short queries naming a real thing, e.g. "Badshahi Mosque".
    kind="artwork" prefers paintings, drawings and maps. Returns up to `count`
    (max 10) results with title, url, author, license and page; credit the author
    and license when you use an image."""
    count = max(1, min(count, 10))
    results = await anyio.to_thread.run_sync(lambda: search_photos(query, n=count, kind=kind))
    return [{"title": r["alt"], "url": r["url"], "author": r["author"],
             "license": r["license"], "page": r["page"]} for r in results]


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                         openWorldHint=True))
async def download_image(url: str, filename: str) -> str:
    """Download an image (e.g. a url from search_images) into the server's
    output/mcp folder. Only real images are saved. Returns the saved file's path."""
    if not url.startswith(("https://", "http://")):
        raise ValueError("url must start with http:// or https://")
    path = _safe_path(filename, ".jpg")
    return str(await anyio.to_thread.run_sync(lambda: download(url, path)))


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def list_voices(language: Literal["en", "ur"] = "en") -> dict[str, str]:
    """The narrator voices available for a language ("en" English, "ur" Urdu),
    with a short description of each."""
    return dict(get_language(language).voices)


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                         openWorldHint=True))
async def narrate(text: str, filename: str, language: Literal["en", "ur"] = "en",
                  voice: str = "") -> dict:
    """Turn text into a narrated MP3 in the server's output/mcp folder, spoken in a
    calm documentary style. `voice` must be one of list_voices(language); leave it
    empty for the default. Returns the file's path and length in seconds."""
    lang = get_language(language)
    voice = voice or lang.default_voice
    if voice not in lang.voices:
        raise ValueError(f"Unknown voice '{voice}' for {language}. Use list_voices to see the options.")
    if not text.strip() or len(text) > MAX_NARRATION_CHARS:
        raise ValueError(f"text must be 1-{MAX_NARRATION_CHARS} characters")
    style = get_style("documentary")
    rate, pitch = lang.speech or (style.rate, style.pitch)
    path = _safe_path(filename, ".mp3")
    await anyio.to_thread.run_sync(
        lambda: text_to_speech(text, path, voice=voice, rate=rate, pitch=pitch))
    return {"path": str(path), "seconds": round(audio_duration(path), 1), "voice": voice}


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False))
async def render_video(manifest_path: str, ctx: Context, preview: bool = True) -> str:
    """Render a documentary made by the Documentary Agent into an MP4.
    manifest_path: a manifest.json inside the project's output folder.
    preview=True renders a fast 480p version, False the full 1080p one (minutes).
    Reports progress while rendering. Returns the MP4's path."""
    manifest = Path(manifest_path)
    manifest = (manifest if manifest.is_absolute() else ROOT / manifest).resolve()
    if manifest.name != "manifest.json" or not manifest.is_relative_to(OUTPUT.resolve()):
        raise ValueError("manifest_path must be a manifest.json inside the project's output folder")
    if not manifest.exists():
        raise ValueError(f"No such manifest: {manifest_path}")

    def report(percent: int) -> None:             # called from the render thread...
        anyio.from_thread.run(ctx.report_progress, percent, 100)   # ...back into the server

    out = await anyio.to_thread.run_sync(
        lambda: render_mp4(manifest, height=480 if preview else 1080, progress=report))
    return str(out)


# --- Resources: read-only data a client can look at ---------------------------

@server.resource("documentaries://videos", mime_type="application/json")
def finished_videos() -> str:
    """All rendered documentary videos in the project, newest first."""
    videos = sorted(OUTPUT.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return json.dumps([{"path": str(v.relative_to(ROOT)), "mb": round(v.stat().st_size / 1e6, 1)}
                       for v in videos], indent=2)


if __name__ == "__main__":
    server.run("stdio")