"""
Media tools for Stage 4 (scene production).

  search_photos      -> find openly licensed images on Openverse (free, no API key),
                        throttled and cached because of Openverse's rate limits
  download           -> save a photo to disk (falls back to a smaller copy)
  text_to_speech     -> narration MP3 with edge-tts (free, no API key)
  audio_duration     -> length of an MP3 in seconds
  placeholder_image  -> a plain title card when no photo is found

These are plain functions (not @tool): the scene subgraph calls them as fixed
workflow steps. Only the choice of the BEST photo is left to the LLM.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import textwrap
import threading
import time
from pathlib import Path

import edge_tts
import requests
from mutagen.mp3 import MP3
from PIL import Image, ImageDraw, ImageFont

OPENVERSE_API = "https://api.openverse.org/v1/images/"
HEADERS = {"User-Agent": "documentary-agent/0.1 (learning project)"}
# Wikimedia is unreachable on some networks, so we skip its images by default.
# If Wikimedia works for you (e.g. on a VPN), set OPENVERSE_EXCLUDE= (empty) in .env.
EXCLUDED_SOURCES = os.getenv("OPENVERSE_EXCLUDE", "wikimedia")
# Narrator voices you have TESTED with edge-tts. The agent may only choose from these.
# Add a voice here only after it worked in voice_test.py.
VOICES = {
    "en-US-AndrewMultilingualNeural": "Warm American male, multilingual: pronounces names "
                                      "from any region well. The default for general or mixed topics.",
    "en-GB-RyanNeural": "British male, classic documentary narrator. British and European "
                        "history, nature, science.",
    "en-IN-PrabhatNeural": "Indian English male. Topics about South Asia "
                           "(India, Pakistan, Bangladesh, Sri Lanka, Nepal).",
}
DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
# Documentary delivery: a bit slower and deeper than the voices' chatty default.
NARRATION_RATE = os.getenv("NARRATION_RATE", "-15%")
NARRATION_PITCH = os.getenv("NARRATION_PITCH", "-8Hz")
VIDEO_SIZE = (1920, 1080)

# Openverse allows anonymous clients 20 requests/minute and 200/day.
SECONDS_BETWEEN_SEARCHES = 3.2               # ~19 per minute, shared by ALL threads
CACHE_DIR = Path(".cache") / "openverse"     # every search result is saved here


class RateLimited(Exception):
    """Openverse answered 429 Too Many Requests."""


_throttle_lock = threading.Lock()
_last_search = 0.0


def _wait_for_turn() -> None:
    """
    Space out requests across ALL threads. Scenes are produced in parallel, so
    without a shared lock five scenes would hit the API at the same moment.
    """
    global _last_search
    with _throttle_lock:
        wait = _last_search + SECONDS_BETWEEN_SEARCHES - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_search = time.monotonic()


# --- Images ---------------------------------------------------------------

def _license_name(r: dict) -> str:
    lic = (r.get("license") or "").upper()
    if lic in ("CC0", "PDM"):          # public domain: no version
        return lic
    return f"CC {lic} {r.get('license_version') or ''}".strip()


def search_photos(query: str, n: int = 5, kind: str = "photo") -> list[dict]:
    """
    Search Openverse (openly licensed images from Flickr, museums and many other sources).
    Returns up to n images as {id, alt, url, thumb, author, license, page}.

    kind="photo"   -> wide photos (places and buildings today)
    kind="artwork" -> paintings, miniatures, drawings, maps (any shape: many old
                      paintings are portrait; Stage 5 will fit them onto the 16:9 frame)

    Short queries work best ("Badshahi Mosque", not a whole sentence).
    Results are cached on disk, so repeating a search costs no API request.
    Raises RateLimited when Openverse refuses because of too many requests.
    """
    key = hashlib.sha1(f"{kind}|{n}|{query.strip().lower()}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    params = {"q": query, "page_size": n, "excluded_source": EXCLUDED_SOURCES}
    if kind == "artwork":
        params["category"] = "digitized_artwork,illustration"
    else:
        params["aspect_ratio"] = "wide"          # landscape photos only, for 16:9 video

    _wait_for_turn()
    resp = requests.get(OPENVERSE_API, headers=HEADERS, timeout=20, params=params)
    if resp.status_code == 429:
        raise RateLimited("Openverse rate limit reached (anonymous: 20/minute, 200/day)")
    resp.raise_for_status()
    photos = [
        {
            "id": r.get("id"),
            "alt": r.get("title") or query,      # the photo's title on its original site
            "url": r.get("url"),                 # full-size image on the original site
            "thumb": r.get("thumbnail"),         # smaller copy served by Openverse itself
            "author": r.get("creator") or "Unknown",
            "license": _license_name(r),
            "page": r.get("foreign_landing_url") or "",   # original photo page, for credits
        }
        for r in resp.json().get("results", [])
        if r.get("url")
    ]
    # Only SUCCESSFUL responses are cached (an empty list too: "nothing exists" is an answer)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(photos), encoding="utf-8")
    return photos


def download(url: str, path: Path, fallback_url: str | None = None) -> Path:
    """Download an image. If the original host fails, try the fallback (e.g. Openverse's thumbnail)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        if not fallback_url:
            raise
        resp = requests.get(fallback_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def placeholder_image(text: str, path: Path) -> Path:
    """Dark title card with the text centered. Used when no photo can be found."""
    img = Image.new("RGB", VIDEO_SIZE, (20, 24, 33))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 56)       # Windows
    except OSError:
        font = ImageFont.load_default(size=56)           # anywhere else
    wrapped = textwrap.fill(text, width=45)
    center = (VIDEO_SIZE[0] // 2, VIDEO_SIZE[1] // 2)
    draw.multiline_text(center, wrapped, font=font, fill=(235, 235, 235), anchor="mm", align="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)
    return path


def image_preview_b64(path: Path, max_side: int = 512) -> str:
    """Small JPEG of an image as base64, for the LLM to look at (smaller image = fewer tokens)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# --- Audio ----------------------------------------------------------------

def text_to_speech(text: str, path: Path, voice: str = DEFAULT_VOICE,
                   rate: str = NARRATION_RATE, pitch: str = NARRATION_PITCH) -> Path:
    """Write narration to an MP3. edge-tts is async, so we run it in its own event loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(path)))
    return path


def audio_duration(path: Path) -> float:
    return float(MP3(str(path)).info.length)