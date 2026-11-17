"""
Stage 5: turn a production manifest into an MP4.

    title card  ->  scene 1  ->  scene 2  ->  ...  ->  credits

Inside a scene, its shots share the narration time equally and cross-fade into
each other. Every shot gets camera moves (the "Ken Burns effect": zooms and pans
over a still image); a shot longer than MAX_MOVE_SECONDS gets several different
moves, so even a scene with one image doesn't look frozen. Images that aren't 16:9 (e.g. portrait paintings) are shown whole,
on top of a blurred, darkened copy of themselves that fills the frame.

This module is plain Python, no LLM: rendering is a fixed workflow step.
All the image work is done with Pillow; moviepy only assembles and encodes.
"""

import bisect
import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np
from moviepy import AudioFileClip, VideoClip, concatenate_videoclips
from proglog import ProgressBarLogger
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ZOOM = 1.35            # stills are prepared this much bigger than the frame, for zooms and pans
MAX_MOVE_SECONDS = 6.0 # longest a single camera move lasts before the next one starts
CROSSFADE = 0.6        # seconds a shot takes to fade into the next one
SCENE_FADE = 0.3       # fade from/to black at the start/end of each scene
SCENE_PAD = 0.7        # silence after each scene's narration (seconds)
TITLE_SECONDS = 4.0
CREDITS_SECONDS = 5.0
CREDIT_LINES_PER_CARD = 9


# --- Loading ----------------------------------------------------------------

def _find(path_str: str, manifest_dir: Path) -> Path:
    """Paths in the manifest are relative to where the agent ran. If the file
    isn't there, look next to the manifest (works for moved folders and for
    Windows paths read on another system)."""
    path = Path(path_str)
    if path.exists():
        return path
    return manifest_dir / Path(path_str.replace("\\", "/")).name


def load_manifest(manifest_path: Path) -> dict:
    """Read a manifest. Accepts the main graph's format ({title, scenes, credits})
    and produce_scenes.py's older format (just a list of scenes)."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"title": Path(manifest_path).parent.parent.name.replace("-", " ").title(),
                "scenes": data}
    data.setdefault("credits", [shot["credit"] for s in data["scenes"]
                                for shot in s.get("shots", []) if shot.get("credit")])
    data["scenes"] = sorted(data["scenes"], key=lambda s: s["scene_number"])
    return data


# --- Images -----------------------------------------------------------------

def _font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size)        # Windows
    except OSError:
        return ImageFont.load_default(size=size)            # anywhere else


def _cover(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale and crop so the image completely fills `size`."""
    scale = max(size[0] / img.width, size[1] / img.height)
    resized = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.LANCZOS)
    left = (resized.width - size[0]) // 2
    top = (resized.height - size[1]) // 2
    return resized.crop((left, top, left + size[0], top + size[1]))


def prepare_still(path: Path, frame: tuple[int, int]) -> tuple[Image.Image, bool]:
    """
    Turn any image into a still that's ZOOM times bigger than the video frame,
    ready for camera moves. Returns (still, fitted): fitted=True when the image
    was shown whole on a blurred background (gets painting-style moves).
      - roughly 16:9 images: scaled and cropped to fill the frame
      - portrait or very wide images (paintings, manuscripts, maps): shown whole
        over a blurred, darkened copy of themselves
    """
    big = (round(frame[0] * ZOOM), round(frame[1] * ZOOM))
    with Image.open(path) as src:
        img = src.convert("RGB")
    ratio = (img.width / img.height) / (frame[0] / frame[1])

    if 0.8 <= ratio <= 1.3:
        return _cover(img, big), False

    background = _cover(img, big).filter(ImageFilter.GaussianBlur(radius=big[0] // 40))
    background = ImageEnhance.Brightness(background).enhance(0.4)
    scale = min(big[0] * 0.92 / img.width, big[1] * 0.92 / img.height)
    fg = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    background.paste(fg, ((big[0] - fg.width) // 2, (big[1] - fg.height) // 2))
    return background, True


def text_card(lines: list[str], frame: tuple[int, int], font_px: int,
              background: Image.Image | None = None) -> Image.Image:
    """Centered text on a dark (or blurred-photo) background, ZOOM times frame size."""
    big = (round(frame[0] * ZOOM), round(frame[1] * ZOOM))
    if background is None:
        card = Image.new("RGB", big, (16, 18, 24))
    else:
        card = background.filter(ImageFilter.GaussianBlur(radius=big[0] // 60))
        card = ImageEnhance.Brightness(card).enhance(0.35)
    draw = ImageDraw.Draw(card)
    font = _font(font_px)
    draw.multiline_text((big[0] // 2, big[1] // 2), "\n".join(lines), font=font,
                        fill=(240, 240, 240), anchor="mm", align="center",
                        spacing=font_px // 2)
    return card


# --- Motion -----------------------------------------------------------------
#
# A "move" is one camera movement over a still: (start zoom, end zoom,
# start center, end center). Zoom 1.0 shows the whole still; higher zooms in.
# Centers are fractions of the image (0.5, 0.5 = middle).
#
# Documentaries keep a single image on screen for a long time by giving it
# SEVERAL moves: pan across it, zoom into a detail, pull back. So a long shot is
# split into moves of at most MAX_MOVE_SECONDS, and every move is different.

PHOTO_MOVES = [                                         # for images that fill the frame
    (1.00, 1.15, (0.50, 0.50), (0.50, 0.50)),           # slow zoom in
    (1.25, 1.25, (0.35, 0.50), (0.65, 0.50)),           # pan left -> right
    (1.12, 1.32, (0.50, 0.50), (0.42, 0.40)),           # push in on a detail
    (1.25, 1.25, (0.65, 0.55), (0.35, 0.50)),           # pan right -> left
    (1.28, 1.00, (0.50, 0.45), (0.50, 0.50)),           # pull back to the whole image
]
FITTED_MOVES = [                                        # paintings/maps shown whole on blur
    (1.00, 1.15, (0.50, 0.50), (0.50, 0.50)),           # slow zoom in
    (1.30, 1.30, (0.50, 0.35), (0.50, 0.62)),           # tilt down the painting
    (1.15, 1.35, (0.50, 0.50), (0.50, 0.40)),           # push in on the upper part (faces)
    (1.30, 1.30, (0.50, 0.62), (0.50, 0.38)),           # tilt back up
    (1.30, 1.00, (0.50, 0.50), (0.50, 0.50)),           # pull back to the whole painting
]
CARD_MOVE = (1.00, 1.08, (0.50, 0.50), (0.50, 0.50))    # title and credits: barely moving


def _ease(p: float) -> float:
    """Smoothstep: starts and ends gently instead of moving at a constant speed."""
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def ken_burns_frame(still: Image.Image, frame: tuple[int, int], progress: float,
                    move: tuple = PHOTO_MOVES[0]) -> np.ndarray:
    """One video frame of a camera move. progress goes 0 -> 1 over the move."""
    z0, z1, (x0, y0), (x1, y1) = move
    p = _ease(progress)
    z = z0 + (z1 - z0) * p
    cx, cy = (x0 + (x1 - x0) * p) * still.width, (y0 + (y1 - y0) * p) * still.height
    crop_w, crop_h = still.width / z, still.height / z
    left = min(max(cx - crop_w / 2, 0), still.width - crop_w)     # never leave the image
    top = min(max(cy - crop_h / 2, 0), still.height - crop_h)
    box = (left, top, left + crop_w, top + crop_h)
    return np.asarray(still.resize(frame, Image.BILINEAR, box=box))


def plan_moves(fitted: list[bool], duration: float) -> list[tuple[int, tuple, float, float]]:
    """
    Split a scene into (still index, move, start, end) segments.
    Every still gets an equal share of the time; a share longer than
    MAX_MOVE_SECONDS becomes several different moves over the same still.
    """
    span = duration / len(fitted)
    segments, counter = [], 0
    for i, is_fitted in enumerate(fitted):
        moves = FITTED_MOVES if is_fitted else PHOTO_MOVES
        k = max(1, math.ceil(span / MAX_MOVE_SECONDS))
        for j in range(k):
            start = i * span + j * span / k
            segments.append((i, moves[counter % len(moves)], start, start + span / k))
            counter += 1                          # the next segment always moves differently
    return segments


def still_clip(stills: list[Image.Image], frame: tuple[int, int], duration: float,
               fade: float = SCENE_FADE, fitted: list[bool] | None = None,
               card: bool = False) -> VideoClip:
    """
    A clip that shows `stills` in turn, each with one or more camera moves,
    cross-fading between segments and fading in/out at the ends.
    """
    if card:
        segments = [(0, CARD_MOVE, 0.0, duration)]
    else:
        segments = plan_moves(fitted or [False] * len(stills), duration)
    starts = [s[2] for s in segments]

    def render(seg: tuple, t: float) -> np.ndarray:
        i, move, start, end = seg
        return ken_burns_frame(stills[i], frame, (t - start) / (end - start), move)

    def frame_at(t: float) -> np.ndarray:
        n = max(bisect.bisect_right(starts, t) - 1, 0)
        img = render(segments[n], t)

        # cross-fade into the next segment during the last CROSSFADE seconds
        until_next = segments[n][3] - t
        if n < len(segments) - 1 and until_next < CROSSFADE:
            nxt = render(segments[n + 1], segments[n + 1][2])
            a = 1 - until_next / CROSSFADE
            img = (img * (1 - a) + nxt * a).astype(np.uint8)

        # fade from / to black at the start and end of the clip
        edge = min(t, duration - t)
        if fade and edge < fade:
            img = (img * max(edge, 0) / fade).astype(np.uint8)
        return img

    return VideoClip(frame_function=frame_at, duration=duration)


class _PercentLogger(ProgressBarLogger):
    """moviepy reports progress through a 'proglog' logger. This one turns the frame
    counter into a callback(percent), called every 5%, e.g. to stream progress to a UI."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback
        self._last = -5

    def bars_callback(self, bar, attr, value, old_value=None):
        if bar == "frame_index" and attr == "index":
            total = self.bars[bar].get("total") or 1
            percent = int(100 * value / total)
            if percent >= self._last + 5:
                self._last = percent
                self._callback(percent)


# --- Assembly ---------------------------------------------------------------

def _credit_lines(manifest: dict, max_len: int = 80) -> list[str]:
    """
    '"Title" by Author (CC BY 2.0) https://...' -> one short line.
    If it's too long, shorten the TITLE only: author and license must always stay
    (crediting them is what the Creative Commons licenses require).
    """
    lines = []
    for credit in manifest["credits"]:
        text = credit.rsplit(" http", 1)[0]
        match = re.match(r'^"(.*)" by (.*)$', text)
        if len(text) > max_len and match:
            title, rest = match.groups()
            room = max(10, max_len - len(rest) - 9)
            text = f'"{title[:room].rstrip()}..." by {rest}'
        lines.append(text if len(text) <= max_len + 30 else text[:max_len + 27] + "...")
    return lines


def render_video(manifest_path: Path, out_path: Path | None = None,
                 height: int = 1080, fps: int = 24, logger: str | None = "bar",
                 progress=None) -> Path:
    """
    Render the whole documentary. Returns the path of the MP4.
    progress: optional callback(percent) for live progress (replaces the terminal bar).
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    manifest = load_manifest(manifest_path)
    width = round(height * 16 / 9) // 2 * 2            # H.264 needs even sizes
    frame = (width, height)
    out_path = Path(out_path or manifest_dir / "documentary.mp4")

    clips, audios = [], []

    # 1) Title card over a blurred version of the first image
    first_shot = manifest["scenes"][0]["shots"][0]["image_path"]
    title_bg, _ = prepare_still(_find(first_shot, manifest_dir), frame)
    title_lines = textwrap.wrap(manifest["title"], width=28)
    clips.append(still_clip([text_card(title_lines, frame, height // 13, title_bg)],
                            frame, TITLE_SECONDS, fade=0.8, card=True))

    # 2) Scenes: narration decides the length, shots share it equally
    for scene in manifest["scenes"]:
        audio = AudioFileClip(str(_find(scene["audio_path"], manifest_dir)))
        audios.append(audio)
        prepared = [prepare_still(_find(s["image_path"], manifest_dir), frame) for s in scene["shots"]]
        stills, fitted = [p[0] for p in prepared], [p[1] for p in prepared]
        clip = still_clip(stills, frame, audio.duration + SCENE_PAD, fitted=fitted)
        clips.append(clip.with_audio(audio))

    # 3) Credits, a few lines per card
    lines = _credit_lines(manifest)
    header = ["Images"]
    footer = ["", f"Narration: {manifest.get('voice', 'Microsoft Edge TTS')}",
              "Research and script: AI agent built with LangGraph"]
    pages = [lines[i:i + CREDIT_LINES_PER_CARD] for i in range(0, len(lines), CREDIT_LINES_PER_CARD)] or [[]]
    for n, page in enumerate(pages):
        body = header + [""] + page + (footer if n == len(pages) - 1 else [])
        clips.append(still_clip([text_card(body, frame, height // 26)], frame, CREDITS_SECONDS,
                                fade=0.5, card=True))

    video = concatenate_videoclips(clips)
    try:
        video.write_videofile(
            str(out_path), fps=fps, codec="libx264", audio_codec="aac",
            preset="veryfast", threads=4,
            logger=_PercentLogger(progress) if progress else logger,
            temp_audiofile_path=str(manifest_dir),     # temp file next to the video, not in cwd
        )
    finally:
        video.close()
        for audio in audios:
            audio.close()
    return out_path