"""
Stage 4: the SCENE SUBGRAPH. It produces the media for ONE scene.

        ┌→ find_visuals ─┐
  START ┤                ├→ finish → END
        └→ voiceover   ──┘

find_visuals and voiceover run IN PARALLEL (both start from START), and
`finish` waits for both. The subgraph has its OWN state (SceneState): the main
graph never sees the shots or audio_path separately. It only receives the
finished ProducedScene through `produced_scenes` (see SceneOutput).

A scene is shown as several SHOTS (one image every ~5 seconds of narration),
mixing present-day photos with historical artwork (miniatures, paintings, maps).

Two error strategies, on purpose:
  * find_visuals -> DEGRADE: each shot tries a FALLBACK CHAIN of searches, and every
                    chosen image must pass a VISION CHECK (the LLM looks at it); a shot
                    that still finds nothing is skipped; only if EVERY shot
                    fails does the scene get a title card. Never crashes.
  * voiceover    -> FALLBACK, then RETRY: if the chosen voice fails, try the other
                    approved voices; only if ALL fail, raise so LangGraph's
                    RetryPolicy runs the node again (3 attempts). There's no
                    substitute for narration, so this branch never "degrades".

The narrator voice is chosen ONCE per documentary (choose_voice) and passed in,
so every scene sounds the same.
"""

import operator
import os
import re
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import aiohttp
from edge_tts.exceptions import EdgeTTSException
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import Field

from .llm import get_llm
from .media import (
    RateLimited,
    audio_duration,
    download,
    image_preview_b64,
    placeholder_image,
    search_photos,
    text_to_speech,
)
from .prompts import IMAGE_CHECK_PROMPT, IMAGE_PICK_PROMPT, SHOT_PLAN_PROMPT, VOICE_PROMPT
from .state import LLMOutput, ProducedScene, Scene, Shot
from .languages import get_language
from .styles import get_style

# Errors a voice SERVICE can cause. Only these trigger the fallback voices; anything
# else (e.g. a TypeError) is a bug in our code and should fail loudly, right away.
TTS_ERRORS = (EdgeTTSException, aiohttp.ClientError, OSError, TimeoutError)

WORDS_PER_SECOND = 140 / 60      # same speaking rate the script writer uses
SECONDS_PER_SHOT = 5             # change the picture about every 5 seconds
MAX_SHOTS = 4
# The voice service sometimes fails for a few seconds. Retry the SAME voice first,
# so the whole documentary keeps one narrator; switch voices only as a last resort.
SAME_VOICE_ATTEMPTS = 3
VOICE_RETRY_WAIT = 2.0          # seconds before the 2nd attempt, doubled before the 3rd

MIN_SHOT_SECONDS = 3             # shots closer together than this are dropped (too hectic)
MAX_PICKS_PER_SHOT = 2           # LLM picker calls per shot, to keep costs predictable
# Vision check: the LLM looks at each chosen image. Set VERIFY_IMAGES=false in .env to save cost.
VERIFY_IMAGES = os.getenv("VERIFY_IMAGES", "true").lower() != "false"


# --- LLM output models --------------------------------------------------------

class ShotIdea(LLMOutput):
    anchor: str = Field(description="The exact words (3-8) from the narration at the moment "
                                    "this shot should appear, copied exactly")
    description: str = Field(description="What this shot shows, one short sentence")
    kind: Literal["photo", "artwork"]
    queries: list[str] = Field(description="2-3 short search queries, most specific first")


class ShotPlan(LLMOutput):
    shots: list[ShotIdea]
    fallback_queries: list[str] = Field(
        description="1-2 broad queries for the scene's overall subject, used when a shot finds nothing")


class ImageChoice(LLMOutput):
    number: int = Field(description="Number of the best image, or 0 if none fit the shot")
    reason: str = Field(description="One short sentence")


class ImageCheck(LLMOutput):
    shows: str = Field(description="What the image actually shows, one short sentence")
    matches: bool = Field(description="True only if the image fits the shot")


class VoiceChoice(LLMOutput):
    voice: str = Field(description="The exact voice name from the list")
    reason: str = Field(description="One short sentence")


# --- Subgraph state -----------------------------------------------------------

class SceneState(TypedDict, total=False):
    # input (sent by the main graph, one scene per run)
    scene: Scene
    media_dir: str
    voice: str              # chosen ONCE for the whole documentary (see choose_voice)
    notes: str              # director's notes when a scene is redone (production review)
    style: str              # style preset: decides the narrator's speed and pitch
    language: str           # narration language: decides which voices may be used
    # filled by the parallel branches
    shots: list[Shot]
    audio_path: str
    duration_sec: float
    voice_used: str
    problems: Annotated[list[str], operator.add]   # both branches may add problems at once
    # output
    produced_scenes: list[ProducedScene]


class SceneOutput(TypedDict):
    """What the subgraph hands back to the main graph: ONLY this key."""
    produced_scenes: list[ProducedScene]


# --- Helpers ------------------------------------------------------------------

def progress(scene_number: int, message: str) -> None:
    """
    Send a live progress message to whoever is streaming the graph
    (stream_mode="custom"). Outside a stream, e.g. produce_scenes.py, it goes nowhere.
    """
    get_stream_writer()({"scene": scene_number, "msg": message})


def choose_voice(topic: str, voice_hint: str = "", language: str | None = None) -> tuple[str, str]:
    """
    Pick ONE narrator for the whole documentary. Returns (voice, reason).
    Called once before the scenes are produced, so every scene sounds the same.
    Only voices approved for the narration's LANGUAGE can be chosen.
    """
    lang = get_language(language)
    forced = os.getenv("TTS_VOICE")
    if forced in lang.voices:                    # the human's choice wins (if it speaks this language)
        return forced, "set by TTS_VOICE in .env"

    catalog = "\n".join(
        f"- {name}{' (default)' if name == lang.default_voice else ''}: {desc}"
        for name, desc in lang.voices.items()
    )
    choice = get_llm().with_structured_output(VoiceChoice).invoke([
        SystemMessage(VOICE_PROMPT.format(catalog=catalog)),
        HumanMessage(f"Documentary topic: {topic}"
                     + (f"\nThe style needs {voice_hint}." if voice_hint else "")),
    ])
    if choice.voice not in lang.voices:          # never trust a name the LLM made up
        return lang.default_voice, f"'{choice.voice}' is not an approved voice, using the default"
    return choice.voice, choice.reason


def _tokens(text: str) -> list[str]:
    """Words of a text in ANY script: \\w also matches Urdu, Arabic, Hindi... letters.
    (A-z only would find no words at all in Urdu, so every anchor would silently fail.)"""
    return re.findall(r"\w+", text.lower())


def anchor_start(narration_tokens: list[str], anchor: str) -> int | None:
    """
    Word position where `anchor` appears in the narration, or None.
    The LLM is asked to copy the words exactly; CODE checks that it did.
    Falls back to the anchor's first two words (LLMs sometimes change a word).
    """
    words = _tokens(anchor)
    for length in (len(words), 2):
        head = words[:length]
        if not head:
            break
        for i in range(len(narration_tokens) - len(head) + 1):
            if narration_tokens[i:i + len(head)] == head:
                return i
    return None


def place_shots(narration: str, ideas: list) -> list[tuple[float, object]]:
    """
    Decide WHEN each planned shot appears: (start fraction 0-1, idea), in narration order.
      - start = where its anchor is spoken (word position / total words; the narrator
        speaks at a steady pace, so word position ~ time)
      - an anchor that isn't in the narration gets an evenly spaced position
      - shots closer than MIN_SHOT_SECONDS to the previous one are dropped
      - the first shot always starts at 0, so the scene never begins without a picture
    """
    tokens = _tokens(narration)
    total = max(len(tokens), 1)
    placed = []
    for i, idea in enumerate(ideas):
        pos = anchor_start(tokens, idea.anchor)
        if pos is None:
            pos = round(i * total / len(ideas))
        placed.append((pos / total, idea))
    placed.sort(key=lambda p: p[0])

    min_gap = MIN_SHOT_SECONDS * WORDS_PER_SECOND / total     # in fractions of the narration
    kept = []
    for start, idea in placed:
        if kept and start - kept[-1][0] < min_gap:
            continue
        kept.append((start, idea))
    if kept:
        kept[0] = (0.0, kept[0][1])
    return kept


def shot_count(narration: str) -> int:
    """Estimate how many shots a scene needs from its length. Code decides, not the LLM."""
    seconds = len(narration.split()) / WORDS_PER_SECOND
    return max(1, min(MAX_SHOTS, round(seconds / SECONDS_PER_SHOT)))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def too_similar(title: str, used_titles: list[str], threshold: float = 0.5) -> bool:
    """
    True if this image's title shares most of its words with an image already used
    in the scene (e.g. "Taj Mahal, Agra" vs "Taj Mahal at dawn"): probably the same
    subject from another angle. Code enforces variety; the prompt only asks for it.
    """
    words = _words(title)
    for used in used_titles:
        other = _words(used)
        if words and other and len(words & other) / len(words | other) >= threshold:
            return True
    return False


def verify_image(path: Path, idea: ShotIdea, title: str = "") -> tuple[bool, str]:
    """
    VISION CHECK: show the downloaded image to the LLM and ask if it really fits.
    The picker only reads titles; a photo titled "mumtaz mahal" can show anyone.
    The checker gets the title TOO: nobody can recognise a 400-year-old painting
    of Shah Jahan from the picture alone, but its museum title says who it is.
    Returns (matches, what the image actually shows).
    """
    if not VERIFY_IMAGES:
        return True, "not verified (VERIFY_IMAGES=false)"
    check = get_llm().with_structured_output(ImageCheck).invoke([
        SystemMessage(IMAGE_CHECK_PROMPT),
        HumanMessage(content=[
            {"type": "text", "text": f"The narrator is saying: “{idea.anchor}”\n"
                                     f"The shot should show ({idea.kind}): {idea.description}\n"
                                     f"The image's title at its source: {title or '(none)'}"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_preview_b64(path)}"}},
        ]),
    ])
    return check.matches, check.shows


def find_image(idea: ShotIdea, narration: str, fallback_queries: list[str],
               used_urls: set[str], used_titles: list[str],
               path: Path) -> tuple[dict | None, str, list[str]]:
    """
    Try a FALLBACK CHAIN of searches until an image passes BOTH checks:
      1. the shot's own queries, in its own kind (photo or artwork)
      2. its first query in the OTHER kind
      3. the scene's broad fallback query, as a photo
    For each search: the LLM picks a candidate by TITLE, the image is downloaded
    to `path`, then the LLM LOOKS at it (vision check).
    Returns (image or None, kind actually used, reasons for every failed attempt).
    """
    other = "photo" if idea.kind == "artwork" else "artwork"
    attempts = [(q, idea.kind) for q in idea.queries[:2]]
    attempts += [(q, other) for q in idea.queries[:1]]
    attempts += [(q, "photo") for q in fallback_queries[:1]]

    reasons, picks, rejected = [], 0, set()
    for query, kind in attempts:
        try:
            results = search_photos(query, kind=kind)
        except RateLimited:
            reasons.append("rate limited by Openverse")
            break                       # every further search would fail too
        except Exception as e:  # noqa: BLE001
            reasons.append(f"'{query}' ({kind}): search error {type(e).__name__}")
            continue

        candidates = [r for r in results
                      if r["url"] not in used_urls | rejected
                      and not too_similar(r["alt"], used_titles)]
        if not candidates:
            reasons.append(f"'{query}' ({kind}): " + ("no results" if not results else "only repeats"))
            continue
        if picks == MAX_PICKS_PER_SHOT:
            reasons.append("picker limit reached")
            break

        # a) pick by title
        picks += 1
        listing = "\n".join(f"[{i}] {c['alt']}" for i, c in enumerate(candidates, 1))
        choice = get_llm().with_structured_output(ImageChoice).invoke([
            SystemMessage(IMAGE_PICK_PROMPT),
            HumanMessage(f"Narration: {narration}\nThe narrator is saying: “{idea.anchor}”\n"
                         f"This shot should show ({kind}): {idea.description}\n\nImages:\n{listing}"),
        ])
        if not 1 <= choice.number <= len(candidates):
            reasons.append(f"'{query}' ({kind}): picker rejected {len(candidates)} ({choice.reason})")
            continue
        image = candidates[choice.number - 1]

        # b) download
        try:
            download(image["url"], path, fallback_url=image.get("thumb"))
        except Exception as e:  # noqa: BLE001
            reasons.append(f"'{image['alt'][:40]}': download failed ({type(e).__name__})")
            rejected.add(image["url"])
            continue

        # c) look at it
        try:
            matches, shows = verify_image(path, idea, image["alt"])
        except Exception as e:  # noqa: BLE001  (checker unavailable: keep the image, but say so)
            matches, shows = True, f"not verified ({type(e).__name__})"
        if matches:
            return image, kind, reasons
        reasons.append(f"'{image['alt'][:40]}': vision check rejected it (shows: {shows})")
        rejected.add(image["url"])
        path.unlink(missing_ok=True)

    return None, idea.kind, reasons


# --- Nodes --------------------------------------------------------------------

def find_visuals(state: SceneState) -> dict:
    scene = state["scene"]
    media_dir = Path(state["media_dir"])
    n = shot_count(scene.narration)

    # 1) LLM as picture editor: plan the shots (following the director's notes on a redo)
    request = (f"Narration: {scene.narration}\n"
               f"Main visual idea from the script: {scene.visual_description}")
    if state.get("notes"):
        request += f"\n\nDirector's notes for this scene (follow them): {state['notes']}"
    plan = get_llm().with_structured_output(ShotPlan).invoke([
        SystemMessage(SHOT_PLAN_PROMPT.format(n=n)),
        HumanMessage(request),
    ])

    # 2) Decide WHEN each shot appears (at its anchor words), then fill them
    #    (code enforces the limit, the timing, "no repeats" and "no near-duplicates")
    placed = place_shots(scene.narration, plan.shots[:MAX_SHOTS])
    shots, problems, used_urls, used_titles = [], [], set(), []
    progress(scene.scene_number, f"planned {len(placed)} shot(s)")
    for i, (start, idea) in enumerate(placed, 1):
        progress(scene.scene_number, f"shot {i} at “{idea.anchor[:40]}”: looking for {idea.kind}")
        path = media_dir / f"scene_{scene.scene_number:02d}_shot_{len(shots) + 1}.jpg"
        image, kind, reasons = find_image(idea, scene.narration, plan.fallback_queries,
                                          used_urls, used_titles, path)
        if image is None:
            progress(scene.scene_number, f"shot {i}: nothing usable found")
            problems.append(f"scene {scene.scene_number} shot {i} ('{idea.description[:50]}'): "
                            + "; ".join(reasons))
            continue
        used_urls.add(image["url"])
        used_titles.append(image["alt"])
        progress(scene.scene_number, f"shot {i}: ✓ {image['alt'][:60]}")
        shots.append(Shot(
            image_path=str(path), kind=kind, description=idea.description,
            credit=f"\"{image['alt']}\" by {image['author']} ({image['license']}) {image['page']}",
            anchor=idea.anchor, start=start,
        ))

    # 3) Every shot failed: one title card so the scene still has a picture
    if not shots:
        path = media_dir / f"scene_{scene.scene_number:02d}_shot_1.jpg"
        placeholder_image(scene.visual_description, path)
        shots = [Shot(image_path=str(path), kind="title_card", description=scene.visual_description)]
        problems.append(f"scene {scene.scene_number}: no images found, used a title card")

    return {"shots": shots, "problems": problems}


def voiceover(state: SceneState) -> dict:
    """
    Record the narration with the documentary's voice.
      1. Try the CHOSEN voice up to SAME_VOICE_ATTEMPTS times, with short waits:
         the free voice service often fails for just a few seconds.
      2. Only then fall back to the other approved voices of the language, so the
         run doesn't fail (the quality check then flags the different narrator).
      3. If every voice fails, raise: RetryPolicy waits and runs this node again.
    """
    scene = state["scene"]
    path = Path(state["media_dir"]) / f"scene_{scene.scene_number:02d}.mp3"
    lang = get_language(state.get("language"))
    chosen = state.get("voice") or lang.default_voice
    order = [chosen] + [v for v in lang.voices if v != chosen]     # fall back within the language

    style = get_style(state.get("style"))
    rate, pitch = lang.speech or (style.rate, style.pitch)   # some languages need their own settings
    attempts = [chosen] * SAME_VOICE_ATTEMPTS + order[1:]

    progress(scene.scene_number, f"recording narration ({chosen})")
    errors = []
    for n, voice in enumerate(attempts):
        if 0 < n < SAME_VOICE_ATTEMPTS:              # retries of the chosen voice: wait a little
            wait = VOICE_RETRY_WAIT * 2 ** (n - 1)
            progress(scene.scene_number, f"voice failed, retrying {chosen} in {wait:.0f}s")
            time.sleep(wait)
        try:
            text_to_speech(scene.narration, path, voice=voice, rate=rate, pitch=pitch)
        except TTS_ERRORS as e:
            errors.append(f"{voice}: {type(e).__name__}")
            continue
        result = {"audio_path": str(path), "duration_sec": audio_duration(path), "voice_used": voice}
        if voice != chosen:
            result["problems"] = [f"scene {scene.scene_number}: voice {chosen} failed "
                                  f"{SAME_VOICE_ATTEMPTS} times, used {voice}"]
        return result

    # ConnectionError on purpose: RetryPolicy's default only retries network-type errors.
    # A RuntimeError would be treated as a bug in our code and NOT retried.
    raise ConnectionError(f"scene {scene.scene_number}: every voice failed ({'; '.join(errors)})")


def finish(state: SceneState) -> dict:
    """Runs once BOTH branches are done. Packs everything into one ProducedScene."""
    scene = state["scene"]
    return {"produced_scenes": [ProducedScene(
        scene_number=scene.scene_number,
        narration=scene.narration,
        shots=state["shots"],
        audio_path=state["audio_path"],
        duration_sec=state["duration_sec"],
        voice=state.get("voice_used", ""),
        problems=state.get("problems", []),
    )]}


# --- Build --------------------------------------------------------------------

def build_scene_graph():
    builder = StateGraph(SceneState, output_schema=SceneOutput)

    builder.add_node("find_visuals", find_visuals)
    builder.add_node("voiceover", voiceover,
                     retry_policy=RetryPolicy(max_attempts=3, initial_interval=2.0))
    builder.add_node("finish", finish)

    builder.add_edge(START, "find_visuals")   # two edges out of START =
    builder.add_edge(START, "voiceover")      # two branches running in parallel
    builder.add_edge(["find_visuals", "voiceover"], "finish")   # wait for BOTH
    builder.add_edge("finish", END)

    # No checkpointer here: inside the main graph, the subgraph uses the parent's.
    return builder.compile()