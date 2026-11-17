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
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import Field

from .llm import get_llm
from .media import (
    DEFAULT_VOICE,
    VOICES,
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

WORDS_PER_SECOND = 140 / 60      # same speaking rate the script writer uses
SECONDS_PER_SHOT = 5             # change the picture about every 5 seconds
MAX_SHOTS = 4
MAX_PICKS_PER_SHOT = 2           # LLM picker calls per shot, to keep costs predictable
# Vision check: the LLM looks at each chosen image. Set VERIFY_IMAGES=false in .env to save cost.
VERIFY_IMAGES = os.getenv("VERIFY_IMAGES", "true").lower() != "false"


# --- LLM output models --------------------------------------------------------

class ShotIdea(LLMOutput):
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
    # filled by the parallel branches
    shots: list[Shot]
    audio_path: str
    duration_sec: float
    problems: Annotated[list[str], operator.add]   # both branches may add problems at once
    # output
    produced_scenes: list[ProducedScene]


class SceneOutput(TypedDict):
    """What the subgraph hands back to the main graph: ONLY this key."""
    produced_scenes: list[ProducedScene]


# --- Helpers ------------------------------------------------------------------

def choose_voice(topic: str) -> tuple[str, str]:
    """
    Pick ONE narrator for the whole documentary. Returns (voice, reason).
    Called once before the scenes are produced, so every scene sounds the same.
    """
    forced = os.getenv("TTS_VOICE")
    if forced:                                   # the human's choice always wins
        return forced, "set by TTS_VOICE in .env"

    catalog = "\n".join(
        f"- {name}{' (default)' if name == DEFAULT_VOICE else ''}: {desc}"
        for name, desc in VOICES.items()
    )
    choice = get_llm().with_structured_output(VoiceChoice).invoke([
        SystemMessage(VOICE_PROMPT.format(catalog=catalog)),
        HumanMessage(f"Documentary topic: {topic}"),
    ])
    if choice.voice not in VOICES:               # never trust a name the LLM made up
        return DEFAULT_VOICE, f"'{choice.voice}' is not an approved voice, using the default"
    return choice.voice, choice.reason


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
            {"type": "text", "text": f"The shot should show ({idea.kind}): {idea.description}\n"
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
            HumanMessage(f"Narration: {narration}\nThis shot should show ({kind}): "
                         f"{idea.description}\n\nImages:\n{listing}"),
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

    # 2) Fill each shot (code enforces the limit, "no repeats" and "no near-duplicates")
    shots, problems, used_urls, used_titles = [], [], set(), []
    for i, idea in enumerate(plan.shots[:MAX_SHOTS], 1):
        path = media_dir / f"scene_{scene.scene_number:02d}_shot_{len(shots) + 1}.jpg"
        image, kind, reasons = find_image(idea, scene.narration, plan.fallback_queries,
                                          used_urls, used_titles, path)
        if image is None:
            problems.append(f"scene {scene.scene_number} shot {i} ('{idea.description[:50]}'): "
                            + "; ".join(reasons))
            continue
        used_urls.add(image["url"])
        used_titles.append(image["alt"])
        shots.append(Shot(
            image_path=str(path), kind=kind, description=idea.description,
            credit=f"\"{image['alt']}\" by {image['author']} ({image['license']}) {image['page']}",
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
    Try the chosen voice first, then the other approved voices. edge-tts sometimes
    fails for one voice at a time, so a fallback voice keeps the documentary going.
    If EVERY voice fails, raise: RetryPolicy waits and runs this node again.
    """
    scene = state["scene"]
    path = Path(state["media_dir"]) / f"scene_{scene.scene_number:02d}.mp3"
    chosen = state.get("voice") or DEFAULT_VOICE
    order = [chosen] + [v for v in VOICES if v != chosen]

    errors = []
    for voice in order:
        try:
            text_to_speech(scene.narration, path, voice=voice)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{voice}: {type(e).__name__}")
            continue
        result = {"audio_path": str(path), "duration_sec": audio_duration(path)}
        if voice != chosen:
            result["problems"] = [f"scene {scene.scene_number}: voice {chosen} failed, used {voice}"]
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