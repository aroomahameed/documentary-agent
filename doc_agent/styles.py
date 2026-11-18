"""
Style presets (Stage 7). A style is chosen once per documentary and changes
the whole pipeline: how the script is written, how the narrator speaks, which
voice fits, and the shape of the video.

The style lives in the graph STATE (not in LangGraph's runtime context):
runs pause for human reviews and resume later, and only state is checkpointed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    label: str
    writer: str            # extra instructions for the script writer
    planner: str           # extra instructions for the research planner
    voice_hint: str        # what kind of narrator fits (given to the voice chooser)
    rate: str              # edge-tts speaking rate
    pitch: str             # edge-tts pitch
    vertical: bool = False # 9:16 video for phones
    fixed_minutes: int | None = None   # e.g. a short is always 1 minute


STYLES: dict[str, Style] = {
    "documentary": Style(
        label="🎬 Documentary",
        writer="Serious and cinematic. Short, evocative sentences with a sense of drama "
               "and wonder, like a classic history documentary.",
        planner="Tell the full story, from origins to legacy.",
        voice_hint="a calm, deep, authoritative documentary narrator",
        rate="-15%", pitch="-8Hz",
    ),
    "kids": Style(
        label="🧒 Kids explainer",
        writer="For children aged 8-12. Simple words, short sentences, fun comparisons to "
               "things kids know, and one question to the viewer. Never scary.",
        planner="Pick the most surprising and fun parts of the topic for children.",
        voice_hint="a warm, friendly, cheerful voice that children enjoy",
        rate="-5%", pitch="+0Hz",
    ),
    "short": Style(
        label="📱 60-second short",
        writer="A vertical video for phones. The first sentence must be a strong hook. "
               "Punchy, fast, no slow build-up. Exactly 3 or 4 scenes.",
        planner="Exactly 3 subtopics: the hook, the most interesting fact, the takeaway.",
        voice_hint="an energetic, clear, engaging voice",
        rate="+0%", pitch="+0Hz",
        vertical=True, fixed_minutes=1,
    ),
}
DEFAULT_STYLE = "documentary"


def get_style(name: str | None) -> Style:
    """Unknown or missing style (e.g. runs made before Stage 7) -> the default."""
    return STYLES.get(name or DEFAULT_STYLE, STYLES[DEFAULT_STYLE])