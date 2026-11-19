"""
Narration languages (Stage 7). Like the style, the language is chosen once per
documentary and stored in the graph STATE, so it survives the review pauses.

Two kinds of text live side by side in the agent:
  * machine-facing: research, fact-checking, image search queries and visual
    descriptions stay in ENGLISH (the sources and the image library are English)
  * human-facing: the narration (what you hear and review) is in the chosen language
"""

from dataclasses import dataclass, field

from .media import DEFAULT_VOICE, VOICES


@dataclass(frozen=True)
class Language:
    label: str
    writer: str                                   # extra instructions for the script writer
    voices: dict[str, str] = field(default_factory=dict)   # approved narrators for this language
    default_voice: str = ""
    rtl: bool = False                             # written right-to-left (for displaying it)
    # Speaking speed/pitch for this language, overriding the style's. None = use the style's.
    # Needed for Urdu: the service returns NO audio when Urdu voices get a slower or deeper
    # delivery (tested: "-15%"/"-8Hz" -> NoAudioReceived, normal -> works).
    speech: tuple[str, str] | None = None


LANGUAGES: dict[str, Language] = {
    "en": Language(
        label="English",
        writer="",
        voices=VOICES,
        default_voice=DEFAULT_VOICE,
    ),
    "ur": Language(
        label="اردو (Urdu)",
        writer=("Write every scene's `narration` in URDU, in Urdu script: natural spoken Urdu as "
                "used in Pakistani TV documentaries, with common words, not a word-by-word "
                "translation. Write names of people and places as Urdu speakers say them.\n"
                "Keep `title` and `visual_description` in ENGLISH: they are used for the title "
                "card and for searching an English image library."),
        voices={
            "ur-PK-AsadNeural": "Urdu (Pakistan) male. Calm and clear, the default.",
            "ur-PK-UzmaNeural": "Urdu (Pakistan) female. Warm and friendly, good for children.",
        },
        default_voice="ur-PK-AsadNeural",
        rtl=True,
        speech=("+0%", "+0Hz"),
    ),
}
DEFAULT_LANGUAGE = "en"


def get_language(code: str | None) -> Language:
    """Unknown or missing language (e.g. runs made before this step) -> English."""
    return LANGUAGES.get(code or DEFAULT_LANGUAGE, LANGUAGES[DEFAULT_LANGUAGE])