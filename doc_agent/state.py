"""
State and data models for the Documentary Agent.

Two kinds of models live here:
  * Pydantic models  -> what the LLM must return (structured output)
  * TypedDict states -> what flows through the LangGraph graph

Rule of thumb: the LLM fills Pydantic models; the graph carries TypedDicts.
"""

import json
import operator
from typing import Annotated, Literal, NotRequired, TypedDict 

from pydantic import BaseModel, Field, model_validator


class LLMOutput(BaseModel):
    """
    Base class for every model an LLM fills (structured output).

    LLMs occasionally return a nested list or object as a JSON *string*
    ('[{"description": ...}]') instead of a real list. Rather than crash,
    parse such strings, but only for fields that aren't meant to be text.
    """

    @model_validator(mode="before")
    @classmethod
    def _parse_json_strings(cls, data):
        if not isinstance(data, dict):
            return data
        fixed = dict(data)
        for name, value in data.items():
            field = cls.model_fields.get(name)
            if field is None or field.annotation is str or not isinstance(value, str):
                continue
            if value.strip()[:1] in ("[", "{"):
                try:
                    fixed[name] = json.loads(value)
                except json.JSONDecodeError:
                    pass                      # leave it: Pydantic will report the real error
        return fixed


# ---------------------------------------------------------------------------
# LLM output models
# ---------------------------------------------------------------------------

class ResearchPlan(LLMOutput):
    """Planner output: how to split the topic."""

    title: str = Field(description="A catchy working title for the documentary")
    subtopics: list[str] = Field(
        description="3 to 5 distinct subtopics, ordered the way the story should flow"
    )


class Fact(LLMOutput):
    claim: str = Field(description="One specific, checkable fact (a date, number, name or event)")
    source_url: str = Field(description="URL of the source this fact came from")


class ResearchNote(LLMOutput):
    """Researcher output for ONE subtopic."""

    subtopic: str
    summary: str = Field(description="2-4 sentence overview of this subtopic")
    facts: list[Fact] = Field(description="4-8 key facts, each tied to a source URL")


class Scene(LLMOutput):
    """One scene of the script. Later stages attach media to it."""

    scene_number: int
    narration: str = Field(description="What the narrator says in this scene (spoken English)")
    visual_description: str = Field(
        description="What the viewer should see, e.g. 'aerial shot of Badshahi Mosque at sunset'"
    )
    source_urls: list[str] = Field(
        default_factory=list, description="Sources backing the claims in this scene"
    )


class Script(LLMOutput):
    title: str
    scenes: list[Scene]


class Shot(BaseModel):
    """One image shown during a scene. A scene has 1-4 shots that change while the narrator speaks."""

    image_path: str
    kind: str                       # "photo", "artwork" or "title_card"
    description: str                # what the picture editor wanted to show
    credit: str = ""                # "Title by Author (CC BY 2.0) <page>", empty for title cards


class ProducedScene(BaseModel):
    """A scene with its media on disk (Stage 4). Only PATHS go in state, never file bytes."""

    scene_number: int
    narration: str
    shots: list[Shot]
    audio_path: str
    duration_sec: float
    problems: list[str] = Field(default_factory=list)   # e.g. "shot 2: nothing found"


def merge_scenes(existing: list[ProducedScene] | None, new: list[ProducedScene]) -> list[ProducedScene]:
    """Reducer: a re-produced scene REPLACES the old one with the same number. Kept in scene order."""
    merged = {s.scene_number: s for s in (existing or [])}
    for s in new:
        merged[s.scene_number] = s
    return [merged[k] for k in sorted(merged)]


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
def merge_research(existing: list[ResearchNote] | None, new: list[ResearchNote]) -> list[ResearchNote]:
    """Reducer: a new note REPLACES the old note for the same subtopic."""
    merged = {n.subtopic: n for n in (existing or [])}
    for n in new:
        merged[n.subtopic] = n
    return list(merged.values())

class DocState(TypedDict, total=False):
    # --- input ---
    topic: str
    target_minutes: int

    # --- planner ---
    title: str
    subtopics: list[str]
    research: Annotated[list[ResearchNote], merge_research]
    unverified_claims: Annotated[list[str], operator.add] 

    # --- script writer ---
    script: Script

    # --- added in later stages (see README roadmap) ---
      # --- fact checker ---
    fact_check_issues: list[str]
    weak_subtopics: list[str]
    research_rounds: int
    human_feedback: str

    # --- scene production (Stage 4) ---
    media_dir: str                     # where this run's images and audio go
    voice: str                         # narrator, chosen once for the whole documentary
    produced_scenes: Annotated[list[ProducedScene], merge_scenes]   # filled in parallel
    manifest_path: str                 # summary of everything produced

    # --- video (Stage 5) ---
    quality_issues: list[str]          # found by code in quality_check, shown at production review
    video_path: str


class ResearchTask(TypedDict):
    topic: str
    subtopic: str
    issues: NotRequired[list[str]]  

class ClaimCheck(LLMOutput):
    subtopic_number: int = Field(description="The [number] of the subtopic this claim belongs to")
    claim: str = Field(description="The claim you checked, short")
    status: Literal["confirmed", "contradicted", "unverified"]
    correction: str = Field(default="", description="Only if contradicted: what reliable sources actually say")


class FactCheckReport(LLMOutput):
    checks: list[ClaimCheck]