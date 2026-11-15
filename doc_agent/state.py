"""
State and data models for the Documentary Agent.

Two kinds of models live here:
  * Pydantic models  -> what the LLM must return (structured output)
  * TypedDict states -> what flows through the LangGraph graph

Rule of thumb: the LLM fills Pydantic models; the graph carries TypedDicts.
"""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict 

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LLM output models
# ---------------------------------------------------------------------------

class ResearchPlan(BaseModel):
    """Planner output: how to split the topic."""

    title: str = Field(description="A catchy working title for the documentary")
    subtopics: list[str] = Field(
        description="3 to 5 distinct subtopics, ordered the way the story should flow"
    )


class Fact(BaseModel):
    claim: str = Field(description="One specific, checkable fact (a date, number, name or event)")
    source_url: str = Field(description="URL of the source this fact came from")


class ResearchNote(BaseModel):
    """Researcher output for ONE subtopic."""

    subtopic: str
    summary: str = Field(description="2-4 sentence overview of this subtopic")
    facts: list[Fact] = Field(description="4-8 key facts, each tied to a source URL")


class Scene(BaseModel):
    """One scene of the script. Later stages attach media to it."""

    scene_number: int
    narration: str = Field(description="What the narrator says in this scene (spoken English)")
    visual_description: str = Field(
        description="What the viewer should see, e.g. 'aerial shot of Badshahi Mosque at sunset'"
    )
    source_urls: list[str] = Field(
        default_factory=list, description="Sources backing the claims in this scene"
    )


class Script(BaseModel):
    title: str
    scenes: list[Scene]


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

    # --- researchers (run in PARALLEL) ---
    # `operator.add` is a REDUCER: each parallel researcher returns
    # {"research": [note]} and LangGraph concatenates the lists instead of
    # letting the last writer overwrite everyone else.
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
    # produced_scenes: Annotated[list[...], operator.add]   # Stage 5
    # video_path: str                        # Stage 6


class ResearchTask(TypedDict):
    topic: str
    subtopic: str
    issues: NotRequired[list[str]]  

class ClaimCheck(BaseModel):
    subtopic_number: int = Field(description="The [number] of the subtopic this claim belongs to")
    claim: str = Field(description="The claim you checked, short")
    status: Literal["confirmed", "contradicted", "unverified"]
    correction: str = Field(default="", description="Only if contradicted: what reliable sources actually say")


class FactCheckReport(BaseModel):
    checks: list[ClaimCheck]
