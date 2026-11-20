"""
Stage 7: the AI EDITOR (supervisor pattern).

After production, the editor reviews the whole documentary the way a human
editor would: it reads an overview of every scene, LOOKS at images whose titles
don't say enough (a tool, with a budget), and suggests redo notes for the scenes
that would improve the most. It only SUGGESTS: the human decides at the
production review.

Code keeps the agent safe: a look budget enforced in the tool itself, invalid
scene numbers dropped, at most MAX_SUGGESTIONS, and a failing editor never
blocks the review.
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.config import get_stream_writer
from pydantic import Field

from .llm import get_llm
from .media import image_preview_b64, is_image
from .prompts import EDITOR_PROMPT
from .state import DocState, LLMOutput, ProducedScene

MAX_LOOKS = 6
MAX_SUGGESTIONS = 3
EDITOR_ENABLED = os.getenv("EDITOR_ENABLED", "true").lower() != "false"   # off = save credit


class Suggestion(LLMOutput):
    scene_number: int
    note: str = Field(description="A concrete redo note for the picture editor: name the real "
                                  "subjects to show, and at which words")
    reason: str = Field(description="Why this scene needs a redo, one short sentence")


class EditorReport(LLMOutput):
    summary: str = Field(description="2-3 sentences: overall quality and the biggest problems")
    suggestions: list[Suggestion] = Field(default_factory=list,
                                          description=f"At most {MAX_SUGGESTIONS}; empty if all is fine")


class ImageDescription(LLMOutput):
    shows: str = Field(description="What the image shows, one sentence")


def _progress(message: str) -> None:
    """Live progress if a graph is streaming; silently nothing otherwise (e.g. in a test)."""
    try:
        get_stream_writer()({"msg": message})
    except (KeyError, RuntimeError):      # not running inside a graph: nobody to tell
        pass


def overview(state: DocState) -> str:
    """Everything the editor needs to know, as plain text (no images: it LOOKS when needed)."""
    lines = [f"Documentary: {state['script'].title}", ""]
    for p in state.get("produced_scenes", []):
        lines.append(f"## Scene {p.scene_number} ({p.duration_sec:.0f}s)")
        lines.append(f"Narration: {p.narration}")
        for i, shot in enumerate(p.shots, 1):
            title = shot.credit.split('" by ')[0].lstrip('"') if shot.credit else "(none)"
            lines.append(f"  shot {i} [{shot.kind}] at “{shot.anchor}”: wanted: {shot.description}"
                         f" | image title: {title}")
        for problem in p.problems:
            lines.append(f"  problem: {problem}")
        lines.append("")
    issues = state.get("quality_issues", [])
    if issues:
        lines.append("Quality check issues:")
        lines += [f"- {i}" for i in issues]
    return "\n".join(lines)


def make_look_tool(scenes: list[ProducedScene]):
    """
    The editor's tool. Built per review, so it can only see THIS documentary's images,
    and the budget is enforced HERE, in code: the LLM can't talk its way past it.
    """
    by_number = {p.scene_number: p for p in scenes}
    looks = {"used": 0}

    @tool
    def look_at_shot(scene_number: int, shot_number: int) -> str:
        """Look at one image of the documentary and get a short description of what it
        really shows. Use it when an image's title doesn't make its content clear."""
        if looks["used"] >= MAX_LOOKS:
            return f"Look budget used up ({MAX_LOOKS} looks). Decide with what you know."
        scene = by_number.get(scene_number)
        if scene is None or not 1 <= shot_number <= len(scene.shots):
            return f"There is no shot {shot_number} in scene {scene_number}."
        path = scene.shots[shot_number - 1].image_path
        if not is_image(path):
            return "That image file is broken or missing."
        looks["used"] += 1
        _progress(f"editor looks at scene {scene_number}, shot {shot_number}")
        described = get_llm().with_structured_output(ImageDescription).invoke([
            SystemMessage("Describe what this image shows in one factual sentence."),
            HumanMessage(content=[{"type": "image_url", "image_url":
                                   {"url": f"data:image/jpeg;base64,{image_preview_b64(path)}"}}]),
        ])
        return described.shows

    return look_at_shot


def editor_review(state: DocState) -> dict:
    """Graph node: review the produced scenes and suggest redo notes (never blocks)."""
    if not EDITOR_ENABLED:
        return {"editor_summary": "", "editor_suggestions": {}}
    _progress("🧑‍💼 the AI editor is reviewing the scenes...")
    scenes = state.get("produced_scenes", [])
    agent = create_agent(get_llm(), tools=[make_look_tool(scenes)],
                         system_prompt=EDITOR_PROMPT.format(max=MAX_SUGGESTIONS, looks=MAX_LOOKS),
                         response_format=EditorReport)
    try:
        result = agent.invoke({"messages": [HumanMessage(overview(state))]}, {"recursion_limit": 30})
        report: EditorReport = result["structured_response"]
    except Exception as e:  # noqa: BLE001  (the editor is optional: never block the review)
        return {"editor_summary": f"The AI editor couldn't finish ({type(e).__name__}).",
                "editor_suggestions": {}}

    # Never trust the agent's numbers: keep real scenes only, one note each, at most MAX_SUGGESTIONS
    valid = {p.scene_number for p in scenes}
    suggestions = {}
    for s in report.suggestions:
        if s.scene_number in valid and s.note.strip() and s.scene_number not in suggestions:
            suggestions[s.scene_number] = s.note.strip()
        if len(suggestions) == MAX_SUGGESTIONS:
            break
    return {"editor_summary": report.summary, "editor_suggestions": suggestions}