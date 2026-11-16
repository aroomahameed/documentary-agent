"""
Graph nodes. Each node: takes state -> returns a PARTIAL state update (a dict).
"""

from langchain_core.messages import HumanMessage, SystemMessage
from .llm import get_llm
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError
from .prompts import FACT_CHECKER_PROMPT, PLANNER_PROMPT, RESEARCHER_PROMPT, SCRIPT_WRITER_PROMPT
from .state import DocState, FactCheckReport, ResearchNote, ResearchPlan, ResearchTask, Script
from .tools import search_web, web_search, wikipedia_lookup
from .production import choose_voice
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send, interrupt

WORDS_PER_MINUTE = 140


# 1) Planner ---------------------------------------------------------------

def plan_research(state: DocState) -> dict:
    planner = get_llm().with_structured_output(ResearchPlan)
    plan = planner.invoke(
        [
            SystemMessage(PLANNER_PROMPT),
            HumanMessage(
                f"Topic: {state['topic']}\nTarget length: {state['target_minutes']} minute(s)"
            ),
        ]
    )
    return {"title": plan.title, "subtopics": plan.subtopics[:5]}


# 2) Fan-out: one Send per subtopic ------------------------------------------

def dispatch_research(state: DocState) -> list[Send]:
    """
    Used as a conditional edge. Returning a list of Send objects makes
    LangGraph run the 'researcher' node once PER subtopic, in parallel.
    Each researcher gets its own private input (ResearchTask), not the full state.
    """
    return [
        Send("researcher", {"topic": state["topic"], "subtopic": subtopic})
        for subtopic in state["subtopics"]
    ]


# 3) Researcher (runs N times in parallel) ---------------------------------

def research_subtopic(task: ResearchTask) -> dict:
    subtopic = task["subtopic"]

    # Errors are information: a failed search shouldn't crash the whole run.
    try:
        results = web_search(f"{task['topic']}: {subtopic}")
    except Exception as e:  # noqa: BLE001
        results = []
        print(f"  ! search failed for '{subtopic}': {e}")

    if not results:
        note = ResearchNote(subtopic=subtopic, summary="No sources found.", facts=[])
        return {"research": [note]}

    sources = "\n\n".join(
        f"[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}"
        for i, r in enumerate(results, 1)
    )
    issues = task.get("issues", [])
    feedback = ""
    if issues:
        feedback = "\n\nA fact-checker found problems in earlier research. Fix these:\n" + \
                   "\n".join(f"- {i}" for i in issues)

    researcher = get_llm().with_structured_output(ResearchNote)
    note = researcher.invoke(
        [
            SystemMessage(RESEARCHER_PROMPT),
            HumanMessage(f"Documentary topic: {task['topic']}\nSubtopic: {subtopic}\n\n"
                         f"Search results:\n{sources}{feedback}"),
        ]
    )
    note.subtopic = subtopic  # keep the exact label so we can re-order later

    # Returned as a LIST so the `operator.add` reducer can merge parallel results.
    return {"research": [note]}


# 4) Script writer (runs once, after ALL researchers finish) ----------------

def write_script(state: DocState) -> dict:
    # Parallel branches finish in random order. Restore the planner's story order.
    order = {s: i for i, s in enumerate(state["subtopics"])}
    notes = sorted(state["research"], key=lambda n: order.get(n.subtopic, 99))

    notes_text = "\n\n".join(
        f"## {n.subtopic}\n{n.summary}\n"
        + "\n".join(f"- {f.claim} (source: {f.source_url})" for f in n.facts)
        for n in notes
    )
    unverified = state.get("unverified_claims", [])
    if unverified:
        notes_text += ("\n\nDo NOT state these unverified claims as fact "
                       "(skip them, or hedge with 'according to some accounts'):\n"
                       + "\n".join(f"- {c}" for c in unverified))
    issues = state.get("fact_check_issues", [])
    if issues:
        notes_text += ("\n\nThese claims were found WRONG. Do not use them; "
                       "use the corrections instead:\n" + "\n".join(f"- {i}" for i in issues))
    minutes = state["target_minutes"]
    prompt = SCRIPT_WRITER_PROMPT.format(
        words=minutes * WORDS_PER_MINUTE,
        minutes=minutes,
        min_scenes=max(3, minutes * 2),
        max_scenes=max(5, minutes * 4),
    )
    writer = get_llm().with_structured_output(Script)
    messages = [
        SystemMessage(prompt),
        HumanMessage(f"Title: {state['title']}\n\nResearch notes:\n{notes_text}"),
    ]
    feedback = state.get("human_feedback")
    if feedback and state.get("script"):
        messages.append(HumanMessage(
            f"Here is your previous draft:\n{state['script'].model_dump_json()}\n\n"
            f"Revise it based on this feedback from the director:\n{feedback}"
        ))
    elif feedback:
        messages.append(HumanMessage(f"Director's notes for this version:\n{feedback}"))
    script = writer.invoke(messages)
    return {"script": script}

def fact_check(state: DocState) -> dict:
    rounds = state.get("research_rounds", 0) + 1

    # Round 2+: only re-check the subtopics that were just re-researched.
    notes = state["research"]
    if state.get("weak_subtopics"):
        notes = [n for n in notes if n.subtopic in state["weak_subtopics"]]

    agent = create_agent(
        get_llm(),
        tools=[search_web, wikipedia_lookup],
        system_prompt=FACT_CHECKER_PROMPT,
        response_format=FactCheckReport,
    )
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(format_notes(notes))]},
            {"recursion_limit": 20},
        )
        report: FactCheckReport = result["structured_response"]
    except GraphRecursionError:
        return {"weak_subtopics": [], "fact_check_issues": ["Fact-check hit step limit"],
                "unverified_claims": [], "research_rounds": rounds}

    def name(c):   # number -> real subtopic name, or None if the LLM gave a bad number
        i = c.subtopic_number - 1
        return notes[i].subtopic if 0 <= i < len(notes) else None

    wrong = [c for c in report.checks if c.status == "contradicted" and name(c)]
    return {
        "weak_subtopics": sorted({name(c) for c in wrong}),
        "fact_check_issues": [f"{name(c)}: '{c.claim}' is wrong. Sources say: {c.correction}" for c in wrong],
        "unverified_claims": [c.claim for c in report.checks if c.status == "unverified"],
        "research_rounds": rounds,
    }
MAX_ROUNDS = 2
def route_after_fact_check(state: DocState):
    """Loop back to research weak subtopics, or move on to the script."""
    # Only trust names that really are subtopics (LLMs sometimes rename things)
    weak = [s for s in state.get("weak_subtopics", []) if s in state["subtopics"]]

    if weak and state["research_rounds"] < MAX_ROUNDS:
        return [
            Send("researcher", {
                "topic": state["topic"],
                "subtopic": s,
                "issues": [i for i in state["fact_check_issues"] if i.startswith(s)],
            })
            for s in weak
        ]
    return "script_writer"

def format_notes(notes: list[ResearchNote]) -> str:
    return "\n\n".join(
        f"## [{i}] {n.subtopic}\n{n.summary}\n"
        + "\n".join(f"- {f.claim} (source: {f.source_url})" for f in n.facts)
        for i, n in enumerate(notes, 1)
    )

def human_review(state: DocState) -> Command[Literal["script_writer", "start_production"]]:
    # interrupt() PAUSES the graph here and saves everything to the checkpointer.
    # Whatever you pass to Command(resume=...) later becomes the return value.
    decision = interrupt({"question": "Approve the script, or give feedback to revise it."})

    if decision["action"] == "approve":
        return Command(goto="start_production")      # Stage 4: approved -> make the media
    return Command(goto="script_writer", update={"human_feedback": decision["feedback"]})

# --- Stage 4: scene production ------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def start_production(state: DocState, config: RunnableConfig) -> dict:
    """
    Runs ONCE after the script is approved: picks the folder and the narrator.

    A node can ask for `config` as a second parameter: that's how it reads the
    thread_id. The timestamp gives every production its own folder, so a fork
    (time travel) never overwrites an earlier version's media.
    """
    thread = config["configurable"]["thread_id"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    media_dir = Path("output") / f"{_slug(state['topic'])}-{thread}" / f"media-{stamp}"

    voice, reason = choose_voice(f"{state['topic']}. {state['script'].title}")
    print(f"🎙  Narrator: {voice} ({reason})")
    return {"media_dir": str(media_dir), "voice": voice}


def dispatch_scenes(state: DocState) -> list[Send]:
    """Fan-out again: one scene_producer SUBGRAPH run per scene, all in parallel."""
    return [
        Send("scene_producer", {"scene": scene, "media_dir": state["media_dir"], "voice": state["voice"]})
        for scene in state["script"].scenes
    ]


def collect_scenes(state: DocState) -> dict:
    """Fan-in: runs once every scene is done. Writes a manifest for Stage 5 (video assembly)."""
    media_dir = Path(state["media_dir"])
    media_dir.mkdir(parents=True, exist_ok=True)
    scenes = state.get("produced_scenes", [])
    manifest = {
        "title": state["script"].title,
        "voice": state["voice"],
        "total_seconds": round(sum(s.duration_sec for s in scenes), 1),
        "scenes": [s.model_dump() for s in scenes],
        "credits": [shot.credit for s in scenes for shot in s.shots if shot.credit],
    }
    path = media_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (media_dir / "script.json").write_text(state["script"].model_dump_json(indent=2), encoding="utf-8")
    return {"manifest_path": str(path)}