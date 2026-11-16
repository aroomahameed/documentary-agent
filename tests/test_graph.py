"""
Offline tests for the Documentary Agent: no API keys, no network, no cost.

We swap the real LLM, web search and fact-checker agent for FAKES whose answers
we control. That lets us test the GRAPH LOGIC (fan-out, loops, reducers,
interrupts, time travel) deterministically.

    python -m pytest -q
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from doc_agent import nodes, production
from doc_agent.graph import build_graph
from doc_agent.state import (
    ClaimCheck,
    Fact,
    FactCheckReport,
    ResearchNote,
    ResearchPlan,
    Scene,
    Script,
)

SUBTOPICS = ["Origins", "Golden age", "Legacy"]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def confirmed(n: int = 1) -> ClaimCheck:
    return ClaimCheck(subtopic_number=n, claim="ok claim", status="confirmed")


def contradicted(n: int, claim: str = "bad claim", correction: str = "the real fact") -> ClaimCheck:
    return ClaimCheck(subtopic_number=n, claim=claim, status="contradicted", correction=correction)


def unverified(n: int, claim: str = "maybe claim") -> ClaimCheck:
    return ClaimCheck(subtopic_number=n, claim=claim, status="unverified")


class Fake:
    """Controls every fake and records what the graph sent to them."""

    def __init__(self):
        self.fact_check_reports: list[FactCheckReport] = []  # one per fact-check round
        self.researched: list[str] = []         # subtopics, in the order researched
        self.researcher_prompts: list[str] = []
        self.writer_prompts: list[str] = []
        self.fact_check_inputs: list[str] = []
        self.search_fails = False
        self.voice_choice = "en-GB-RyanNeural"
        self.tts_calls: list[str] = []      # voices used, one per narrated scene
        self.tts_fails = False

    # --- LLM with structured output ---
    def with_structured_output(self, schema):
        fake = self

        class Structured:
            def invoke(self, messages):
                text = "\n".join(m.content if isinstance(m.content, str) else str(m.content)
                                 for m in messages)
                if schema is ResearchPlan:
                    return ResearchPlan(title="Test Doc", subtopics=SUBTOPICS)
                if schema is ResearchNote:
                    fake.researcher_prompts.append(text)
                    return ResearchNote(
                        subtopic="?",  # the node overwrites this with the real name
                        summary="A summary.",
                        facts=[Fact(claim="Something happened in 1600.", source_url="https://example.com")],
                    )
                if schema is Script:
                    fake.writer_prompts.append(text)
                    version = len(fake.writer_prompts)
                    return Script(
                        title="Test Doc",
                        scenes=[
                            Scene(scene_number=1, narration=f"Draft {version}.", visual_description="A fort."),
                            Scene(scene_number=2, narration="Second scene.", visual_description="A garden."),
                        ],
                    )
                # --- Stage 4: scene production ---
                if schema is production.VoiceChoice:
                    return production.VoiceChoice(voice=fake.voice_choice, reason="test")
                if schema is production.ShotPlan:
                    return production.ShotPlan(
                        shots=[production.ShotIdea(description="a fort", kind="photo", queries=["fort"])],
                        fallback_queries=["castle"],
                    )
                if schema is production.ImageChoice:
                    return production.ImageChoice(number=1, reason="test")
                if schema is production.ImageCheck:
                    return production.ImageCheck(shows="a fort", matches=True)
                raise ValueError(f"unexpected schema {schema}")

        return Structured()

    # --- web search used by the researcher ---
    def web_search(self, query, max_results=4):
        if self.search_fails:
            raise ConnectionError("API down")
        self.researched.append(query.split(": ", 1)[1])
        return [{"title": "t", "url": "https://example.com", "content": "c"}]

    # --- the fact-checker agent ---
    def create_agent(self, model, tools, system_prompt, response_format):
        fake = self

        class Agent:
            def invoke(self, inputs, config=None):
                fake.fact_check_inputs.append(inputs["messages"][0].content)
                # Default to "all confirmed" once the scripted reports run out.
                report = fake.fact_check_reports.pop(0) if fake.fact_check_reports \
                    else FactCheckReport(checks=[confirmed(1)])
                return {"structured_response": report}

        return Agent()


    # --- Stage 4 media: no network, tiny fake files ---
    def search_photos(self, query, n=5, kind="photo"):
        return [{"alt": f"{query} image", "url": f"https://img/{query}", "thumb": "t",
                 "author": "Tester", "license": "CC0", "page": "https://page"}]

    @staticmethod
    def download(url, path, fallback_url=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake image")
        return path

    def text_to_speech(self, text, path, voice=None):
        if self.tts_fails:
            raise ConnectionError("TTS down")
        self.tts_calls.append(voice)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake mp3")
        return path


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = Fake()
    monkeypatch.chdir(tmp_path)                 # output/ is created inside a temp folder
    monkeypatch.delenv("TTS_VOICE", raising=False)
    monkeypatch.setattr(nodes, "get_llm", lambda: f)
    monkeypatch.setattr(nodes, "web_search", f.web_search)
    monkeypatch.setattr(nodes, "create_agent", f.create_agent)
    monkeypatch.setattr(production, "get_llm", lambda: f)
    monkeypatch.setattr(production, "search_photos", f.search_photos)
    monkeypatch.setattr(production, "download", f.download)
    monkeypatch.setattr(production, "image_preview_b64", lambda path: "ZmFrZQ==")
    monkeypatch.setattr(production, "text_to_speech", f.text_to_speech)
    monkeypatch.setattr(production, "audio_duration", lambda path: 3.0)
    return f


@pytest.fixture
def graph():
    # InMemorySaver: a checkpointer that lives only for the test (no file on disk).
    return build_graph(checkpointer=InMemorySaver())


def config(thread="t1"):
    return {"configurable": {"thread_id": thread}}


def start(graph, thread="t1"):
    """Run a new thread until it pauses (or ends). Returns the latest snapshot."""
    graph.invoke({"topic": "Mughal architecture", "target_minutes": 1}, config(thread))
    return graph.get_state(config(thread))


def resume(graph, decision, thread="t1"):
    graph.invoke(Command(resume=decision), config(thread))
    return graph.get_state(config(thread))


# ---------------------------------------------------------------------------
# Stage 1: parallel research
# ---------------------------------------------------------------------------

def test_researches_every_subtopic_once(fake, graph):
    snap = start(graph)

    assert sorted(fake.researched) == sorted(SUBTOPICS)
    assert sorted(n.subtopic for n in snap.values["research"]) == sorted(SUBTOPICS)


def test_search_failure_does_not_crash(fake, graph):
    fake.search_fails = True
    snap = start(graph)

    assert all(n.summary == "No sources found." for n in snap.values["research"])
    assert snap.next == ("human_review",)  # the run still reached review


# ---------------------------------------------------------------------------
# Stage 2: fact-check loop
# ---------------------------------------------------------------------------

def test_all_confirmed_goes_straight_to_script(fake, graph):
    fake.fact_check_reports = [FactCheckReport(checks=[confirmed(1), confirmed(2)])]
    snap = start(graph)

    assert len(fake.researched) == 3          # no re-research
    assert snap.values["research_rounds"] == 1
    assert len(fake.writer_prompts) == 1


def test_contradiction_reresearches_only_that_subtopic(fake, graph):
    fake.fact_check_reports = [
        FactCheckReport(checks=[contradicted(2, "Built in 1500", "Built in 1632")]),
        FactCheckReport(checks=[confirmed(1)]),
    ]
    snap = start(graph)

    # 3 initial researches + 1 re-research of subtopic #2 ("Golden age")
    assert fake.researched[3:] == ["Golden age"]
    # The re-researcher was told what to fix
    assert "Built in 1632" in fake.researcher_prompts[-1]
    # Round 2 only re-checked the re-researched subtopic
    assert "Golden age" in fake.fact_check_inputs[1]
    assert "Origins" not in fake.fact_check_inputs[1]
    assert snap.values["research_rounds"] == 2


def test_merge_reducer_replaces_old_note(fake, graph):
    fake.fact_check_reports = [FactCheckReport(checks=[contradicted(1)])]
    snap = start(graph)

    # Re-research REPLACED the note: still exactly one note per subtopic
    assert len(snap.values["research"]) == 3
    assert sorted(n.subtopic for n in snap.values["research"]) == sorted(SUBTOPICS)


def test_round_limit_passes_corrections_to_writer(fake, graph):
    fake.fact_check_reports = [
        FactCheckReport(checks=[contradicted(1, "Wrong A", "Fix A")]),
        FactCheckReport(checks=[contradicted(1, "Wrong B", "Fix B")]),
    ]
    snap = start(graph)

    assert snap.values["research_rounds"] == 2          # stopped at MAX_ROUNDS
    assert len(fake.researched) == 4                    # only ONE re-research round
    assert "found WRONG" in fake.writer_prompts[0]      # fallback: writer was warned
    assert "Fix B" in fake.writer_prompts[0]


def test_unverified_claims_are_hedged_not_reresearched(fake, graph):
    fake.fact_check_reports = [FactCheckReport(checks=[unverified(1, "It ran out of water")])]
    snap = start(graph)

    assert len(fake.researched) == 3                    # unverified does NOT trigger re-research
    assert snap.values["unverified_claims"] == ["It ran out of water"]
    assert "It ran out of water" in fake.writer_prompts[0]


def test_invalid_subtopic_number_is_ignored(fake, graph):
    fake.fact_check_reports = [FactCheckReport(checks=[contradicted(99)])]
    snap = start(graph)

    assert snap.values["weak_subtopics"] == []          # LLM's bad number didn't cause a loop
    assert len(fake.researched) == 3


# ---------------------------------------------------------------------------
# Stage 3: human review, persistence, time travel
# ---------------------------------------------------------------------------

def test_pauses_for_review_then_approve_ends(fake, graph):
    snap = start(graph)
    assert snap.next == ("human_review",)               # paused, waiting for a human

    snap = resume(graph, {"action": "approve"})
    assert snap.next == ()                              # finished
    assert snap.values["script"].scenes[0].narration == "Draft 1."


def test_feedback_revises_script_and_pauses_again(fake, graph):
    start(graph)
    snap = resume(graph, {"action": "revise", "feedback": "More dramatic please"})

    assert snap.next == ("human_review",)               # asks again after revising
    assert "More dramatic please" in fake.writer_prompts[-1]
    assert "previous draft" in fake.writer_prompts[-1]
    assert snap.values["script"].scenes[0].narration == "Draft 2."


def test_threads_are_independent(fake, graph):
    start(graph, thread="a")
    resume(graph, {"action": "approve"}, thread="a")
    snap_b = start(graph, thread="b")

    assert graph.get_state(config("a")).next == ()
    assert snap_b.next == ("human_review",)


def test_time_travel_fork_rewrites_without_new_research(fake, graph):
    start(graph)
    resume(graph, {"action": "approve"})
    searches_before = len(fake.researched)

    # Same logic as main.py --fork: find the original checkpoint before script_writer
    history = list(graph.get_state_history(config()))
    target = next(s for s in reversed(history) if s.next == ("script_writer",))
    graph.update_state(target.config, {"human_feedback": "For kids"})
    graph.invoke(None, config())

    snap = graph.get_state(config())
    assert len(fake.researched) == searches_before      # no new research
    assert "Director's notes" in fake.writer_prompts[-1]
    assert "For kids" in fake.writer_prompts[-1]
    assert snap.next == ("human_review",)               # the fork also waits for review

# ---------------------------------------------------------------------------
# Stage 4: scene production after approval
# ---------------------------------------------------------------------------

def test_approval_produces_every_scene(fake, graph):
    start(graph)
    snap = resume(graph, {"action": "approve"})

    assert snap.next == ()                                      # ran to the end
    produced = snap.values["produced_scenes"]
    assert [p.scene_number for p in produced] == [1, 2]         # all scenes, in order
    assert all(p.shots and p.audio_path for p in produced)
    assert snap.values["voice"] == "en-GB-RyanNeural"


def test_narrator_chosen_once_for_whole_documentary(fake, graph):
    start(graph)
    resume(graph, {"action": "approve"})

    assert fake.tts_calls == ["en-GB-RyanNeural", "en-GB-RyanNeural"]   # same voice, every scene


def test_made_up_voice_falls_back_to_default(fake, graph):
    fake.voice_choice = "en-US-MadeUpNeural"
    start(graph)
    snap = resume(graph, {"action": "approve"})

    assert snap.values["voice"] == production.DEFAULT_VOICE


def test_manifest_written_for_stage5(fake, graph):
    import json
    from pathlib import Path

    start(graph)
    snap = resume(graph, {"action": "approve"})

    manifest = json.loads(Path(snap.values["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["title"] == "Test Doc"
    assert len(manifest["scenes"]) == 2
    assert manifest["total_seconds"] == 6.0
    assert manifest["credits"]                                  # every photo is credited


def test_no_production_before_approval(fake, graph):
    snap = start(graph)

    assert snap.next == ("human_review",)
    assert not snap.values.get("produced_scenes")    # reducer fields start as an empty list
    assert fake.tts_calls == []


def test_failed_production_can_be_resumed(fake, graph):
    fake.tts_fails = True                       # every voice fails -> retries -> error
    start(graph)
    with pytest.raises(ConnectionError):
        graph.invoke(Command(resume={"action": "approve"}), config())

    snap = graph.get_state(config())
    assert snap.next                            # stopped mid-run, NOT finished
    assert not snap.interrupts                  # and NOT waiting for a human

    fake.tts_fails = False                      # "fix the problem" ...
    graph.invoke(None, config())                # ... and continue with None
    snap = graph.get_state(config())
    assert snap.next == ()
    assert len(snap.values["produced_scenes"]) == 2