"""
Tests that run WITHOUT API keys, by swapping the LLM and search for fakes.
Testing agents offline like this is a habit worth building early.

    pytest -q
"""

from doc_agent import nodes
from doc_agent.graph import build_graph
from doc_agent.state import Fact, ResearchNote, ResearchPlan, Scene, Script


class FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        if self.schema is ResearchPlan:
            return ResearchPlan(title="Test Doc", subtopics=["Origins", "Golden age", "Legacy"])
        if self.schema is ResearchNote:
            return ResearchNote(
                subtopic="?",
                summary="A summary.",
                facts=[Fact(claim="Something happened in 1600.", source_url="https://example.com")],
            )
        if self.schema is Script:
            return Script(
                title="Test Doc",
                scenes=[Scene(scene_number=1, narration="Hello.", visual_description="A fort.")],
            )
        raise ValueError(self.schema)


class FakeLLM:
    def with_structured_output(self, schema):
        return FakeStructured(schema)


def test_full_stage1_run(monkeypatch):
    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        nodes, "web_search",
        lambda q, max_results=4: [{"title": "t", "url": "https://example.com", "content": "c"}],
    )

    result = build_graph().invoke({"topic": "Mughal architecture", "target_minutes": 1})

    # One research note per subtopic, merged by the reducer
    assert sorted(n.subtopic for n in result["research"]) == ["Golden age", "Legacy", "Origins"]
    assert result["script"].scenes[0].narration == "Hello."


def test_search_failure_does_not_crash(monkeypatch):
    def broken_search(q, max_results=4):
        raise ConnectionError("API down")

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())
    monkeypatch.setattr(nodes, "web_search", broken_search)

    result = build_graph().invoke({"topic": "x", "target_minutes": 1})
    assert all(n.summary == "No sources found." for n in result["research"])
