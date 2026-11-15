"""
Stage 3 graph:

    START -> planner -(Send x N)-> researcher -> fact_checker
    fact_checker -(weak subtopics, Send)-> researcher        (loop, max 2 rounds)
    fact_checker -(ok / limit reached)-> script_writer
    script_writer -> human_review
    human_review -(approve)-> END
    human_review -(feedback)-> script_writer                  (revision loop)
"""

from langgraph.graph import START, StateGraph

from .nodes import (
    dispatch_research,
    fact_check,
    human_review,
    plan_research,
    research_subtopic,
    route_after_fact_check,
    write_script,
)
from .state import DocState


def build_graph(checkpointer=None):
    builder = StateGraph(DocState)

    builder.add_node("planner", plan_research)
    builder.add_node("researcher", research_subtopic)
    builder.add_node("fact_checker", fact_check)
    builder.add_node("script_writer", write_script)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_research, ["researcher"])
    builder.add_edge("researcher", "fact_checker")
    builder.add_conditional_edges(
        "fact_checker", route_after_fact_check, ["researcher", "script_writer"]
    )
    builder.add_edge("script_writer", "human_review")
    # No edges OUT of human_review: it returns Command(goto=...) itself.
    # No script_writer -> END edge either, or the graph would finish without asking you.

    return builder.compile(checkpointer=checkpointer)