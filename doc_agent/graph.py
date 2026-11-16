"""
Main graph (Stages 1-4):

    START -> planner -(Send x N)-> researcher -> fact_checker
    fact_checker -(weak subtopics, Send)-> researcher        (loop, max 2 rounds)
    fact_checker -(ok / limit reached)-> script_writer
    script_writer -> human_review
    human_review -(feedback)-> script_writer                  (revision loop)
    human_review -(approve)-> start_production                (pick folder + narrator)
    start_production -(Send x scenes)-> scene_producer        (SUBGRAPH, in parallel)
    scene_producer -> collect_scenes -> END                   (manifest for Stage 5)
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    collect_scenes,
    dispatch_research,
    dispatch_scenes,
    fact_check,
    human_review,
    plan_research,
    research_subtopic,
    route_after_fact_check,
    start_production,
    write_script,
)
from .production import build_scene_graph
from .state import DocState


def build_graph(checkpointer=None):
    builder = StateGraph(DocState)

    builder.add_node("planner", plan_research)
    builder.add_node("researcher", research_subtopic)
    builder.add_node("fact_checker", fact_check)
    builder.add_node("script_writer", write_script)
    builder.add_node("human_review", human_review)
    builder.add_node("start_production", start_production)
    builder.add_node("scene_producer", build_scene_graph())   # a compiled SUBGRAPH as a node
    builder.add_node("collect_scenes", collect_scenes)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", dispatch_research, ["researcher"])
    builder.add_edge("researcher", "fact_checker")
    builder.add_conditional_edges(
        "fact_checker", route_after_fact_check, ["researcher", "script_writer"]
    )
    builder.add_edge("script_writer", "human_review")
    # human_review -> start_production happens via Command(goto=...) on approval.
    builder.add_conditional_edges("start_production", dispatch_scenes, ["scene_producer"])
    builder.add_edge("scene_producer", "collect_scenes")      # fan-in: waits for every scene
    builder.add_edge("collect_scenes", END)
    # No edges OUT of human_review: it returns Command(goto=...) itself.
    # No script_writer -> END edge either, or the graph would finish without asking you.

    return builder.compile(checkpointer=checkpointer)