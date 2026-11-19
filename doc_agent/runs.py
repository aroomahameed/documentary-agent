"""
Everything about RUNS (threads), shared by the command line (main.py) and the
web app (app.py): where checkpoints are stored, which runs exist, what state a
run is in, and time travel.
"""

import os
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# The checkpointer saves state to disk after every node. Our state contains
# Pydantic objects, and LangGraph only rebuilds types you explicitly allow
# (a security measure: loading arbitrary classes from a file can be dangerous).
SAFE_TYPES = [
    ("doc_agent.state", "ResearchNote"),
    ("doc_agent.state", "Fact"),
    ("doc_agent.state", "Script"),
    ("doc_agent.state", "Scene"),
    ("doc_agent.state", "ProducedScene"),
    ("doc_agent.state", "Shot"),
]


def make_checkpointer(path: str | None = None) -> SqliteSaver:
    path = path or os.getenv("CHECKPOINT_DB", "checkpoints.db")
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=SAFE_TYPES))
    saver.setup()                      # create the tables now, so list_threads works on a new file
    return saver


def list_threads(checkpointer: SqliteSaver) -> list[str]:
    """All saved runs, most recently active first (checkpoint IDs are time-ordered)."""
    rows = checkpointer.conn.execute(
        "SELECT thread_id, MAX(checkpoint_id) AS last FROM checkpoints "
        "GROUP BY thread_id ORDER BY last DESC"
    ).fetchall()
    return [row[0] for row in rows]


def run_status(snapshot) -> str:
    """
    What a run is doing, from its latest snapshot:
      "script_review" / "production_review" -> paused, waiting for a human
      "stopped"   -> interrupted by an error or Ctrl+C: continue with None
      "finished"  -> reached END
      "new"       -> nothing saved yet
    """
    if snapshot.interrupts:
        return snapshot.interrupts[0].value.get("type", "script_review")
    if snapshot.next:
        return "stopped"
    if snapshot.values:
        return "finished"
    return "new"


def fork_before_script(graph, config: dict, notes: str, changes: dict | None = None) -> bool:
    """
    Time travel: branch from the ORIGINAL checkpoint just before the first script
    was written (research done), with `notes` for the writer. `changes` can also
    change run settings for the new version, e.g. {"language": "ur"} for an Urdu
    version of the same research. The original timeline stays in history.
    Returns False if there is no such checkpoint.
    Afterwards, continue the run with graph.stream(None, config).
    """
    history = list(graph.get_state_history(config))          # newest first
    target = next((s for s in reversed(history) if s.next == ("script_writer",)), None)
    if target is None:
        return False
    graph.update_state(target.config, {"human_feedback": notes, **(changes or {})})
    return True