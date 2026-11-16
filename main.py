"""
Run the documentary agent from the command line.

    python main.py "History of Mughal architecture" --thread mughal-1
    python main.py --thread mughal-1      # resume a paused run
    python main.py --draw                 # print the graph as a Mermaid diagram
"""

import argparse
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

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


def make_checkpointer(path: str = "checkpoints.db") -> SqliteSaver:
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=SAFE_TYPES))

from doc_agent.graph import build_graph  # noqa: E402


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def save_script(script, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")

    lines = [f"# {script.title}\n"]
    for s in script.scenes:
        lines.append(f"## Scene {s.scene_number}")
        lines.append(f"**Visual:** {s.visual_description}\n")
        lines.append(s.narration + "\n")
        if s.source_urls:
            lines.append("Sources: " + ", ".join(s.source_urls) + "\n")
    (out_dir / "script.md").write_text("\n".join(lines), encoding="utf-8")


def print_update(update: dict) -> None:
    """Print a friendly line for each node that finished."""
    for node, data in update.items():
        if not isinstance(data, dict):  # e.g. the "__interrupt__" event, or a node with no update
            continue
        if node == "planner":
            print(f"📋 Plan: {data['title']}")
            for s in data["subtopics"]:
                print(f"   - {s}")
        elif node == "researcher":
            note = data["research"][0]
            print(f"🔎 Researched: {note.subtopic} ({len(note.facts)} facts)")
        elif node == "fact_checker":
            print(f"🧐 Fact-check: {len(data['weak_subtopics'])} weak subtopic(s), "
                  f"{len(data['unverified_claims'])} unverified claim(s)")
            for i in data["fact_check_issues"]:
                print(f"   ❌ {i}")
        elif node == "script_writer":
            print(f"✍️  Script ready: {len(data['script'].scenes)} scenes")
        elif node == "start_production":
            print(f"🎬 Producing scenes into {data['media_dir']}")
        elif node == "scene_producer":
            p = data["produced_scenes"][0]
            kinds = ", ".join(s.kind for s in p.shots)
            print(f"   🎞  scene {p.scene_number}: {p.duration_sec:.1f}s, {len(p.shots)} shot(s): {kinds}")
            for problem in p.problems:
                print(f"      ⚠ {problem}")
        elif node == "collect_scenes":
            print(f"📦 Manifest: {data['manifest_path']}")


def ask_human(script) -> dict:
    """Show the script and collect the director's decision."""
    print("\n" + "=" * 70)
    print(f"📝 {script.title}")
    for s in script.scenes:
        print(f"\n[{s.scene_number}] {s.narration}\n    🎥 {s.visual_description}")
    print("=" * 70)
    answer = input("\nPress Enter to approve, or type feedback to revise: ").strip()
    if answer == "":
        return {"action": "approve"}
    return {"action": "revise", "feedback": answer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", nargs="?", help="Documentary topic (only for new runs)")
    parser.add_argument("--minutes", type=int, default=1)
    parser.add_argument("--thread", default="default", help="Run ID; reuse it to resume")
    parser.add_argument("--draw", action="store_true", help="Print the graph as Mermaid and exit")
    parser.add_argument("--history", action="store_true", help="List this thread's checkpoints and exit")
    parser.add_argument("--fork", metavar="NOTES",
                        help="Time travel: rewind to before the script was written and rewrite it with these notes")
    args = parser.parse_args()

    checkpointer = make_checkpointer()
    try:
        graph = build_graph(checkpointer=checkpointer)

        if args.draw:
            print(graph.get_graph().draw_mermaid())
            return

        config = {"configurable": {"thread_id": args.thread}}

        # --- Time travel: look at every saved checkpoint (newest first) ---
        if args.history:
            for snap in graph.get_state_history(config):
                step = snap.metadata.get("step")
                cp_id = snap.config["configurable"]["checkpoint_id"]
                print(f"step {step:>3} | next: {str(snap.next):<32} | checkpoint {cp_id[-8:]}")
            return

        snapshot = graph.get_state(config)

        # Decide what to run first: fork, resume a paused run, or start a new one.
        if args.fork:
            if not snapshot.values:
                parser.error(f"thread '{args.thread}' doesn't exist yet")
            # Find the ORIGINAL checkpoint just before the first script was written
            # (history is newest-first, so search from the oldest end).
            history = list(graph.get_state_history(config))
            target = next((s for s in reversed(history) if s.next == ("script_writer",)), None)
            if target is None:
                print("No checkpoint before script_writer found in this thread.")
                return
            print(f"⏪ Rewinding to checkpoint {target.config['configurable']['checkpoint_id'][-8:]} "
                  f"(research done, no script yet)")
            # update_state on an OLD checkpoint creates a NEW branch from that point.
            # The original timeline stays in history untouched.
            graph.update_state(target.config, {"human_feedback": args.fork})
            next_input = None  # None = "continue from the latest checkpoint" (our new fork)
        elif snapshot.interrupts:
            # Paused at human review: ask again
            print(f"⏯  Resuming thread '{args.thread}' (waiting for your review)")
            next_input = Command(resume=ask_human(snapshot.values["script"]))
        elif snapshot.next:
            # Stopped mid-run (crash, Ctrl+C, network error): just continue.
            # Work that already finished (e.g. scenes already produced) is NOT redone.
            print(f"⏯  Continuing unfinished thread '{args.thread}' at {snapshot.next}")
            next_input = None
        elif snapshot.values:
            print(f"Thread '{args.thread}' is already finished. Use a new --thread name.")
            return
        else:
            if not args.topic:
                parser.error("topic is required for a new thread")
            print(f"▶  Starting thread '{args.thread}': {args.topic}")
            next_input = {"topic": args.topic, "target_minutes": args.minutes}

        # Run -> pause at review -> ask human -> resume -> ... until the graph ends.
        while True:
            for update in graph.stream(next_input, config, stream_mode="updates"):
                print_update(update)

            snapshot = graph.get_state(config)
            if not snapshot.interrupts:  # not waiting for a human: the graph reached END
                break
            next_input = Command(resume=ask_human(snapshot.values["script"]))

        script = snapshot.values["script"]
        folder = f"{slugify(snapshot.values['topic'])}-{args.thread}" + ("-fork" if args.fork else "")
        out_dir = Path("output") / folder
        save_script(script, out_dir)
        print(f"\n✅ Script saved to {out_dir}/script.md")
        if snapshot.values.get("manifest_path"):
            print(f"✅ Media and manifest in {Path(snapshot.values['manifest_path']).parent}")
    except KeyboardInterrupt:
        print(f"\n⏸  Stopped. Everything is saved. Resume with: python main.py --thread {args.thread}")
    except Exception as e:  # noqa: BLE001
        # e.g. every narrator voice failed 3 times. Finished work is checkpointed.
        print(f"\n❌ Stopped: {type(e).__name__}: {str(e)[:200]}")
        print(f"   Finished steps are saved. Fix the problem, then resume with: python main.py --thread {args.thread}")
    finally:
        checkpointer.conn.close()


if __name__ == "__main__":
    main()