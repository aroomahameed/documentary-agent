"""
Web app for the Documentary Agent (Stage 6).

    streamlit run app.py

The SAME graph as main.py, with a different "human": instead of input() in a
terminal, the two reviews are pages with buttons. Everything that makes this
work is LangGraph:
  - the checkpointer keeps every run, so any run can be reopened at any time
  - interrupts tell the app which page to show (script or production review)
  - Command(resume=...) sends the button press back into the graph
  - stream_mode=["updates", "custom"] + subgraphs=True shows live progress
"""

import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command  # noqa: E402

from doc_agent.graph import build_graph  # noqa: E402
from doc_agent.media import is_image  # noqa: E402
from doc_agent.runs import fork_before_script, list_threads, make_checkpointer, run_status  # noqa: E402
from doc_agent.styles import STYLES, get_style  # noqa: E402

st.set_page_config(page_title="Documentary Agent", page_icon="🎬", layout="wide")

STATUS_LABEL = {
    "script_review": "📝 waiting: script review",
    "production_review": "🎞 waiting: production review",
    "stopped": "⏸ stopped",
    "finished": "✅ finished",
    "new": "new",
}


@st.cache_resource
def get_app():
    """Built once per app process (Streamlit re-runs this file on every click)."""
    checkpointer = make_checkpointer()
    return build_graph(checkpointer=checkpointer), checkpointer


graph, checkpointer = get_app()


def cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def describe(node: str, data) -> str | None:
    """One readable line for a finished node of the MAIN graph."""
    if not isinstance(data, dict):
        return None
    if node == "planner":
        return f"📋 Plan: {data['title']} ({len(data['subtopics'])} subtopics)"
    if node == "researcher":
        note = data["research"][0]
        return f"🔎 Researched: {note.subtopic} ({len(note.facts)} facts)"
    if node == "fact_checker":
        return (f"🧐 Fact-check: {len(data['weak_subtopics'])} weak subtopic(s), "
                f"{len(data['unverified_claims'])} unverified claim(s)")
    if node == "script_writer":
        return f"✍️ Script ready: {len(data['script'].scenes)} scenes"
    if node == "start_production":
        return f"🎬 Producing scenes into {data['media_dir']}"
    if node == "scene_producer":
        p = data["produced_scenes"][0]
        return f"🎞 Scene {p.scene_number} done: {len(p.shots)} shot(s), {p.duration_sec:.0f}s"
    if node == "quality_check":
        n = len(data["quality_issues"])
        return "🔍 Quality check: " + (f"{n} issue(s)" if n else "all files OK")
    if node == "render_video":
        return "🎬 Video rendered"
    return None


def run(thread: str, graph_input) -> None:
    """Run the graph until it pauses or ends, showing live progress, then refresh the page."""
    with st.status("The agent is working…", expanded=True) as box:
        bar = None
        try:
            for namespace, mode, chunk in graph.stream(
                    graph_input, cfg(thread), stream_mode=["updates", "custom"], subgraphs=True):
                if mode == "custom":
                    if "percent" in chunk:                       # render progress -> a progress bar
                        bar = bar or box.progress(0, text="Rendering video")
                        bar.progress(chunk["percent"] / 100, text=f"Rendering video: {chunk['percent']}%")
                    else:
                        prefix = f"[scene {chunk['scene']}] " if "scene" in chunk else ""
                        box.caption(f"{prefix}{chunk['msg']}")
                elif not namespace:                             # main-graph updates only
                    for node, data in chunk.items():
                        line = describe(node, data)
                        if line:
                            box.write(line)
        except Exception as e:  # noqa: BLE001
            box.update(label="Stopped with an error", state="error")
            st.error(f"{type(e).__name__}: {str(e)[:300]}\n\nFinished steps are saved: "
                     "use **Continue** to retry from where it stopped.")
            return
        box.update(label="Done", state="complete", expanded=False)
    st.rerun()


# --- Sidebar: new run + past runs -------------------------------------------

with st.sidebar:
    st.header("🎬 Documentary Agent")
    with st.form("new_run", clear_on_submit=True):
        topic = st.text_input("Topic", placeholder="e.g. History of the Taj Mahal")
        style = st.selectbox("Style", list(STYLES), format_func=lambda s: STYLES[s].label)
        minutes = st.number_input("Length (minutes)", min_value=1, max_value=5, value=1,
                                  help="Ignored for the 60-second short")
        start = st.form_submit_button("Create documentary", type="primary")

    st.subheader("Runs")
    threads = list_threads(checkpointer)
    if threads:
        labels = {t: f"{t} · {STATUS_LABEL[run_status(graph.get_state(cfg(t)))]}" for t in threads}
        current = st.session_state.get("thread")
        index = threads.index(current) if current in threads else 0
        st.session_state.thread = st.radio("Open a run", threads, index=index,
                                           format_func=labels.get, label_visibility="collapsed")
    else:
        st.caption("No runs yet.")

if start and topic.strip():
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:30]
    thread = f"{slug}-{datetime.now():%H%M%S}"
    st.session_state.thread = thread
    st.title(topic)
    run(thread, {"topic": topic.strip(), "target_minutes": int(minutes), "style": style})

thread = st.session_state.get("thread")
if not thread:
    st.title("Documentary Agent")
    st.write("Enter a topic in the sidebar. The agent researches it, fact-checks the research "
             "and writes a script for **your review**. After you approve, it finds images and "
             "records narration, and you **review the scenes** before it renders the video.")
    st.stop()

snapshot = graph.get_state(cfg(thread))
values = snapshot.values
status = run_status(snapshot)
st.title(values.get("script").title if values.get("script") else values.get("topic", thread))
st.caption(f"Run `{thread}` · {get_style(values.get('style')).label} · {STATUS_LABEL[status]}")


# --- Pages, one per run status ---------------------------------------------

if status == "script_review":
    st.subheader("Review the script")
    for scene in values["script"].scenes:
        with st.container(border=True):
            st.markdown(f"**Scene {scene.scene_number}**")
            st.write(scene.narration)
            st.caption(f"🎥 {scene.visual_description}")
    feedback = st.text_area("Feedback (to get a revised version)")
    left, right = st.columns(2)
    if left.button("✅ Approve script", type="primary", use_container_width=True):
        run(thread, Command(resume={"action": "approve"}))
    if right.button("✏️ Revise with feedback", use_container_width=True, disabled=not feedback.strip()):
        run(thread, Command(resume={"action": "revise", "feedback": feedback.strip()}))

elif status == "production_review":
    st.subheader("Review the scenes")
    issues = values.get("quality_issues", [])
    if issues:
        st.warning("Quality check:\n\n" + "\n".join(f"- {i}" for i in issues))
    else:
        st.success("Quality check: all files OK")
    st.caption("Write a note under any scene you want redone, e.g. *use a map of the empire*.")

    # Note fields get a "round" in their key: after a redo the round goes up, so the
    # page shows NEW, empty fields (Streamlit remembers a field's text by its key).
    round_key = f"redo_round_{thread}"
    round_no = st.session_state.get(round_key, 0)
    notes = {}
    for p in values.get("produced_scenes", []):
        with st.container(border=True):
            st.markdown(f"**Scene {p.scene_number}** · {p.duration_sec:.0f}s · {len(p.shots)} shot(s)")
            shown = [s for s in p.shots if is_image(s.image_path)]   # never crash on a broken file
            if shown:
                st.image([s.image_path for s in shown], width=240,
                         caption=[f"{s.kind}: {s.description[:50]}" for s in shown])
            if len(shown) < len(p.shots):
                st.warning(f"{len(p.shots) - len(shown)} image file(s) are broken and can't be "
                           "shown. Redo this scene to replace them.")
            st.write(p.narration)
            st.audio(p.audio_path)
            for problem in p.problems:
                st.caption(f"⚠ {problem}")
            notes[p.scene_number] = st.text_input("Redo note", key=f"note_{thread}_{round_no}_{p.scene_number}",
                                                  placeholder="leave empty to keep this scene")

    redo = {n: note.strip() for n, note in notes.items() if note.strip()}
    c1, c2, c3 = st.columns(3)
    if c1.button("🎬 Render video", type="primary", use_container_width=True):
        run(thread, Command(resume={"action": "render"}))
    if c2.button("🔁 Redo", use_container_width=True, disabled=not redo,
                 help="Re-produce the scenes that have a redo note"):
        st.session_state[round_key] = round_no + 1      # next page: empty note fields
        run(thread, Command(resume={"action": "redo", "scenes": redo}))
    if c3.button("⏹ Stop (keep the media)", use_container_width=True):
        run(thread, Command(resume={"action": "stop"}))

elif status == "stopped":
    st.info(f"This run stopped before finishing (next step: {', '.join(snapshot.next)}). "
            "Everything finished so far is saved.")
    if st.button("▶ Continue", type="primary"):
        run(thread, None)

elif status == "finished":
    video = values.get("video_path")
    if video and Path(video).exists():
        st.video(video)
        with open(video, "rb") as f:
            st.download_button("⬇ Download MP4", f, file_name=Path(video).name, mime="video/mp4")
    elif values.get("media_dir"):
        st.info(f"Stopped without rendering. The media is in `{values['media_dir']}`.")

    manifest = values.get("manifest_path")
    if manifest and Path(manifest).exists():
        credits = json.loads(Path(manifest).read_text(encoding="utf-8")).get("credits", [])
        if credits:
            with st.expander(f"Image credits ({len(credits)})"):
                st.markdown("\n".join(f"- {c}" for c in credits))

    if values.get("script"):
        st.subheader("Make another version (time travel)")
        st.caption("Rewinds to just after the research, so only the script and media are made again.")
        notes = st.text_input("Notes for the new version", placeholder="e.g. rewrite it for kids")
        if st.button("⏪ Create new version", disabled=not notes.strip()):
            if fork_before_script(graph, cfg(thread), notes.strip()):
                run(thread, None)
            else:
                st.error("This run has no checkpoint from before the script was written.")