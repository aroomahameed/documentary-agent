# 🎬 Documentary Agent

A multi-agent system built with **LangGraph** that turns a topic into a narrated mini-documentary video: research → fact-check → script → human review → visuals + voiceover → rendered video.

Each stage of this project teaches one core LangGraph concept.

---

## Quick start (Stage 1)

```bash
python main.py "History of Mughal architecture" --thread mughal-1   # new run, pauses for your review
python main.py --thread mughal-1                                    # resume a paused run
python main.py --thread mughal-1 --history                          # list all saved checkpoints
python main.py --thread mughal-1 --fork "Rewrite it for kids"       # time travel: new script version
python main.py --draw                                               # print the graph diagram
```

Output goes to `output/<topic>/script.md` and `script.json`.

## Project structure

```
doc_agent/
  state.py     # Pydantic models (LLM outputs) + graph state (TypedDict)
  prompts.py   # all prompts in one place
  llm.py       # provider-agnostic model (switch with MODEL env var)
  tools.py     # external tools (web search, later: images, TTS)
  nodes.py     # graph nodes
  graph.py     # wiring: nodes + edges
main.py        # CLI with live progress streaming
tests/         # offline tests with a fake LLM
```

---

## Roadmap

### ✅ Stage 1: Research → Script

```
START → planner ─(Send × N)→ researcher → script_writer → END
```

**Concepts:** `StateGraph`, TypedDict state, structured output with Pydantic, the **Send API** (dynamic parallel fan-out), **reducers** (`Annotated[list, operator.add]`), fan-in, `stream_mode="updates"`.

**Your tasks before moving on:**

- [ ] Run it on 3 different topics and read the scripts critically.
- [ ] Turn on LangSmith tracing (see `.env.example`) and look at a full trace. Confirm the researchers really ran in parallel.
- [ ] Remove `operator.add` from `research` in `state.py` and run it again. Note what breaks and _why_ in `LEARNINGS.md`.
- [ ] Add Wikipedia as a second source inside `research_subtopic`.

---

### ✅ Stage 2: Fact-checker with a feedback loop

```
... → researcher → fact_checker ─(issues & rounds < 2)→ re-research weak subtopics
                               └─(ok)→ script_writer
```

**Concepts:** conditional edges, cycles, loop limits, and your first **real ReAct agent**: the fact-checker _chooses_ its own tools (`web_search`, `wikipedia_lookup`) to verify claims.

**Build:**

- A `fact_checker` node that uses `create_react_agent` (from `langgraph.prebuilt`) or your own `ToolNode` loop, with tools decorated with `@tool`.
- It returns `fact_check_issues: list[str]` and which subtopics need more research.
- A router function: if there are issues and `research_rounds < 2`, Send only the weak subtopics back to `researcher`; otherwise go to `script_writer`.

**Watch out:** the `research` reducer _appends_. When re-researching, old notes stay. Decide how to replace them (hint: a custom reducer function that replaces notes by subtopic).

**Lesson:** compare the fixed workflow (Stage 1 researcher) with the agentic step (fact-checker) and write down when each one makes sense.

---

### ✅ Stage 3: Human review and persistence

```
... → script_writer → human_review ─(approve)→ END (for now)
                                   └─(edit / reject + feedback)→ script_writer
```

**Concepts:** `interrupt()`, `Command(resume=...)`, checkpointers, `thread_id`, resuming after a crash, **time travel**.

**Hints:**

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver   # pip install langgraph-checkpoint-sqlite

def human_review(state):
    decision = interrupt({"script": state["script"].model_dump(),
                          "question": "approve / reject (with feedback)?"})
    if decision["action"] == "approve":
        return Command(goto=END)
    return Command(goto="script_writer", update={"human_feedback": decision["feedback"]})

with SqliteSaver.from_conn_string("checkpoints.db") as cp:
    graph = build_graph(checkpointer=cp)
    config = {"configurable": {"thread_id": "mughal-1"}}
    graph.invoke(inputs, config)                                    # pauses at interrupt
    graph.invoke(Command(resume={"action": "approve"}), config)     # continues
```

**Tasks:**

- [ ] Kill the program while it's waiting for review. Restart it and resume the same `thread_id`.
- [ ] Use `graph.get_state_history(config)` to go back to the state _before_ the script was written and fork a new version with a different tone.

---

### Stage 4: Scene production subgraph

```
script approved ─(Send × scenes)→ [scene_subgraph] → collect
scene_subgraph:  find_visual → generate_voiceover → measure_duration
```

**Concepts:** **subgraphs** with their own state schema, map-reduce over scenes, retries.

**Tools:**

- Images: Pexels API (free) or Wikimedia Commons, searched with `visual_description`. Let an LLM pick the best of 5 candidates.
- Voice: `edge-tts` (free; English voices like `en-US-GuyNeural`, Urdu like `ur-PK-AsadNeural`)
  ```python
  import edge_tts
  await edge_tts.Communicate(text, "en-US-GuyNeural").save("scene_1.mp3")
  ```
- Add a `RetryPolicy` to flaky nodes (`builder.add_node("find_visual", fn, retry_policy=RetryPolicy(max_attempts=3))`).

**Watch out:** media files go to disk. Only _paths_ go in state, never raw bytes, because the checkpointer saves state after every step.

---

### Stage 5: Assembly and quality check

```
collect → quality_checker ─(broken scenes)→ re-run only those scenes
                         └─(ok)→ preview interrupt → render_video → END
```

**Concepts:** fixed pipeline steps inside an agent graph, a self-correcting loop, a second human checkpoint.

- `moviepy` (v2): `ImageClip(img).with_duration(d).with_audio(AudioFileClip(mp3))`, then `concatenate_videoclips`.
- The quality checker verifies files exist, durations make sense, and no scene is missing an image.
- Bonus: add a slow zoom (Ken Burns effect), subtitles, and background music.

**Task:** crash it on purpose during rendering, then resume. Only the unfinished work should re-run.

---

### Stage 6: UI with live streaming

**Concepts:** `stream_mode=["updates", "custom"]`, `get_stream_writer()` for custom progress events from inside nodes, and serving the graph over an API.

- A Streamlit or FastAPI + Next.js frontend: a topic box → live progress → script review form (resumes the interrupt) → video player.

---

### Stage 7: Multi-agent and style presets

- Turn research, writing and production into separate agents coordinated by a **supervisor**.
- Style presets: "Netflix", "Kids explainer", "YouTube Short (60s, vertical)". These change prompts, scene counts and aspect ratio.
- Bilingual: generate English and Urdu narration from the same research.

---

### Stage 8: MCP

- Expose your tools (`search_images`, `text_to_speech`, `render_video`) as an **MCP server** using the official `mcp` Python SDK.
- Connect your LangGraph agent to it with `langchain-mcp-adapters` instead of importing the functions directly.
- Connect Claude Desktop to the same server, so another agent can now make videos with your tools.

---
