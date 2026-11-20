"""
Tests for the web app, with Streamlit's AppTest: it runs app.py without a
browser, so tests can click buttons and read the page. Same fakes as test_graph.py.

    python -m pytest -q
"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from test_graph import fake  # noqa: F401  (the fixture with fake LLM, search, voice, render)

APP = "app.py"


@pytest.fixture
def app(fake, monkeypatch, tmp_path, request):  # noqa: F811
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "test.db"))
    st.cache_resource.clear()                  # a fresh graph + database for every test
    at = AppTest.from_file(str(request.config.rootpath / APP), default_timeout=30)
    at.run()
    yield at
    st.cache_resource.clear()


def button(at, label_start):
    return next(b for b in at.button if b.label.startswith(label_start))


def create(at, topic="History of the Taj Mahal"):
    at.sidebar.text_input[0].input(topic)
    at.sidebar.button[0].click().run()         # the form's "Create documentary" button
    return at


def test_start_page(app):
    assert not app.exception
    assert app.title[0].value == "Documentary Agent"
    assert "No runs yet." in [c.value for c in app.sidebar.caption]


def test_new_run_stops_at_script_review(app, fake):  # noqa: F811
    create(app)

    assert not app.exception
    assert app.subheader[0].value == "Review the script"
    assert "Draft 1." in [m.value for m in app.markdown] + [w.value for w in app.get("text")]
    assert fake.tts_calls == []                # nothing produced before approval


def test_approve_then_production_review_shows_scenes(app, fake):  # noqa: F811
    create(app)
    button(app, "✅ Approve").click().run()

    assert not app.exception
    assert app.subheader[0].value == "Review the scenes"
    assert len(app.main.text_input) == 2            # one redo note per scene
    assert len(fake.tts_calls) == 2


def test_redo_note_reproduces_only_that_scene(app, fake):  # noqa: F811
    create(app)
    button(app, "✅ Approve").click().run()
    app.main.text_input[1].input("use a map of the empire")      # note under scene 2
    app.run()
    button(app, "🔁 Redo").click().run()

    assert not app.exception
    assert len(fake.tts_calls) == 3                          # only scene 2 again
    assert "use a map of the empire" in fake.plan_requests[-1]
    assert app.subheader[0].value == "Review the scenes"     # asked again


def test_render_finishes_with_video(app, fake):  # noqa: F811
    create(app)
    button(app, "✅ Approve").click().run()
    button(app, "🎬 Render").click().run()

    assert not app.exception
    assert len(fake.renders) == 1
    assert "finished" in app.caption[0].value
    assert app.subheader[0].value == "Make another version (time travel)"


def test_time_travel_from_the_app(app, fake):  # noqa: F811
    create(app)
    button(app, "✅ Approve").click().run()
    button(app, "🎬 Render").click().run()
    searches = len(fake.researched)

    app.main.text_input[0].input("rewrite it for kids")
    app.run()
    button(app, "⏪ Create new version").click().run()

    assert not app.exception
    assert len(fake.researched) == searches                  # research was reused
    assert "rewrite it for kids" in fake.writer_prompts[-1]
    assert app.subheader[0].value == "Review the script"     # the new version waits for review


def test_broken_image_file_does_not_crash_the_gallery(app, fake):  # noqa: F811
    fake.broken_image = "scene_02_shot_1.jpg"          # e.g. an HTML error page saved as .jpg
    create(app)
    button(app, "✅ Approve").click().run()

    assert not app.exception
    assert app.subheader[0].value == "Review the scenes"
    assert any("broken" in w.value for w in app.warning)


def test_redo_clears_the_notes(app, fake):  # noqa: F811
    create(app)
    button(app, "✅ Approve").click().run()
    app.main.text_input[0].input("more paintings")
    app.run()
    assert button(app, "🔁 Redo").label == "🔁 Redo"          # no counter in the label
    button(app, "🔁 Redo").click().run()

    assert not app.exception
    assert all(t.value == "" for t in app.main.text_input)   # fresh, empty note fields
    assert button(app, "🔁 Redo").disabled                    # nothing to redo yet


def test_urdu_narration_is_shown_right_to_left(app, fake):  # noqa: F811
    app.sidebar.selectbox[1].select("ur")                    # the "Narration" menu
    create(app)

    assert not app.exception
    assert "اردو" in app.caption[0].value                     # the run's language is shown
    assert any('dir="rtl"' in m.value for m in app.markdown)  # narration aligned right-to-left


def test_editor_suggestions_are_prefilled_in_the_app(app, fake):  # noqa: F811
    from doc_agent import editor
    fake.editor_report = editor.EditorReport(summary="Scene 2 needs a map.", suggestions=[
        editor.Suggestion(scene_number=2, note="show a map of the empire", reason="r")])
    create(app)
    button(app, "✅ Approve").click().run()

    assert any("AI editor" in i.value for i in app.info)
    assert app.main.text_input[1].value == "show a map of the empire"   # scene 2's note field
    assert not button(app, "🔁 Redo").disabled                          # one click to accept