"""
Tests for the MCP server (Stage 8), on two levels:
  1. in-process: call the tools through the server (with fake image/voice services)
  2. over the real protocol: a client starts `python mcp_server.py` and talks MCP
     over stdin/stdout, exactly like Claude Desktop would
"""

import json
import sys
from pathlib import Path

import anyio
import pytest
from PIL import Image

import mcp_server

ROOT = Path(mcp_server.__file__).parent


def call(name, **arguments):
    """Call a tool the way an MCP client would; returns the structured result."""
    result = anyio.run(lambda: mcp_server.server.call_tool(name, arguments))
    content, structured = result if isinstance(result, tuple) else (result, None)
    if structured:                                      # typed results come back structured...
        return structured.get("result", structured)
    return json.loads(content[0].text)                  # ...a plain dict comes back as JSON text


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Send everything into a temp folder, and fake the network services."""
    monkeypatch.setattr(mcp_server, "MCP_FILES", tmp_path / "mcp")
    monkeypatch.setattr(mcp_server, "OUTPUT", tmp_path)
    monkeypatch.setattr(mcp_server, "search_photos", lambda q, n=5, kind="photo": [
        {"alt": f"{q} {i}", "url": f"https://img/{i}", "author": "A", "license": "CC0",
         "page": "https://p", "thumb": "t"} for i in range(n)])

    def fake_download(url, path, fallback_url=None):
        Image.new("RGB", (8, 8)).save(path, "JPEG")
        return path
    monkeypatch.setattr(mcp_server, "download", fake_download)
    monkeypatch.setattr(mcp_server, "text_to_speech",
                        lambda text, path, voice=None, rate=None, pitch=None: path.write_bytes(b"mp3") or path)
    monkeypatch.setattr(mcp_server, "audio_duration", lambda path: 4.25)
    return tmp_path


# --- 1. In-process -----------------------------------------------------------

def test_tools_are_listed_with_descriptions():
    tools = {t.name: t for t in anyio.run(mcp_server.server.list_tools)}
    assert set(tools) == {"search_images", "download_image", "list_voices", "narrate", "render_video"}
    assert "openly licensed" in tools["search_images"].description
    assert tools["search_images"].annotations.readOnlyHint is True
    assert "query" in tools["search_images"].inputSchema["properties"]


def test_search_images_caps_the_count(files):
    results = call("search_images", query="Taj Mahal", count=50)
    assert len(results) == 10
    assert set(results[0]) == {"title", "url", "author", "license", "page"}


def test_files_stay_inside_the_output_folder(files):
    path = Path(call("download_image", url="https://img/1", filename="../../../windows/evil"))
    assert path.parent == files / "mcp" and path.name == "evil.jpg"


def test_narrate_validates_the_voice(files):
    ok = call("narrate", text="Hello", filename="hi", language="ur")
    assert ok["voice"] == "ur-PK-AsadNeural" and ok["seconds"] == 4.2

    with pytest.raises(Exception, match="Unknown voice"):
        call("narrate", text="Hello", filename="hi", language="ur", voice="en-GB-RyanNeural")


def test_narrate_runs_the_real_speech_code_inside_the_servers_event_loop(files, monkeypatch):
    """Regression test: the REAL text_to_speech (which uses asyncio.run) must work when
    called by the server, which is already running an event loop. Only the network
    part (edge-tts) is faked. This failed before narrate ran it in a worker thread."""
    from doc_agent import media

    class FakeCommunicate:
        def __init__(self, text, voice, rate=None, pitch=None):
            pass

        async def save(self, path):
            Path(path).write_bytes(b"mp3")

    monkeypatch.setattr(media.edge_tts, "Communicate", FakeCommunicate)
    monkeypatch.setattr(mcp_server, "text_to_speech", media.text_to_speech)   # the real one

    result = call("narrate", text="Hello", filename="real-code")
    assert Path(result["path"]).read_bytes() == b"mp3"


def test_render_refuses_files_outside_the_output_folder(files):
    with pytest.raises(Exception, match="inside the project's output folder"):
        call("render_video", manifest_path=str(ROOT / "README.md"))


def test_videos_resource(files):
    (files / "doc").mkdir()
    (files / "doc" / "documentary.mp4").write_bytes(b"x" * 2_000_000)
    monkey_root = files.parent
    original_root = mcp_server.ROOT
    mcp_server.ROOT = monkey_root
    try:
        contents = list(anyio.run(lambda: mcp_server.server.read_resource("documentaries://videos")))
    finally:
        mcp_server.ROOT = original_root
    videos = json.loads(contents[0].content)
    assert videos[0]["path"].endswith("documentary.mp4") and videos[0]["mb"] == 2.0


# --- 2. Over the real MCP protocol (a separate server process) ---------------

def test_real_client_over_stdio():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def session_run():
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = [t.name for t in (await session.list_tools()).tools]
                voices = await session.call_tool("list_voices", {"language": "ur"})
                bad = await session.call_tool("narrate", {"text": "x", "filename": "f", "voice": "nope"})
                resources = [str(r.uri) for r in (await session.list_resources()).resources]
                return tools, voices, bad, resources

    tools, voices, bad, resources = anyio.run(session_run)
    assert "narrate" in tools and "render_video" in tools
    assert "ur-PK-AsadNeural" in voices.content[0].text
    assert bad.isError and "Unknown voice" in bad.content[0].text   # errors go back to the client
    assert "documentaries://videos" in resources