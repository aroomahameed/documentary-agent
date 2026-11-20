"""The LangGraph side of MCP: tools arrive through langchain-mcp-adapters."""

import asyncio
import json

import mcp_agent


def test_agent_gets_all_server_tools():
    tools = asyncio.run(mcp_agent.load_tools())
    names = {t.name for t in tools}
    assert {"search_images", "download_image", "list_voices", "narrate", "render_video"} <= names


def test_adapted_tool_calls_the_real_server():
    tools = {t.name: t for t in asyncio.run(mcp_agent.load_tools())}
    result = asyncio.run(tools["list_voices"].ainvoke({"language": "ur"}))
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    assert "ur-PK" in text