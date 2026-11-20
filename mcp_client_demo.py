"""
A tiny MCP CLIENT: starts mcp_server.py and uses its tools over the MCP protocol.
This is what Claude Desktop (or any MCP app) does behind the scenes.

    python mcp_client_demo.py
"""

import sys

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("🔧 Tools the server offers:")
            for tool in (await session.list_tools()).tools:
                print(f"   - {tool.name}: {tool.description.splitlines()[0]}")

            print("\n🎙  Urdu voices:")
            voices = await session.call_tool("list_voices", {"language": "ur"})
            print("  ", voices.content[0].text)

            print("\n🔎 search_images('Fatehpur Sikri'):")
            found = await session.call_tool("search_images", {"query": "Fatehpur Sikri", "count": 3})
            for block in found.content:
                print("  ", block.text[:150])

            print("\n🗣  narrate(...):")
            spoken = await session.call_tool("narrate", {
                "text": "Fatehpur Sikri was the capital of the Mughal Empire for fourteen years.",
                "filename": "mcp-demo"})
            print("  ", spoken.content[0].text)


if __name__ == "__main__":
    anyio.run(main)