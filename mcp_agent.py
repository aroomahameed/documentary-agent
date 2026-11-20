"""
Stage 8, Step 3: a LangGraph agent whose tools come from our MCP server.

The agent never imports doc_agent.media or doc_agent.video. It only knows the
tools the server advertises over MCP: their names, descriptions and arguments.
Swap in any other MCP server and the same agent can use it.

    python mcp_agent.py
    python mcp_agent.py "Find 2 artworks of Akbar's court and narrate one line about them in Urdu"
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from doc_agent.llm import get_llm

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DEFAULT_TASK = ("Find 3 openly licensed photos of Fatehpur Sikri, download the best one, "
                "and narrate one sentence about it in Urdu.")

SYSTEM_PROMPT = """You are a documentary assistant. Use the tools to do the task.
Pick images whose title matches the subject. When you narrate in a language,
write the sentence in that language. Finish with a short summary listing every
file you created and the credit line for each image you used."""


def server_config() -> dict:
    """How to start our MCP server. The same info Claude Desktop's config holds."""
    return {
        "documentary-tools": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(ROOT / "mcp_server.py")],
        }
    }


async def load_tools():
    client = MultiServerMCPClient(server_config())
    return await client.get_tools()  # MCP tools -> LangChain tools


async def run(task: str) -> str:
    tools = await load_tools()
    print("🔧 Tools from MCP:", ", ".join(t.name for t in tools))

    agent = create_agent(get_llm(), tools, system_prompt=SYSTEM_PROMPT)

    # Stream each step so you can watch the agent call MCP tools.
    final = ""
    async for step in agent.astream({"messages": [{"role": "user", "content": task}]},
                                    stream_mode="updates"):
        for node, update in step.items():
            for msg in update.get("messages", []):
                for call in getattr(msg, "tool_calls", None) or []:
                    print(f"🤖 calls {call['name']}({call['args']})")
                if msg.type == "tool":
                    print(f"📦 {msg.name} -> {str(msg.content)[:150]}")
                elif msg.type == "ai" and not msg.tool_calls:
                    final = msg.text
    return final


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or DEFAULT_TASK
    print("📝 Task:", task)
    print("\n✅", asyncio.run(run(task)))