"""
External tools. In Stage 1 these are called directly by nodes (a fixed workflow
step), not chosen by the LLM. In Stage 2 you'll let the fact-checker agent
choose between tools itself. Knowing when to use which is the core lesson.
"""

import os

from tavily import TavilyClient


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Search the web. Returns a list of {title, url, content}."""
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=max_results)
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]

import requests
from langchain_core.tools import tool

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "documentary-agent/0.1 (learning project)"}


@tool
def search_web(query: str) -> str:
    """Search the web. Use this to verify specific claims (dates, numbers,
    names, events) or for information that may not be on Wikipedia.
    Returns titles, URLs and short snippets."""
    try:
        results = web_search(query, max_results=3)
    except Exception as e:
        return f"Search failed: {e}"   # errors go back to the LLM as text
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['content'][:500]}" for r in results)


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia. Best for well-established historical
    facts about people, buildings, empires and dates. Returns the intro
    of the best-matching article."""
    try:
        hits = requests.get(WIKI_API, headers=WIKI_HEADERS, timeout=10, params={
            "action": "query", "list": "search", "srsearch": topic,
            "srlimit": 1, "format": "json",
        }).json()["query"]["search"]
        if not hits:
            return "No Wikipedia article found."
        title = hits[0]["title"]

        pages = requests.get(WIKI_API, headers=WIKI_HEADERS, timeout=10, params={
            "action": "query", "prop": "extracts", "exintro": 1,
            "explaintext": 1, "titles": title, "format": "json",
        }).json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
        return f"{title}\n{url}\n{extract[:2000]}"
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"