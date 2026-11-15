import os
from functools import lru_cache

from langchain.chat_models import init_chat_model


@lru_cache
def get_llm():
    """
    Provider-agnostic chat model. Switch providers with the MODEL env var, e.g.
      openai:gpt-4o-mini
      groq:llama-3.3-70b-versatile          (pip install langchain-groq)
      google_genai:gemini-2.0-flash         (pip install langchain-google-genai)
    """
    return init_chat_model(os.getenv("MODEL", "anthropic:claude-haiku-5-5"), max_tokens=4096)
