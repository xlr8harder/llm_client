"""
Chutes provider implementation for LLM client.
"""

from .openai_style import OpenAIStyleProvider

CHUTES_API_BASE = "https://llm.chutes.ai/v1"


class ChutesProvider(OpenAIStyleProvider):
    """Provider implementation for Chutes API."""

    api_base = CHUTES_API_BASE
    api_key_env_var = "CHUTES_API_TOKEN"
    provider_name = "chutes"
