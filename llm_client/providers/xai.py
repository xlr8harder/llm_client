"""
X.AI provider implementation for LLM client.
"""
from .openai_style import OpenAIStyleProvider

XAI_API_BASE = "https://api.x.ai/v1"


class XAIProvider(OpenAIStyleProvider):
    """Provider implementation for X.AI's OpenAI-compatible API."""

    api_base = XAI_API_BASE
    api_key_env_var = "XAI_API_KEY"
    provider_name = "xai"
