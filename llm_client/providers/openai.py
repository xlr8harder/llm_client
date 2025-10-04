"""
OpenAI provider implementation for LLM client.
"""
from .openai_style import OpenAIStyleProvider

OPENAI_API_BASE = "https://api.openai.com/v1"


class OpenAIProvider(OpenAIStyleProvider):
    """Provider implementation for OpenAI API."""

    api_base = OPENAI_API_BASE
    api_key_env_var = "OPENAI_API_KEY"
    provider_name = "openai"
