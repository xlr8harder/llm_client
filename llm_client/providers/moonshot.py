"""
Moonshot provider implementation for LLM client.
"""
from .openai_style import OpenAIStyleProvider

MOONSHOT_API_BASE = "https://api.moonshot.ai/v1"


class MoonshotProvider(OpenAIStyleProvider):
    """Provider implementation for the Moonshot API."""

    api_base = MOONSHOT_API_BASE
    api_key_env_var = "MOONSHOT_API_KEY"
    provider_name = "moonshot"
