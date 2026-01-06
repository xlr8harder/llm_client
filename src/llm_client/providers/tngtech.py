"""
TNGTech provider implementation for LLM client.
"""

from .openai_style import OpenAIStyleProvider

API_BASE = "https://chat.model.tngtech.com/v1"


class TNGTechProvider(OpenAIStyleProvider):
    """Provider implementation for TNGTech's OpenAI-compatible API."""

    api_base = API_BASE
    api_key_env_var = "TNGTECH_API_KEY"
    provider_name = "tngtech"
