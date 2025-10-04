"""
Fireworks AI provider implementation for LLM client.
"""
from .openai_style import OpenAIStyleProvider

FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"


class FireworksProvider(OpenAIStyleProvider):
    """Provider implementation for Fireworks AI API."""

    api_base = FIREWORKS_API_BASE
    api_key_env_var = "FIREWORKS_API_KEY"
    provider_name = "fireworks"
