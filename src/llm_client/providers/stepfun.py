"""
Stepfun provider implementation for LLM client.
"""

from .openai_style import OpenAIStyleProvider

STEPFUN_API_BASE = "https://api.stepfun.ai/v1"


class StepfunProvider(OpenAIStyleProvider):
    """Provider implementation for Stepfun's OpenAI-compatible API."""

    api_base = STEPFUN_API_BASE
    api_key_env_var = "STEPFUN_API_KEY"
    provider_name = "stepfun"
