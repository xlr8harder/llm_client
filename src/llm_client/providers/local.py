"""
Local OpenAI-compatible provider implementation.
"""

import os
from typing import Dict

from .openai_style import OpenAIStyleProvider

LOCAL_LLM_API_BASE = "http://127.0.0.1:8000/v1"
LOCAL_LLM_API_KEY_ENV_VAR = "LOCAL_LLM_API_KEY"
LOCAL_LLM_API_BASE_ENV_VAR = "LOCAL_LLM_BASE_URL"


class LocalProvider(OpenAIStyleProvider):
    """Provider for local OpenAI-compatible inference servers."""

    api_base = LOCAL_LLM_API_BASE
    api_key_env_var = LOCAL_LLM_API_KEY_ENV_VAR
    provider_name = "local"

    def _get_api_base(self) -> str:
        api_base = os.getenv(LOCAL_LLM_API_BASE_ENV_VAR) or self.api_base
        return api_base.rstrip("/")

    def get_api_key(self):
        """Return the optional local API key without requiring it to be set."""
        if self._api_key is None:
            self._api_key = os.getenv(self.api_key_env_var, "")
        return self._api_key

    def _build_request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers
