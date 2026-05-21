"""
Google Gemini Enterprise Agent Platform OpenAI-compatible provider.
"""

import os

from .openai_style import OpenAIStyleProvider

GOOGLE_AGENT_PLATFORM_API_BASE = "https://aiplatform.googleapis.com/v1"


class GoogleAgentPlatformProvider(OpenAIStyleProvider):
    """Provider for Agent Platform's OpenAI-compatible Chat Completions API."""

    api_key_env_var = "GOOGLE_API_KEY"
    provider_name = "google_agent_platform"
    default_location = "global"
    default_endpoint = "openapi"

    def _build_request_headers(self):
        return {
            "x-goog-api-key": self.get_api_key(),
            "Content-Type": "application/json",
        }

    def _get_api_base(self):
        project_id = os.getenv("GOOGLE_AGENT_PLATFORM_PROJECT_ID")
        if not project_id:
            raise ValueError(
                "Required environment variable 'GOOGLE_AGENT_PLATFORM_PROJECT_ID' is not set"
            )

        location = os.getenv("GOOGLE_AGENT_PLATFORM_LOCATION") or self.default_location
        endpoint = os.getenv("GOOGLE_AGENT_PLATFORM_ENDPOINT") or self.default_endpoint

        return (
            f"{GOOGLE_AGENT_PLATFORM_API_BASE}/projects/{project_id}"
            f"/locations/{location}/endpoints/{endpoint}"
        )
