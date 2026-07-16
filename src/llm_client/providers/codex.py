"""Legacy Codex provider backed by V2 Responses and OAuth."""

from __future__ import annotations

import os
from typing import Any, Dict

from ..codex_oauth import CODEX_API_BASE, CodexOAuthManager
from .openai_responses import OpenAIResponsesStyleProvider


class CodexProvider(OpenAIResponsesStyleProvider):
    """Compatibility provider using the V2 Codex OAuth and transport stack."""

    api_base = CODEX_API_BASE
    provider_name = "codex"
    force_stream = True

    def __init__(
        self,
        auth_manager: Any | None = None,
        *,
        transport=None,
        client_id: str | None = None,
    ):
        super().__init__(transport=transport)
        self.auth_manager = auth_manager
        self.client_id = client_id

    def _get_api_key_env_var(self) -> str:
        return "LLM_CLIENT_CODEX_CLIENT_ID"

    def _build_responses_headers(self) -> Dict[str, str]:
        manager = self.auth_manager
        if manager is None:
            client_id = self.client_id or os.getenv("LLM_CLIENT_CODEX_CLIENT_ID", "")
            if not client_id:
                raise ValueError(
                    "Codex OAuth requires LLM_CLIENT_CODEX_CLIENT_ID. Configure it and complete the llm_client Codex login flow."
                )
            manager = self.auth_manager = CodexOAuthManager.create(client_id=client_id)
        return {
            **manager.get_headers(),
            "originator": "llm_client",
            "User-Agent": "llm_client",
            "OpenAI-Beta": "responses=experimental",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

    def _normalize_responses_model_id(self, model_id: str) -> str:
        return model_id.removeprefix("codex/")

    def _v2_options(self, options: dict[str, Any]) -> dict[str, Any]:
        result = super()._v2_options(options)
        result["store"] = False
        result["stream"] = True
        result.setdefault("instructions", "You are a helpful assistant.")
        result.setdefault("text", {"verbosity": "low"})
        result.setdefault("include", ["reasoning.encrypted_content"])
        return result
