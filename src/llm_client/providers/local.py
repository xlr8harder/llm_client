"""
Local OpenAI-compatible provider implementation.
"""

import os
from typing import Dict
from urllib.parse import urlsplit, urlunsplit

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

    def _resolve_chat_completion_target(self, model_id: str) -> tuple[str, str]:
        parsed = self._parse_model_endpoint(model_id)
        if parsed is None:
            return self._get_api_base(), model_id
        return parsed

    def _parse_model_endpoint(self, model_id: str) -> tuple[str, str] | None:
        raw_model_id = str(model_id)
        value = raw_model_id.strip()
        prefixed = False

        for prefix in ("local/", "openai_compatible/"):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                prefixed = True
                break

        if not value:
            raise ValueError(
                "Invalid local model_id. Expected '<host>:<port>/<model>' or '<base_url>::<model>'."
            )

        if "://" in value and "::" not in value:
            raise ValueError(
                "Full URL local model_ids must use '<base_url>::<model>', for example 'http://127.0.0.1:8000/v1::served-model'."
            )

        if "::" in value:
            base_url, served_model = value.split("::", 1)
            if not base_url.strip() or not served_model.strip():
                raise ValueError(
                    "Invalid local model_id. Expected '<base_url>::<model>' with a non-empty base URL and model."
                )
            if self._looks_like_endpoint(base_url):
                return self._normalize_api_base(base_url), served_model
            if prefixed:
                raise ValueError(
                    "Invalid local model_id. After 'local/', expected '<host>:<port>/<model>' or '<base_url>::<model>'."
                )
            return None

        endpoint, separator, served_model = value.partition("/")
        if separator and self._looks_like_endpoint(endpoint):
            if not served_model.strip():
                raise ValueError(
                    "Invalid local model_id. Expected '<host>:<port>/<model>' with a non-empty model."
                )
            return self._normalize_api_base(endpoint), served_model

        if prefixed:
            raise ValueError(
                "Invalid local model_id. After 'local/', expected '<host>:<port>/<model>' or '<base_url>::<model>'."
            )

        return None

    def _looks_like_endpoint(self, value: str) -> bool:
        stripped = value.strip()
        if not stripped:
            return False
        if "://" in stripped:
            parsed = urlsplit(stripped)
            return bool(parsed.scheme and parsed.netloc)
        endpoint = stripped.split("/", 1)[0]
        return (
            endpoint == "localhost"
            or endpoint.startswith("127.")
            or endpoint.startswith("0.0.0.0")
            or "." in endpoint
            or ":" in endpoint
        )

    def _normalize_api_base(self, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Local API base URL must be non-empty.")
        if "://" not in text:
            text = f"http://{text}"

        parsed = urlsplit(text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "Local API base URL must include an http(s) scheme and host."
            )

        path = parsed.path.rstrip("/")
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")].rstrip("/")
        if not path:
            path = "/v1"

        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

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
