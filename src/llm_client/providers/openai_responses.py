"""Legacy ``LLMResponse`` facade over the V2 Responses transport."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import httpx

from ..base import LLMProvider, LLMResponse, with_finish_reason_metadata
from ..oauth import OAuthError
from ..v2_client import Client


class _InvalidResponsesOption(ValueError):
    pass


class _ProviderAuth:
    def __init__(self, provider: "OpenAIResponsesStyleProvider"):
        self.provider = provider

    def get_headers(self) -> dict[str, str]:
        return self.provider._build_responses_headers()


class OpenAIResponsesStyleProvider(LLMProvider):
    """Project V2 Responses results into the stable legacy response shape."""

    api_base: Optional[str] = None
    api_key_env_var: Optional[str] = None
    provider_name: Optional[str] = None
    default_timeout: int = 60
    default_max_tokens: int = 4096
    force_stream: bool = False

    def __init__(self, *, transport: httpx.BaseTransport | None = None):
        super().__init__()
        self._responses_transport = transport

    def _get_api_key_env_var(self) -> str:
        if not self.api_key_env_var:
            raise NotImplementedError("Subclasses must define 'api_key_env_var'")
        return self.api_key_env_var

    def _get_api_base(self) -> str:
        if not self.api_base:
            raise NotImplementedError("Subclasses must define 'api_base'")
        return self.api_base.rstrip("/")

    def _get_provider_name(self) -> str:
        if not self.provider_name:
            raise NotImplementedError("Subclasses must define 'provider_name'")
        return self.provider_name

    def _build_responses_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_api_key()}",
            "Content-Type": "application/json",
        }

    def _normalize_responses_model_id(self, model_id: str) -> str:
        return model_id

    def _v2_options(self, options: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(options)
        transport = result.pop("transport", None)
        if transport not in {None, "stream"}:
            raise _InvalidResponsesOption(
                "Responses transport must be 'stream' when specified"
            )
        if result.get("stream") and transport != "stream" and not self.force_stream:
            raise _InvalidResponsesOption(
                "Direct stream=True is not supported. Use transport='stream' for aggregated streaming."
            )
        if transport == "stream" or self.force_stream:
            result["stream"] = True
        result.setdefault("max_tokens", self.default_max_tokens)
        return result

    def make_chat_completion_request(
        self, messages, model_id, context=None, **options
    ) -> LLMResponse:
        return self.make_responses_request(
            messages=messages, model_id=model_id, context=context, **options
        )

    def make_request(
        self,
        messages,
        model_id,
        context=None,
        request_format="chat_completions",
        **options,
    ) -> LLMResponse:
        normalized = (request_format or "chat_completions").replace("-", "_")
        if normalized not in {
            "chat_completion",
            "chat_completions",
            "response",
            "responses",
        }:
            return self._invalid_option(
                f"Unsupported request_format: {request_format}", context
            )
        return self.make_responses_request(
            messages=messages, model_id=model_id, context=context, **options
        )

    def make_responses_request(
        self, messages, model_id, context=None, **options
    ) -> LLMResponse:
        if options.get("top_k") is not None:
            return self._invalid_option(
                "top_k is not supported by the OpenAI Responses API. Remove --top-k for this model.",
                context,
            )
        timeout = options.pop("timeout", self.default_timeout)
        try:
            options = self._v2_options(options)
            provider = self._get_provider_name()
            normalized_model = self._normalize_responses_model_id(model_id)
            model_ref = f"{provider}/{normalized_model}"
            with Client(
                timeout=timeout,
                max_retries=0,
                transport=self._responses_transport,
                auth={provider: _ProviderAuth(self)},
            ) as client:
                result = client.model(
                    model_ref,
                    protocol="responses",
                    endpoint=self._get_api_base(),
                ).generate(messages=messages, **options)
        except _InvalidResponsesOption as error:
            return self._invalid_option(str(error), context)
        except (OSError, OAuthError, ValueError) as error:
            return LLMResponse(
                success=False,
                error_info={"type": "auth_error", "message": str(error)},
                request_format="responses",
                raw_response_format="llm_client.error",
                is_retryable=False,
                context=context,
            )
        return self._project_v2_response(result, context)

    def _project_v2_response(self, result, context) -> LLMResponse:
        raw = deepcopy(result.raw)
        if not result.ok:
            error = result.error
            category = error.category if error else "api_error"
            error_type = {
                "transport": "network_error",
                "rate_limit": "api_error",
                "authentication": "auth_error",
                "content_filter": "content_filter",
                "invalid_request": "invalid_option",
            }.get(category, category or "api_error")
            error_info = {
                "type": error_type,
                "message": error.message if error else "Responses request failed",
                "status_code": error.status_code if error else None,
                "raw_response": raw,
            }
            return LLMResponse(
                success=False,
                error_info=error_info,
                raw_provider_response=raw,
                request_format="responses",
                raw_response_format="openai.responses.error",
                is_retryable=bool(error.retryable) if error else False,
                context=context,
            )
        standardized = with_finish_reason_metadata(
            {
                "content": result.content or "",
                "reasoning": _responses_reasoning(raw),
                "model": raw.get("model") if isinstance(raw, dict) else None,
                "provider": self._get_provider_name(),
                "usage": deepcopy(result.usage),
            },
            source="response.status",
            value=result.native_finish_reason,
            normalized=result.finish_reason,
        )
        return LLMResponse(
            success=True,
            standardized_response=standardized,
            raw_provider_response=raw,
            request_format="responses",
            raw_response_format="openai.responses",
            is_retryable=False,
            context=context,
        )

    def _invalid_option(self, message, context) -> LLMResponse:
        return LLMResponse(
            success=False,
            error_info={"type": "invalid_option", "message": message},
            request_format="responses",
            raw_response_format="llm_client.error",
            is_retryable=False,
            context=context,
        )


def _responses_reasoning(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    parts = []
    for item in raw.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for summary in item.get("summary") or []:
            if isinstance(summary, dict) and isinstance(summary.get("text"), str):
                parts.append(summary["text"])
    return "".join(parts) or None
