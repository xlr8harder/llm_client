"""
Shared base implementation for OpenAI-compatible providers.
"""
import json
from typing import Any, Dict, Optional

import requests

from ..base import LLMProvider, LLMResponse


class OpenAIStyleProvider(LLMProvider):
    """Base class for providers that implement the OpenAI-compatible chat API."""

    api_base: Optional[str] = None
    api_key_env_var: Optional[str] = None
    provider_name: Optional[str] = None
    default_timeout: int = 60
    default_max_tokens: int = 4096

    def _get_api_key_env_var(self) -> str:
        if not self.api_key_env_var:
            raise NotImplementedError("Subclasses must define 'api_key_env_var'")
        return self.api_key_env_var

    def _get_api_base(self) -> str:
        if not self.api_base:
            raise NotImplementedError("Subclasses must define 'api_base'")
        return self.api_base

    def _get_provider_name(self) -> str:
        if not self.provider_name:
            raise NotImplementedError("Subclasses must define 'provider_name'")
        return self.provider_name

    def _build_request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_api_key()}",
            "Content-Type": "application/json",
        }

    def make_chat_completion_request(self, messages, model_id, context=None, **options) -> LLMResponse:
        timeout = options.pop("timeout", self.default_timeout)
        max_tokens = options.pop("max_tokens", self.default_max_tokens)

        data: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if options:
            data.update(options)

        try:
            response = requests.post(
                url=f"{self._get_api_base()}/chat/completions",
                headers=self._build_request_headers(),
                data=json.dumps(data),
                timeout=timeout,
            )

            if response.status_code != 200:
                return self._handle_error_response(response, context)

            raw_response = response.json()

            if self._has_content_filter_error(raw_response):
                error_info = self._extract_content_filter_error(raw_response)
                return LLMResponse(
                    success=False,
                    error_info=error_info,
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context,
                )

            standardized_response = self._standardize_response(raw_response)

            return LLMResponse(
                success=True,
                standardized_response=standardized_response,
                raw_provider_response=raw_response,
                is_retryable=False,
                context=context,
            )

        except requests.exceptions.Timeout as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "timeout",
                    "message": f"Request timed out after {timeout} seconds: {str(e)}",
                    "exception": str(e),
                },
                is_retryable=True,
                context=context,
            )
        except requests.exceptions.RequestException as e:
            status_code = e.response.status_code if getattr(e, "response", None) else None
            return LLMResponse(
                success=False,
                error_info={
                    "type": "network_error",
                    "message": str(e),
                    "exception": str(e),
                    "status_code": status_code,
                },
                is_retryable=True,
                context=context,
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "unexpected_error",
                    "message": str(e),
                    "exception": str(e),
                },
                is_retryable=False,
                context=context,
            )

    def _handle_error_response(self, response, context) -> LLMResponse:
        status_code = response.status_code
        is_retryable = status_code in [408, 425, 429, 500, 502, 503, 504]

        try:
            error_json = response.json()
        except ValueError:
            error_json = None

        error_message = self._extract_error_message(error_json, response)

        error_info = {
            "type": "api_error",
            "status_code": status_code,
            "message": error_message,
            "raw_response": response.text,
        }

        return LLMResponse(
            success=False,
            error_info=error_info,
            raw_provider_response=error_json,
            is_retryable=is_retryable,
            context=context,
        )

    def _extract_error_message(self, error_json, response) -> str:
        if error_json and 'error' in error_json:
            error_obj = error_json['error']
            if isinstance(error_obj, dict) and 'message' in error_obj:
                return error_obj['message']
            return str(error_obj)

        status = getattr(response, "status_code", "unknown")
        text = getattr(response, "text", "")
        return f"Error (HTTP {status}): {text[:200]}"

    def _has_content_filter_error(self, response) -> bool:
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'content_filter' or 'error' in choice:
                return True
        return False

    def _extract_content_filter_error(self, response) -> Dict[str, str]:
        choice = response['choices'][0]
        if 'error' in choice:
            error_obj = choice['error']
            message = error_obj.get('message', 'Content filtered')
        else:
            message = "Response stopped due to content filter"

        return {
            "type": "content_filter",
            "message": message,
        }

    def _standardize_response(self, provider_response) -> Dict[str, Any]:
        standardized: Dict[str, Any] = {
            "id": provider_response.get("id"),
            "created": provider_response.get("created"),
            "model": provider_response.get("model"),
            "provider": self._get_provider_name(),
            "content": None,
            "usage": provider_response.get("usage", {}),
        }

        if "choices" in provider_response and provider_response["choices"]:
            choice = provider_response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                standardized["content"] = choice["message"]["content"]
            standardized["finish_reason"] = choice.get("finish_reason")

        return standardized
