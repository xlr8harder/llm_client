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

        # Opt-in streaming transport that returns a final aggregated response.
        # Supported interface: transport='stream'.
        # If the caller provides stream=True without transport='stream', raise an error
        # because we don't expose token-by-token streaming.
        transport = options.pop("transport", None)
        user_stream_flag = options.pop("stream", None)
        if user_stream_flag and transport != "stream":
            return LLMResponse(
                success=False,
                error_info={
                    "type": "invalid_option",
                    "message": "Direct stream=True is not supported. Use transport='stream' to enable streaming transport with aggregated output.",
                },
                is_retryable=False,
                context=context,
            )
        use_stream_transport = transport == "stream"

        data: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if options:
            data.update(options)

        # If using streaming transport, set provider stream flag (we manage this internally)
        if use_stream_transport:
            data["stream"] = True

        try:
            headers = self._build_request_headers()
            if use_stream_transport:
                # Hint for SSE responses; most OpenAI-compatible APIs ignore if not needed
                headers = {**headers, "Accept": "text/event-stream"}

            response = requests.post(
                url=f"{self._get_api_base()}/chat/completions",
                headers=headers,
                data=json.dumps(data),
                timeout=timeout,
                stream=use_stream_transport,
            )

            if response.status_code != 200:
                return self._handle_error_response(response, context)

            if use_stream_transport:
                return self._consume_streaming_response(response, context)

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

    def _consume_streaming_response(self, response, context) -> LLMResponse:
        """
        Consume an OpenAI-compatible SSE response, aggregate content, and return a final LLMResponse.
        """
        try:
            aggregated_content: str = ""
            last_event: Optional[Dict[str, Any]] = None
            finish_reason: Optional[str] = None
            model: Optional[str] = None
            resp_id: Optional[str] = None
            created: Optional[int] = None

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:"):].strip()
                if line == "[DONE]":
                    break
                # Some servers may send comments or keepalives starting with ':'
                if line.startswith(":"):
                    continue

                try:
                    event = json.loads(line)
                except Exception:
                    # Skip unparsable chunks; treat as transient noise
                    continue

                last_event = event
                resp_id = event.get("id", resp_id)
                created = event.get("created", created)
                model = event.get("model", model)

                # OpenAI-style streaming puts text deltas under choices[].delta.content
                try:
                    choices = event.get("choices") or []
                    if choices:
                        choice0 = choices[0]
                        finish_reason = choice0.get("finish_reason", finish_reason)
                        delta = choice0.get("delta") or {}
                        content_piece = delta.get("content")
                        if content_piece:
                            aggregated_content += content_piece
                        # Some providers might stream full messages per chunk
                        if not content_piece and isinstance(choice0.get("message"), dict):
                            msg_content = choice0["message"].get("content")
                            if msg_content:
                                aggregated_content += msg_content
                except Exception:
                    # Continue on minor schema oddities
                    pass

                # Content filter surfaced mid-stream
                if self._has_content_filter_error(event):
                    err = self._extract_content_filter_error(event)
                    return LLMResponse(
                        success=False,
                        error_info=err,
                        raw_provider_response=event,
                        is_retryable=False,
                        context=context,
                    )

            standardized = {
                "id": resp_id,
                "created": created,
                "model": model,
                "provider": self._get_provider_name(),
                "content": aggregated_content,
                "finish_reason": finish_reason or "stop",
                "usage": (last_event or {}).get("usage", {}),
            }

            return LLMResponse(
                success=True,
                standardized_response=standardized,
                raw_provider_response=last_event,
                is_retryable=False,
                context=context,
            )
        except requests.exceptions.Timeout as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "timeout",
                    "message": str(e),
                    "exception": str(e),
                },
                is_retryable=True,
                context=context,
            )
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "network_error",
                    "message": str(e),
                    "exception": str(e),
                    "status_code": e.response.status_code if getattr(e, "response", None) else None,
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
                is_retryable=True,
                context=context,
            )
