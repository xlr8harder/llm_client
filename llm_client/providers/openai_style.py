"""
Shared base implementation for OpenAI-compatible providers.
"""
import json
from typing import Any, Dict, Optional

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
                headers = {**headers, "Accept": "text/event-stream"}

            if use_stream_transport:
                # Use urllib3 to enforce an overall timeout for the full request
                import urllib3
                from urllib3.util import Timeout as _Timeout

                http = urllib3.PoolManager()
                body_bytes = json.dumps(data).encode("utf-8")
                total_timeout = None
                # Interpret numeric timeout as overall total; if tuple provided, approximate with sum
                if isinstance(timeout, tuple) and len(timeout) == 2:
                    total_timeout = float(timeout[0]) + float(timeout[1])
                elif isinstance(timeout, (int, float)):
                    total_timeout = float(timeout)

                u3_timeout = _Timeout(total=total_timeout) if total_timeout is not None else _Timeout(total=None)
                u3_resp = http.request(
                    "POST",
                    f"{self._get_api_base()}/chat/completions",
                    body=body_bytes,
                    headers=headers,
                    preload_content=False,
                    timeout=u3_timeout,
                )

                if u3_resp.status != 200:
                    # Read small error payload for message
                    try:
                        err_bytes = u3_resp.read(1024)
                        err_text = err_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        err_text = ""
                    error_info = {
                        "type": "api_error",
                        "status_code": u3_resp.status,
                        "message": f"Error (HTTP {u3_resp.status}): {err_text[:200]}",
                        "raw_response": err_text,
                    }
                    return LLMResponse(
                        success=False,
                        error_info=error_info,
                        raw_provider_response=None,
                        is_retryable=u3_resp.status in [408, 425, 429, 500, 502, 503, 504],
                        context=context,
                    )

                return self._consume_streaming_response_urllib3(u3_resp, context)

            # Non-streaming path uses urllib3 as well
            import urllib3
            from urllib3.util import Timeout as _Timeout

            http = urllib3.PoolManager()
            body_bytes = json.dumps(data).encode("utf-8")
            u3_timeout = _Timeout(total=float(timeout)) if isinstance(timeout, (int, float)) else _Timeout(total=None)
            u3_resp = http.request(
                "POST",
                f"{self._get_api_base()}/chat/completions",
                body=body_bytes,
                headers=headers,
                preload_content=True,
                timeout=u3_timeout,
            )

            if u3_resp.status != 200:
                return self._handle_error_response(u3_resp, context)

            try:
                raw_response = json.loads(u3_resp.data.decode("utf-8", errors="ignore"))
            except Exception:
                raw_response = {}

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

        except Exception as e:
            # Map urllib3 exceptions to retryable/non-retryable
            try:
                from urllib3 import exceptions as u3e
            except Exception:
                u3e = None
            is_timeout = u3e and isinstance(e, (getattr(u3e, 'TimeoutError', tuple()), getattr(u3e, 'ReadTimeoutError', tuple()), getattr(u3e, 'ConnectTimeoutError', tuple())))
            is_ssl = u3e and isinstance(e, getattr(u3e, 'SSLError', tuple()))
            is_location = u3e and isinstance(e, getattr(u3e, 'LocationParseError', tuple()))
            if is_timeout:
                return LLMResponse(
                    success=False,
                    error_info={"type": "timeout", "message": str(e), "exception": str(e)},
                    is_retryable=True,
                    context=context,
                )
            if is_ssl or is_location:
                return LLMResponse(
                    success=False,
                    error_info={"type": "network_error", "message": str(e), "exception": str(e)},
                    is_retryable=False,
                    context=context,
                )
            return LLMResponse(
                success=False,
                error_info={"type": "network_error", "message": str(e), "exception": str(e)},
                is_retryable=True,
                context=context,
            )

    def _handle_error_response(self, response, context) -> LLMResponse:
        # Support both urllib3 and requests-like responses
        status_code = getattr(response, "status", None)
        if status_code is None:
            status_code = getattr(response, "status_code", None)
        is_retryable = status_code in [408, 425, 429, 500, 502, 503, 504]
        error_text = None
        error_json = None
        # Try to obtain text/bytes
        try:
            if hasattr(response, "data"):
                error_text = response.data.decode("utf-8", errors="ignore") if isinstance(response.data, (bytes, bytearray)) else str(response.data)
            elif hasattr(response, "text"):
                error_text = response.text
        except Exception:
            error_text = None

        if error_text:
            try:
                error_json = json.loads(error_text)
            except Exception:
                error_json = None

        error_message = self._extract_error_message(error_json, response)

        error_info = {
            "type": "api_error",
            "status_code": status_code,
            "message": error_message,
            "raw_response": (error_text or ""),
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
        Consume an OpenAI-compatible SSE response (requests), aggregate content, and return a final LLMResponse.
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
        except Exception as e:
            msg = str(e).lower()
            err_type = "timeout" if "timed out" in msg or "timeout" in msg else "network_error"
            return LLMResponse(
                success=False,
                error_info={
                    "type": err_type,
                    "message": str(e),
                    "exception": str(e),
                },
                is_retryable=True,
                context=context,
            )

    def _consume_streaming_response_urllib3(self, u3_response, context) -> LLMResponse:
        """
        Consume SSE stream using urllib3 HTTPResponse, aggregate content, and return final LLMResponse.
        Enforces overall timeout via urllib3's total timeout.
        """
        try:
            aggregated_content: str = ""
            last_event: Optional[Dict[str, Any]] = None
            finish_reason: Optional[str] = None
            model: Optional[str] = None
            resp_id: Optional[str] = None
            created: Optional[int] = None

            buffer = ""
            for chunk in u3_response.stream(amt=65536, decode_content=True):
                if not chunk:
                    continue
                if isinstance(chunk, bytes):
                    text = chunk.decode("utf-8", errors="ignore")
                else:
                    text = str(chunk)
                buffer += text

                while True:
                    if "\n" not in buffer:
                        break
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip("\r")
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                    else:
                        payload = line
                    if payload == "[DONE]":
                        # Finish cleanly
                        u3_response.close()
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

                    # Parse JSON event
                    try:
                        event = json.loads(payload)
                    except Exception:
                        continue
                    last_event = event
                    resp_id = event.get("id", resp_id)
                    created = event.get("created", created)
                    model = event.get("model", model)

                    try:
                        choices = event.get("choices") or []
                        if choices:
                            choice0 = choices[0]
                            finish_reason = choice0.get("finish_reason", finish_reason)
                            delta = choice0.get("delta") or {}
                            content_piece = delta.get("content")
                            if content_piece:
                                aggregated_content += content_piece
                            if not content_piece and isinstance(choice0.get("message"), dict):
                                msg_content = choice0["message"].get("content")
                                if msg_content:
                                    aggregated_content += msg_content
                    except Exception:
                        pass

                    if self._has_content_filter_error(event):
                        err = self._extract_content_filter_error(event)
                        u3_response.close()
                        return LLMResponse(
                            success=False,
                            error_info=err,
                            raw_provider_response=event,
                            is_retryable=False,
                            context=context,
                        )

            # If stream ended without [DONE], return what we have
            u3_response.close()
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
        except Exception as e:
            try:
                from urllib3 import exceptions as u3e
            except Exception:
                u3e = None
            is_timeout = u3e and isinstance(e, (getattr(u3e, 'TimeoutError', tuple()), getattr(u3e, 'ReadTimeoutError', tuple()), getattr(u3e, 'ConnectTimeoutError', tuple())))
            is_ssl = u3e and isinstance(e, getattr(u3e, 'SSLError', tuple()))
            is_location = u3e and isinstance(e, getattr(u3e, 'LocationParseError', tuple()))
            if is_timeout:
                err_type = 'timeout'; retryable = True
            elif is_ssl or is_location:
                err_type = 'network_error'; retryable = False
            else:
                err_type = 'network_error'; retryable = True
            return LLMResponse(
                success=False,
                error_info={"type": err_type, "message": str(e), "exception": str(e)},
                is_retryable=retryable,
                context=context,
            )
