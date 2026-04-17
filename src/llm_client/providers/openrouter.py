"""
OpenRouter provider implementation for LLM client.
"""

import json
import os

from ..base import LLMProvider, LLMResponse

# API Endpoint Constants
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterProvider(LLMProvider):
    """Provider implementation for OpenRouter API"""

    def _get_api_key_env_var(self):
        return "OPENROUTER_API_KEY"

    def make_request(
        self,
        messages,
        model_id,
        context=None,
        request_format="chat_completions",
        **options,
    ):
        """Make a request using one of OpenRouter's supported API shapes."""
        normalized_format = self._normalize_request_format(request_format)
        if normalized_format == "chat_completions":
            return self.make_chat_completion_request(
                messages=messages,
                model_id=model_id,
                context=context,
                **options,
            )
        if normalized_format == "anthropic_messages":
            response = self._make_anthropic_messages_request(
                messages=messages,
                model_id=model_id,
                context=context,
                **options,
            )
            return self._annotate_response(response, normalized_format)
        return self._invalid_request_format_response(
            request_format=request_format,
            normalized_format=normalized_format,
            context=context,
        )

    def make_chat_completion_request(self, messages, model_id, context=None, **options):
        """Make a request to the OpenRouter chat-completions API."""
        response = self._make_chat_completion_request(
            messages=messages,
            model_id=model_id,
            context=context,
            **options,
        )
        return self._annotate_response(response, "chat_completions")

    def _normalize_request_format(self, request_format):
        normalized = (request_format or "chat_completions").replace("-", "_")
        if normalized in {"chat_completion", "chat_completions"}:
            return "chat_completions"
        if normalized in {"anthropic_message", "anthropic_messages", "anthropic_messages_api"}:
            return "anthropic_messages"
        return normalized

    def _invalid_request_format_response(
        self, request_format, normalized_format, context=None
    ):
        return LLMResponse(
            success=False,
            error_info={
                "type": "invalid_option",
                "message": f"Unsupported request_format: {request_format}",
            },
            request_format=normalized_format,
            raw_response_format="llm_client.error",
            is_retryable=False,
            context=context,
        )

    def _annotate_response(self, response, request_format):
        if not isinstance(response, LLMResponse):
            return response
        response.request_format = response.request_format or request_format
        if response.raw_response_format is None:
            response.raw_response_format = self._classify_raw_response(
                response, request_format
            )
        return response

    def _classify_raw_response(self, response, request_format):
        raw = response.raw_provider_response
        error_type = (response.error_info or {}).get("type")
        if raw is None:
            if error_type in {"invalid_option", "network_error", "timeout"}:
                return "llm_client.error"
            return "openrouter.http_error"
        if isinstance(raw, dict):
            if raw.get("error") is not None:
                return "openrouter.error"
            if request_format == "anthropic_messages" and isinstance(
                raw.get("content"), list
            ):
                return "openrouter.anthropic_messages"
            if request_format == "chat_completions" and isinstance(
                raw.get("choices"), list
            ):
                return "openrouter.chat_completions"
        return f"openrouter.{request_format}.unknown"

    def _build_headers(self):
        return {
            "Authorization": f"Bearer {self.get_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv(
                "OPENROUTER_REFERRER", "https://SpeechMap.ai"
            ),
            "X-Title": os.getenv("OPENROUTER_TITLE", "SpeechMap.ai"),
        }

    def _provider_routing_from_options(self, options):
        provider_routing = {}
        only = options.pop("only", None)
        allow_list = options.pop("allow_list", None)
        ignore_list = options.pop("ignore_list", None)

        if only:
            provider_routing["order"] = list(only)
            provider_routing["allow_fallbacks"] = False
        elif allow_list:
            provider_routing["order"] = list(allow_list)
            provider_routing["allow_fallbacks"] = False

        if ignore_list:
            provider_routing["ignore"] = list(ignore_list)

        return provider_routing

    def _make_chat_completion_request(self, messages, model_id, context=None, **options):
        """
        Make a request to the OpenRouter API

        Args:
            messages: List of message objects with 'role' and 'content' keys
            model_id: OpenRouter model identifier
            context: Optional context object to include in the response
            **options: Additional options (including allow_list/only and ignore_list for providers)

        Returns:
            LLMResponse object with standardized result
        """
        try:
            url = f"{OPENROUTER_API_BASE}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.get_api_key()}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv(
                    "OPENROUTER_REFERRER", "https://SpeechMap.ai"
                ),
                "X-Title": os.getenv("OPENROUTER_TITLE", "SpeechMap.ai"),
            }

            timeout = options.pop("timeout", 60)
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

            data = {
                "model": model_id,
                "messages": messages,
                "max_tokens": options.pop("max_tokens", 4096),
            }

            # Provider routing controls
            provider_routing = {}
            only = options.pop("only", None)
            allow_list = options.pop("allow_list", None)
            ignore_list = options.pop("ignore_list", None)

            if only:
                provider_routing["order"] = list(only)
                provider_routing["allow_fallbacks"] = False
            elif allow_list:
                provider_routing["order"] = list(allow_list)
                provider_routing["allow_fallbacks"] = False

            if ignore_list:
                provider_routing["ignore"] = list(ignore_list)

            if provider_routing:
                data["provider"] = provider_routing

            # Remaining options passthrough
            data.update(options)

            # Ensure provider gets stream=True if we are using streaming transport
            if use_stream_transport:
                data["stream"] = True

            # For streaming, request SSE and stream the HTTP response
            req_headers = dict(headers)
            if use_stream_transport:
                req_headers["Accept"] = "text/event-stream"

            if use_stream_transport:
                # Use urllib3 to enforce an overall timeout for the full streamed request
                import urllib3
                from urllib3.util import Timeout as _Timeout

                http = urllib3.PoolManager()
                body_bytes = json.dumps(data).encode("utf-8")
                total_timeout = None
                if isinstance(timeout, tuple) and len(timeout) == 2:
                    total_timeout = float(timeout[0]) + float(timeout[1])
                elif isinstance(timeout, (int, float)):
                    total_timeout = float(timeout)

                u3_timeout = (
                    _Timeout(total=total_timeout)
                    if total_timeout is not None
                    else _Timeout(total=None)
                )
                u3_resp = http.request(
                    "POST",
                    url,
                    body=body_bytes,
                    headers=req_headers,
                    preload_content=False,
                    timeout=u3_timeout,
                )

                if u3_resp.status != 200:
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
                        is_retryable=u3_resp.status
                        in [408, 425, 429, 500, 502, 503, 504],
                        context=context,
                    )

                return self._consume_streaming_response_urllib3(u3_resp, context)

            # Non-streaming path: urllib3
            import urllib3
            from urllib3.util import Timeout as _Timeout

            http = urllib3.PoolManager()
            body_bytes = json.dumps(data).encode("utf-8")
            u3_timeout = (
                _Timeout(total=float(timeout))
                if isinstance(timeout, (int, float))
                else _Timeout(total=None)
            )
            u3_resp = http.request(
                "POST",
                url,
                body=body_bytes,
                headers=req_headers,
                timeout=u3_timeout,
                preload_content=True,
            )

            if u3_resp.status != 200:
                return self._handle_error_response(u3_resp, context)

            try:
                raw_response = json.loads(u3_resp.data.decode("utf-8", errors="ignore"))
            except Exception:
                raw_response = {}
            if isinstance(raw_response, dict) and "error" in raw_response:
                error_message = self._extract_error_message(raw_response, "")
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "api_error",
                        "status_code": getattr(u3_resp, "status", None),
                        "message": error_message,
                        "raw_response": u3_resp.data.decode("utf-8", errors="ignore")
                        if getattr(u3_resp, "data", None)
                        else "",
                    },
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context,
                )
            standardized_response = self._standardize_response(raw_response)

            if self._has_content_filter_error(raw_response):
                error_info = self._extract_content_filter_error(raw_response)
                return LLMResponse(
                    success=False,
                    error_info=error_info,
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context,
                )
            # Final sanity: if no content despite 200 and no explicit error, treat as non-retryable
            content_text = (standardized_response.get("content") or "").strip()
            if content_text == "":
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "content_filter",
                        "message": "Response contained no content.",
                    },
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context,
                )

            return LLMResponse(
                success=True,
                standardized_response=standardized_response,
                raw_provider_response=raw_response,
                is_retryable=False,
                context=context,
            )

        except Exception as e:
            try:
                from urllib3 import exceptions as u3e
            except Exception:
                u3e = None
            is_timeout = u3e and isinstance(
                e,
                (
                    getattr(u3e, "TimeoutError", tuple()),
                    getattr(u3e, "ReadTimeoutError", tuple()),
                    getattr(u3e, "ConnectTimeoutError", tuple()),
                ),
            )
            is_ssl = u3e and isinstance(e, getattr(u3e, "SSLError", tuple()))
            is_location = u3e and isinstance(
                e, getattr(u3e, "LocationParseError", tuple())
            )
            if is_timeout:
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
            if is_ssl or is_location:
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "network_error",
                        "message": str(e),
                        "exception": str(e),
                    },
                    is_retryable=False,
                    context=context,
                )
            return LLMResponse(
                success=False,
                error_info={
                    "type": "network_error",
                    "message": str(e),
                    "exception": str(e),
                },
                is_retryable=True,
                context=context,
            )

    def _make_anthropic_messages_request(
        self, messages, model_id, context=None, **options
    ):
        """Make a request to OpenRouter's Anthropic Messages API."""
        try:
            timeout = options.pop("timeout", 60)
            transport = options.pop("transport", None)
            user_stream_flag = options.pop("stream", None)
            if user_stream_flag or transport == "stream":
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "invalid_option",
                        "message": "Streaming transport is not supported for request_format='anthropic_messages'.",
                    },
                    is_retryable=False,
                    context=context,
                )
            if transport is not None:
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "invalid_option",
                        "message": f"Unsupported transport for request_format='anthropic_messages': {transport}",
                    },
                    is_retryable=False,
                    context=context,
                )

            system_content, request_messages = self._split_anthropic_system_messages(
                messages
            )
            data = {
                "model": model_id,
                "messages": request_messages,
                "max_tokens": options.pop("max_tokens", 4096),
            }
            if system_content is not None:
                data["system"] = system_content

            provider_routing = self._provider_routing_from_options(options)
            if provider_routing:
                data["provider"] = provider_routing

            data.update(options)

            import urllib3
            from urllib3.util import Timeout as _Timeout

            http = urllib3.PoolManager()
            body_bytes = json.dumps(data).encode("utf-8")
            u3_timeout = (
                _Timeout(total=float(timeout))
                if isinstance(timeout, (int, float))
                else _Timeout(total=None)
            )
            u3_resp = http.request(
                "POST",
                f"{OPENROUTER_API_BASE}/messages",
                body=body_bytes,
                headers=self._build_headers(),
                timeout=u3_timeout,
                preload_content=True,
            )

            if u3_resp.status != 200:
                return self._handle_error_response(u3_resp, context)

            try:
                raw_response = json.loads(u3_resp.data.decode("utf-8", errors="ignore"))
            except Exception:
                raw_response = {}

            if isinstance(raw_response, dict) and "error" in raw_response:
                error_message = self._extract_error_message(raw_response, "")
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "api_error",
                        "status_code": getattr(u3_resp, "status", None),
                        "message": error_message,
                        "raw_response": u3_resp.data.decode("utf-8", errors="ignore")
                        if getattr(u3_resp, "data", None)
                        else "",
                    },
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context,
                )

            standardized_response = self._standardize_anthropic_messages_response(
                raw_response
            )
            content_text = (standardized_response.get("content") or "").strip()
            if content_text == "":
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "content_filter",
                        "message": "Response contained no content.",
                    },
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context,
                )

            return LLMResponse(
                success=True,
                standardized_response=standardized_response,
                raw_provider_response=raw_response,
                is_retryable=False,
                context=context,
            )
        except Exception as e:
            try:
                from urllib3 import exceptions as u3e
            except Exception:
                u3e = None
            is_timeout = u3e and isinstance(
                e,
                (
                    getattr(u3e, "TimeoutError", tuple()),
                    getattr(u3e, "ReadTimeoutError", tuple()),
                    getattr(u3e, "ConnectTimeoutError", tuple()),
                ),
            )
            is_ssl = u3e and isinstance(e, getattr(u3e, "SSLError", tuple()))
            is_location = u3e and isinstance(
                e, getattr(u3e, "LocationParseError", tuple())
            )
            if is_timeout:
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
            if is_ssl or is_location:
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "network_error",
                        "message": str(e),
                        "exception": str(e),
                    },
                    is_retryable=False,
                    context=context,
                )
            return LLMResponse(
                success=False,
                error_info={
                    "type": "network_error",
                    "message": str(e),
                    "exception": str(e),
                },
                is_retryable=True,
                context=context,
            )

    def _split_anthropic_system_messages(self, messages):
        system_parts = []
        request_messages = []
        for message in messages:
            if not isinstance(message, dict):
                request_messages.append(message)
                continue
            if message.get("role") == "system":
                system_parts.append(message.get("content", ""))
                continue
            request_messages.append(dict(message))
        return self._merge_anthropic_system_content(system_parts), request_messages

    def _merge_anthropic_system_content(self, system_parts):
        if not system_parts:
            return None
        if all(isinstance(part, str) for part in system_parts):
            return "\n\n".join(part for part in system_parts if part)

        blocks = []
        for part in system_parts:
            if isinstance(part, str):
                if part:
                    blocks.append({"type": "text", "text": part})
            elif isinstance(part, list):
                blocks.extend(self._normalize_content_blocks(part))
            elif isinstance(part, dict):
                blocks.append(part)
            elif part is not None:
                blocks.append({"type": "text", "text": str(part)})
        return blocks

    def _normalize_content_blocks(self, blocks):
        normalized = []
        for block in blocks:
            if isinstance(block, str):
                normalized.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                normalized.append(block)
            elif block is not None:
                normalized.append({"type": "text", "text": str(block)})
        return normalized

    def _consume_streaming_response(self, response, context):
        """Consume OpenRouter SSE stream (requests) and build final response."""
        try:
            import json as _json

            aggregated = ""
            last_event = None
            finish_reason = None
            model = None
            resp_id = None
            created = None

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[len("data:") :].strip()
                if line == "[DONE]":
                    break
                if line.startswith(":"):
                    continue
                try:
                    evt = _json.loads(line)
                except Exception:
                    continue

                last_event = evt
                resp_id = evt.get("id", resp_id)
                created = evt.get("created", created)
                model = evt.get("model", model)

                try:
                    choices = evt.get("choices") or []
                    if choices:
                        c0 = choices[0]
                        finish_reason = c0.get("finish_reason", finish_reason)
                        delta = c0.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            aggregated += piece
                        if not piece and isinstance(c0.get("message"), dict):
                            msg_piece = c0["message"].get("content")
                            if msg_piece:
                                aggregated += msg_piece
                except Exception:
                    pass

                if self._has_content_filter_error(evt):
                    err = self._extract_content_filter_error(evt)
                    return LLMResponse(
                        success=False,
                        error_info=err,
                        raw_provider_response=evt,
                        is_retryable=False,
                        context=context,
                    )

            standardized = {
                "id": resp_id,
                "created": created,
                "model": model,
                "provider": "openrouter",
                "content": aggregated,
                "finish_reason": finish_reason or "stop",
                "usage": (last_event or {}).get("usage", {}),
            }
            if (aggregated or "").strip() == "":
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "content_filter",
                        "message": "Response contained no content.",
                    },
                    raw_provider_response=last_event,
                    is_retryable=False,
                    context=context,
                )
            return LLMResponse(
                success=True,
                standardized_response=standardized,
                raw_provider_response=last_event,
                is_retryable=False,
                context=context,
            )
        except Exception as e:
            msg = str(e).lower()
            err_type = (
                "timeout" if "timed out" in msg or "timeout" in msg else "network_error"
            )
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

    def _consume_streaming_response_urllib3(self, u3_response, context):
        """Consume OpenRouter SSE stream via urllib3 and build final response."""
        try:
            import json as _json

            aggregated = ""
            last_event = None
            finish_reason = None
            model = None
            resp_id = None
            created = None

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
                        payload = line[len("data:") :].strip()
                    else:
                        payload = line
                    if payload == "[DONE]":
                        u3_response.close()
                        standardized = {
                            "id": resp_id,
                            "created": created,
                            "model": model,
                            "provider": "openrouter",
                            "content": aggregated,
                            "finish_reason": finish_reason or "stop",
                            "usage": (last_event or {}).get("usage", {}),
                        }
                        if (aggregated or "").strip() == "":
                            return LLMResponse(
                                success=False,
                                error_info={
                                    "type": "content_filter",
                                    "message": "Response contained no content.",
                                },
                                raw_provider_response=last_event,
                                is_retryable=False,
                                context=context,
                            )
                        return LLMResponse(
                            success=True,
                            standardized_response=standardized,
                            raw_provider_response=last_event,
                            is_retryable=False,
                            context=context,
                        )

                    try:
                        evt = _json.loads(payload)
                    except Exception:
                        continue

                    last_event = evt
                    resp_id = evt.get("id", resp_id)
                    created = evt.get("created", created)
                    model = evt.get("model", model)
                    if isinstance(evt, dict) and "error" in evt:
                        err = evt.get("error")
                        if isinstance(err, dict):
                            msg = err.get("message") or str(err)
                        else:
                            msg = str(err)
                        u3_response.close()
                        return LLMResponse(
                            success=False,
                            error_info={"type": "api_error", "message": msg},
                            raw_provider_response=evt,
                            is_retryable=False,
                            context=context,
                        )
                    try:
                        choices = evt.get("choices") or []
                        if choices:
                            c0 = choices[0]
                            finish_reason = c0.get("finish_reason", finish_reason)
                            delta = c0.get("delta") or {}
                            piece = delta.get("content")
                            if piece:
                                aggregated += piece
                            if not piece and isinstance(c0.get("message"), dict):
                                msg_piece = c0["message"].get("content")
                                if msg_piece:
                                    aggregated += msg_piece
                    except Exception:
                        pass

                    if self._has_content_filter_error(evt):
                        err = self._extract_content_filter_error(evt)
                        u3_response.close()
                        return LLMResponse(
                            success=False,
                            error_info=err,
                            raw_provider_response=evt,
                            is_retryable=False,
                            context=context,
                        )

            u3_response.close()
            standardized = {
                "id": resp_id,
                "created": created,
                "model": model,
                "provider": "openrouter",
                "content": aggregated,
                "finish_reason": finish_reason or "stop",
                "usage": (last_event or {}).get("usage", {}),
            }
            if (aggregated or "").strip() == "":
                return LLMResponse(
                    success=False,
                    error_info={
                        "type": "content_filter",
                        "message": "Response contained no content.",
                    },
                    raw_provider_response=last_event,
                    is_retryable=False,
                    context=context,
                )
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
            is_timeout = u3e and isinstance(
                e,
                (
                    getattr(u3e, "TimeoutError", tuple()),
                    getattr(u3e, "ReadTimeoutError", tuple()),
                    getattr(u3e, "ConnectTimeoutError", tuple()),
                ),
            )
            is_ssl = u3e and isinstance(e, getattr(u3e, "SSLError", tuple()))
            is_location = u3e and isinstance(
                e, getattr(u3e, "LocationParseError", tuple())
            )
            if is_timeout:
                err_type = "timeout"
                retryable = True
            elif is_ssl or is_location:
                err_type = "network_error"
                retryable = False
            else:
                err_type = "network_error"
                retryable = True
            return LLMResponse(
                success=False,
                error_info={"type": err_type, "message": str(e), "exception": str(e)},
                is_retryable=retryable,
                context=context,
            )

    def _handle_error_response(self, response, context):
        """Process error responses from the API (urllib3 or requests-like)"""
        status_code = getattr(response, "status", None)
        if status_code is None:
            status_code = getattr(response, "status_code", None)
        is_retryable = status_code in [408, 425, 429, 500, 502, 503, 504]

        error_text = None
        error_json = None
        try:
            if hasattr(response, "data"):
                error_text = (
                    response.data.decode("utf-8", errors="ignore")
                    if isinstance(response.data, (bytes, bytearray))
                    else str(response.data)
                )
            elif hasattr(response, "text"):
                error_text = response.text
        except Exception:
            error_text = None

        if error_text:
            try:
                error_json = json.loads(error_text)
            except Exception:
                error_json = None

        error_message = self._extract_error_message(error_json, error_text or "")

        error_info = {
            "type": "api_error",
            "status_code": status_code,
            "message": error_message,
            "raw_response": error_text or "",
        }

        return LLMResponse(
            success=False,
            error_info=error_info,
            raw_provider_response=error_json,
            is_retryable=is_retryable,
            context=context,
        )

    def _extract_error_message(self, error_json, response_text):
        """Extract error message from response"""
        if error_json and "error" in error_json:
            error_obj = error_json["error"]
            if isinstance(error_obj, dict) and "message" in error_obj:
                return error_obj["message"]
            return str(error_obj)
        return f"Error: {response_text[:200]}"

    def _has_content_filter_error(self, response):
        """Check if the response contains a content filter error"""
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if choice.get("finish_reason") == "content_filter" or "error" in choice:
                return True
        return False

    def _extract_content_filter_error(self, response):
        """Extract content filter error from response"""
        choice = response["choices"][0]
        if "error" in choice:
            error_obj = choice["error"]
            message = error_obj.get("message", "Content filtered")
        else:
            message = "Response stopped due to content filter"

        return {"type": "content_filter", "message": message}

    def _standardize_anthropic_messages_response(self, provider_response):
        """Convert an OpenRouter Anthropic Messages response to standard format."""
        content_blocks = provider_response.get("content") or []
        text_parts = []
        reasoning_parts = []
        reasoning_details = []

        if isinstance(content_blocks, list):
            for index, block in enumerate(content_blocks):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif block_type == "thinking":
                    thinking = block.get("thinking") or block.get("text") or ""
                    if isinstance(thinking, str) and thinking.strip():
                        reasoning_parts.append(thinking)
                    detail = {
                        "type": "reasoning.text",
                        "format": "anthropic-claude-v1",
                        "index": index,
                    }
                    if isinstance(thinking, str):
                        detail["text"] = thinking
                    if block.get("signature") is not None:
                        detail["signature"] = block.get("signature")
                    reasoning_details.append(detail)
                elif block_type == "redacted_thinking":
                    data = block.get("data") or block.get("redacted_thinking") or ""
                    detail = {
                        "type": "reasoning.encrypted",
                        "format": "anthropic-claude-v1",
                        "index": index,
                    }
                    if data:
                        detail["data"] = data
                    reasoning_details.append(detail)

        usage = dict(provider_response.get("usage") or {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, (int, float)):
            usage.setdefault("prompt_tokens", input_tokens)
        if isinstance(output_tokens, (int, float)):
            usage.setdefault("completion_tokens", output_tokens)
        if isinstance(input_tokens, (int, float)) and isinstance(
            output_tokens, (int, float)
        ):
            usage.setdefault("total_tokens", input_tokens + output_tokens)

        standardized = {
            "id": provider_response.get("id"),
            "created": provider_response.get("created"),
            "model": provider_response.get("model"),
            "provider": "openrouter",
            "role": provider_response.get("role"),
            "content": "\n".join(text_parts) if text_parts else None,
            "finish_reason": provider_response.get("stop_reason"),
            "stop_reason": provider_response.get("stop_reason"),
            "stop_sequence": provider_response.get("stop_sequence"),
            "usage": usage,
            "content_blocks": content_blocks,
            "request_format": "anthropic_messages",
        }
        if reasoning_parts:
            standardized["reasoning"] = "\n".join(reasoning_parts)
        if reasoning_details:
            standardized["reasoning_details"] = reasoning_details

        return standardized

    def _standardize_response(self, provider_response):
        """Convert OpenRouter response to standardized format"""
        standardized = {
            "id": provider_response.get("id"),
            "created": provider_response.get("created"),
            "model": provider_response.get("model"),
            "provider": "openrouter",
            "content": None,
            "usage": provider_response.get("usage", {}),
        }

        if "_provider_used" in provider_response:
            standardized["sub_provider"] = provider_response["_provider_used"]

        if "choices" in provider_response and len(provider_response["choices"]) > 0:
            choice = provider_response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                standardized["content"] = choice["message"].get("content")
            standardized["finish_reason"] = choice.get("finish_reason")

        return standardized

    def get_available_providers(self, model_id):
        """
        Get list of providers available for the given model ID
        """
        try:
            import urllib3
            from urllib3.util import Timeout as _Timeout

            endpoints_url = f"{OPENROUTER_API_BASE}/models/{model_id}/endpoints"
            headers = {"Authorization": f"Bearer {self.get_api_key()}"}
            http = urllib3.PoolManager()
            resp = http.request(
                "GET",
                endpoints_url,
                headers=headers,
                timeout=_Timeout(total=30),
                preload_content=True,
            )
            if resp.status != 200:
                print(
                    f"ERROR: Failed to fetch OpenRouter providers for {model_id}: HTTP {resp.status}"
                )
                return None
            try:
                data = json.loads(resp.data.decode("utf-8", errors="ignore"))
            except Exception as e:
                print(
                    f"ERROR: Failed to parse OpenRouter providers JSON for {model_id}: {e}"
                )
                return None

            if "data" in data and isinstance(data["data"], list):
                providers = [
                    ep.get("provider_name")
                    for ep in data["data"]
                    if ep.get("provider_name")
                ]
                unique_providers = sorted(list(set(p for p in providers if p)))
                return unique_providers

            elif (
                "data" in data
                and "endpoints" in data["data"]
                and isinstance(data["data"]["endpoints"], list)
            ):
                providers = [
                    ep.get("provider_name")
                    for ep in data["data"]["endpoints"]
                    if ep.get("provider_name")
                ]
                unique_providers = sorted(list(set(p for p in providers if p)))
                return unique_providers

            else:
                print(
                    f"ERROR: Unexpected structure in OpenRouter model data for {model_id}"
                )
                return None

        except Exception as e:
            print(
                f"ERROR: Failed to fetch OpenRouter providers for {model_id}: {str(e)}"
            )
            return None

    def is_model_available(self, model_id):
        """
        Check if a model is currently available on OpenRouter
        """
        providers = self.get_available_providers(model_id)
        return providers is not None and len(providers) > 0
