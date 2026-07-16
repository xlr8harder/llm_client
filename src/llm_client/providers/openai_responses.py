"""Shared transport for providers implementing the OpenAI Responses API."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from ..base import LLMProvider, LLMResponse, with_finish_reason_metadata


class OpenAIResponsesStyleProvider(LLMProvider):
    """Translate chat-style messages to a stateless Responses API request."""

    api_base: Optional[str] = None
    api_key_env_var: Optional[str] = None
    provider_name: Optional[str] = None
    default_timeout: int = 60
    default_max_tokens: int = 4096
    force_stream: bool = False

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

    def _get_responses_url(self) -> str:
        return f"{self._get_api_base()}/responses"

    def _build_responses_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_api_key()}",
            "Content-Type": "application/json",
        }

    def make_chat_completion_request(
        self, messages, model_id, context=None, **options
    ) -> LLMResponse:
        """Compatibility entry point for chat-shaped callers such as mq."""
        return self.make_responses_request(
            messages=messages,
            model_id=model_id,
            context=context,
            **options,
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
        if normalized in {
            "chat_completion",
            "chat_completions",
            "response",
            "responses",
        }:
            return self.make_responses_request(
                messages=messages,
                model_id=model_id,
                context=context,
                **options,
            )
        return LLMResponse(
            success=False,
            error_info={
                "type": "invalid_option",
                "message": f"Unsupported request_format: {request_format}",
            },
            request_format=normalized,
            raw_response_format="llm_client.error",
            is_retryable=False,
            context=context,
        )

    def make_responses_request(
        self, messages, model_id, context=None, **options
    ) -> LLMResponse:
        timeout = options.pop("timeout", self.default_timeout)
        invalid = self._validate_responses_options(options, context)
        if invalid is not None:
            return invalid

        try:
            data = self._build_responses_payload(messages, model_id, options)
            headers = self._build_responses_headers()
        except (OSError, ValueError) as error:
            return LLMResponse(
                success=False,
                error_info={"type": "auth_error", "message": str(error)},
                request_format="responses",
                raw_response_format="llm_client.error",
                is_retryable=False,
                context=context,
            )

        use_stream = bool(data.get("stream"))
        try:
            import urllib3
            from urllib3.util import Timeout as U3Timeout

            response = urllib3.PoolManager().request(
                "POST",
                self._get_responses_url(),
                body=json.dumps(data).encode("utf-8"),
                headers=headers,
                preload_content=not use_stream,
                timeout=U3Timeout(total=float(timeout)),
            )
            if response.status != 200:
                return self._handle_responses_http_error(response, context)
            if use_stream:
                return self._consume_responses_stream(response, context)
            return self._consume_responses_json(response, context)
        except Exception as error:
            return self._transport_error(error, context)

    def _validate_responses_options(self, options, context) -> LLMResponse | None:
        if options.get("top_k") is not None:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "invalid_option",
                    "message": "top_k is not supported by the OpenAI Responses API. Remove --top-k for this model.",
                },
                request_format="responses",
                raw_response_format="llm_client.error",
                is_retryable=False,
                context=context,
            )
        transport = options.get("transport")
        if transport not in {None, "stream"}:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "invalid_option",
                    "message": f"Unsupported Responses transport: {transport}",
                },
                request_format="responses",
                raw_response_format="llm_client.error",
                is_retryable=False,
                context=context,
            )
        if options.get("stream") and transport != "stream" and not self.force_stream:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "invalid_option",
                    "message": "Direct stream=True is not supported. Use transport='stream' for aggregated streaming.",
                },
                request_format="responses",
                raw_response_format="llm_client.error",
                is_retryable=False,
                context=context,
            )
        return None

    def _build_responses_payload(self, messages, model_id, options) -> Dict[str, Any]:
        options = dict(options)
        options.pop("top_k", None)
        transport = options.pop("transport", None)
        requested_stream = options.pop("stream", None)
        max_output_tokens = options.pop(
            "max_output_tokens", options.pop("max_tokens", self.default_max_tokens)
        )
        reasoning_effort = options.pop("reasoning_effort", None)
        reasoning_summary = options.pop("reasoning_summary", None)
        text_verbosity = options.pop("text_verbosity", None)

        instructions, input_items = self._translate_messages(messages)
        data: Dict[str, Any] = {
            "model": self._normalize_responses_model_id(str(model_id)),
            "input": input_items,
            "max_output_tokens": max_output_tokens,
        }
        if instructions:
            data["instructions"] = instructions
        if self.force_stream or transport == "stream" or requested_stream:
            data["stream"] = True
        if reasoning_effort is not None or reasoning_summary is not None:
            reasoning: Dict[str, Any] = {}
            if reasoning_effort is not None:
                reasoning["effort"] = reasoning_effort
            if reasoning_summary is not None:
                reasoning["summary"] = reasoning_summary
            data["reasoning"] = reasoning
        if text_verbosity is not None:
            data["text"] = {"verbosity": text_verbosity}
        data.update(options)
        return self._customize_responses_payload(data)

    def _normalize_responses_model_id(self, model_id: str) -> str:
        return model_id

    def _customize_responses_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def _translate_messages(self, messages) -> tuple[str | None, list[dict]]:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list of role/content objects")
        instructions: list[str] = []
        input_items: list[dict] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"messages[{index}] must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "developer", "user", "assistant"}:
                raise ValueError(
                    f"messages[{index}].role must be system, developer, user, or assistant"
                )
            if not isinstance(content, (str, list)):
                raise ValueError(f"messages[{index}].content must be text or a content list")
            if role in {"system", "developer"}:
                if not isinstance(content, str):
                    raise ValueError(
                        f"messages[{index}] {role} content must be text for Responses translation"
                    )
                if content.strip():
                    instructions.append(content)
                continue
            input_items.append({"role": role, "content": content})
        return "\n\n".join(instructions) or None, input_items

    def _consume_responses_json(self, response, context) -> LLMResponse:
        try:
            payload = json.loads(response.data.decode("utf-8", errors="replace"))
        except (AttributeError, json.JSONDecodeError) as error:
            return LLMResponse(
                success=False,
                error_info={"type": "api_error", "message": f"Invalid Responses JSON: {error}"},
                request_format="responses",
                raw_response_format="openai.responses.invalid",
                is_retryable=False,
                context=context,
            )
        return self._response_from_payload(payload, context)

    def _consume_responses_stream(self, response, context) -> LLMResponse:
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        final_payload: dict | None = None
        last_event: dict | None = None
        try:
            for event in self._iter_sse_json(response.stream(decode_content=True)):
                last_event = event
                event_type = event.get("type")
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        text_parts.append(delta)
                elif event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }:
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        reasoning_parts.append(delta)
                elif event_type in {"response.completed", "response.incomplete"}:
                    candidate = event.get("response")
                    if isinstance(candidate, dict):
                        final_payload = candidate
                elif event_type in {"response.failed", "error"} or event.get("error"):
                    return self._response_event_error(event, context)
        except Exception as error:
            return self._transport_error(error, context)
        finally:
            try:
                response.close()
            except Exception:
                pass

        payload = final_payload or last_event or {}
        return self._response_from_payload(
            payload,
            context,
            streamed_text="".join(text_parts),
            streamed_reasoning="".join(reasoning_parts),
        )

    def _iter_sse_json(self, chunks: Iterable[bytes]):
        buffer = ""
        for chunk in chunks:
            if not chunk:
                continue
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                parsed = self._parse_sse_data_line(line)
                if parsed is not None:
                    yield parsed
        if buffer:
            parsed = self._parse_sse_data_line(buffer)
            if parsed is not None:
                yield parsed

    def _parse_sse_data_line(self, line: str) -> dict | None:
        line = line.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            return None
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            return None
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def _response_from_payload(
        self,
        payload,
        context,
        *,
        streamed_text: str = "",
        streamed_reasoning: str = "",
    ) -> LLMResponse:
        if not isinstance(payload, dict):
            payload = {}
        if payload.get("error"):
            return self._response_event_error(payload, context)

        content = self._extract_output_text(payload) or streamed_text
        reasoning = self._extract_reasoning_summary(payload) or streamed_reasoning
        status = payload.get("status") or "completed"
        finish_reason = self._normalize_responses_finish_reason(payload, status)
        standardized: Dict[str, Any] = {
            "id": payload.get("id"),
            "created": payload.get("created_at"),
            "model": payload.get("model"),
            "provider": self._get_provider_name(),
            "content": content,
            "usage": payload.get("usage") or {},
        }
        with_finish_reason_metadata(
            standardized,
            source="status/incomplete_details.reason",
            value=(payload.get("incomplete_details") or {}).get("reason") or status,
            normalized=finish_reason,
        )
        if reasoning:
            standardized["reasoning"] = reasoning

        if not content.strip():
            return LLMResponse(
                success=False,
                error_info={
                    "type": "content_filter" if finish_reason == "content_filter" else "api_error",
                    "message": "Response contained no content.",
                    "finish_reason": finish_reason,
                },
                raw_provider_response=payload,
                request_format="responses",
                raw_response_format="openai.responses",
                is_retryable=False,
                context=context,
            )
        return LLMResponse(
            success=True,
            standardized_response=standardized,
            raw_provider_response=payload,
            request_format="responses",
            raw_response_format="openai.responses",
            is_retryable=False,
            context=context,
        )

    def _extract_output_text(self, payload: dict) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        parts: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)

    def _extract_reasoning_summary(self, payload: dict) -> str:
        parts: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            for block in item.get("summary") or []:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)

    def _normalize_responses_finish_reason(self, payload: dict, status: str) -> str:
        reason = (payload.get("incomplete_details") or {}).get("reason")
        if reason in {"max_output_tokens", "max_tokens"}:
            return "length"
        if reason in {"content_filter", "content_policy_violation"}:
            return "content_filter"
        if status in {"failed", "cancelled"}:
            return "error"
        return "stop"

    def _response_event_error(self, event: dict, context) -> LLMResponse:
        error = event.get("error") or (event.get("response") or {}).get("error") or event
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or str(error)
            status_code = error.get("status_code")
        else:
            message = str(error)
            status_code = None
        return LLMResponse(
            success=False,
            error_info={
                "type": "api_error",
                "message": message,
                **({"status_code": status_code} if status_code is not None else {}),
            },
            raw_provider_response=event,
            request_format="responses",
            raw_response_format="openai.responses.error",
            is_retryable=status_code in {408, 425, 429, 500, 502, 503, 504},
            context=context,
        )

    def _handle_responses_http_error(self, response, context) -> LLMResponse:
        raw = getattr(response, "data", b"") or b""
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        message = error.get("message") if isinstance(error, dict) else text[:500]
        status = getattr(response, "status", None)
        return LLMResponse(
            success=False,
            error_info={
                "type": "api_error",
                "status_code": status,
                "message": message or f"Responses request failed with HTTP {status}",
                "raw_response": text,
            },
            raw_provider_response=payload,
            request_format="responses",
            raw_response_format="openai.responses.error",
            is_retryable=status in {408, 425, 429, 500, 502, 503, 504},
            context=context,
        )

    def _transport_error(self, error: Exception, context) -> LLMResponse:
        message = str(error)
        lowered = message.lower()
        error_type = "timeout" if "timeout" in lowered or "timed out" in lowered else "network_error"
        return LLMResponse(
            success=False,
            error_info={"type": error_type, "message": message, "exception": message},
            request_format="responses",
            raw_response_format="llm_client.error",
            is_retryable=True,
            context=context,
        )
