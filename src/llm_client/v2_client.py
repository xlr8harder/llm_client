"""Ergonomic sync and async execution API for llm_client V2."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import os
import threading
import time
from typing import Any
import warnings

import httpx

from ._version import __version__
from .v2_builder import ConversationBuilder
from .v2_models import (
    Conversation,
    ErrorInfo,
    Message,
    ModelResponse,
    ProviderState,
    RequestAttempt,
    WireRecord,
    now_iso,
)


PROTOCOLS = {"chat_completions", "messages", "responses"}
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
KNOWN_TOP_LEVEL = {
    "chat_completions": {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
        "openrouter_metadata",
        "provider",
        "service_tier",
        "error",
    },
    "messages": {
        "id",
        "type",
        "role",
        "model",
        "content",
        "stop_reason",
        "stop_sequence",
        "usage",
        "openrouter_metadata",
        "provider",
        "service_tier",
        "container",
        "context_management",
        "stop_details",
        "error",
    },
    "responses": {
        "background",
        "completed_at",
        "id",
        "object",
        "created_at",
        "status",
        "model",
        "output",
        "output_text",
        "usage",
        "error",
        "incomplete_details",
        "openrouter_metadata",
        "service_tier",
        "provider",
        "frequency_penalty",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
        "presence_penalty",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "store",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
    },
}
_UNKNOWN_FIELDS: set[tuple[str, str, str]] = set()
_UNKNOWN_FIELDS_LOCK = threading.Lock()


class UnknownProviderFieldWarning(UserWarning):
    pass


def reset_unknown_field_warnings() -> None:
    with _UNKNOWN_FIELDS_LOCK:
        _UNKNOWN_FIELDS.clear()


class Model:
    def __init__(
        self,
        client: Client,
        model_ref: str,
        *,
        protocol: str | None = None,
        options: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ):
        self.client = client
        self.model_ref = model_ref
        self.provider, self.model_id, parsed_endpoint = _parse_model_ref(model_ref)
        self.protocol = _normalize_protocol(
            protocol or _default_protocol(self.provider)
        )
        self.options = deepcopy(options or {})
        self.endpoint = (
            endpoint or parsed_endpoint or client._endpoint_for(self.provider)
        )

    @property
    def serialized_route(self) -> dict[str, Any]:
        model = (
            f"local/unset/{self.model_id}"
            if self.provider == "local"
            else self.model_ref
        )
        return {
            "model": model,
            "protocol": self.protocol,
            "options": deepcopy(self.options),
        }

    def generate(
        self, content: Any = None, *, messages=None, system=None, **options
    ) -> ModelResponse:
        if content is not None and messages is not None:
            raise ValueError("Pass either content or messages, not both")
        conversation = self.conversation(system=system, messages=messages)
        if content is None:
            return self._send_conversation(
                conversation, None, options, append_input=False
            )
        return conversation.send(content, **options)

    def conversation(
        self, *, system: str | None = None, messages=None, metadata=None
    ) -> Conversation:
        initial = list(messages or [])
        if system is not None:
            initial.insert(0, {"role": "system", "content": system})
        conversation = Conversation.from_messages(
            initial,
            model=self.serialized_route["model"],
            protocol=self.protocol,
            metadata=metadata,
        )
        conversation.default_route = self.serialized_route
        conversation._binding = self
        return conversation

    def _send_conversation(
        self,
        conversation: Conversation,
        content: Any,
        options: dict[str, Any],
        *,
        append_input: bool = True,
    ) -> ModelResponse:
        builder = ConversationBuilder.from_conversation(conversation)
        if append_input:
            input_message = builder.add_message("user", content)
            input_ids = [input_message.id]
        else:
            input_ids = [m.id for m in builder._conversation.pending_messages]
        merged = _merge_options(self.options, options)
        request_spec = self._request_spec(merged)
        builder.begin_operation(request_spec, input_ids)
        result = self.client._execute(
            self, builder._conversation.messages, merged, builder
        )
        operation = builder.complete_operation(result)
        updated = builder.build()
        _replace_conversation(conversation, updated)
        return replace(result, operation=operation)

    def _request_spec(self, options: dict[str, Any]) -> dict[str, Any]:
        requested_model = self.serialized_route["model"]
        return {
            "requested": {
                "model": requested_model,
                "protocol": self.protocol,
                "options": deepcopy(options),
            },
            "resolved": {
                "provider": self.provider,
                "provider_model_id": self.model_id,
                "protocol": self.protocol,
                "options": deepcopy(options),
                "endpoint": "local://unset"
                if self.provider == "local"
                else self.endpoint,
            },
            "resolution": {
                "llm_client_version": __version__,
                "protocol_source": "explicit_or_provider_default",
            },
        }


class AsyncModel(Model):
    client: AsyncClient

    async def generate(
        self, content: Any = None, *, messages=None, system=None, **options
    ) -> ModelResponse:
        if content is not None and messages is not None:
            raise ValueError("Pass either content or messages, not both")
        conversation = self.conversation(system=system, messages=messages)
        return await self.send(
            conversation, content, append_input=content is not None, **options
        )

    def conversation(
        self, *, system: str | None = None, messages=None, metadata=None
    ) -> AsyncConversation:
        initial = list(messages or [])
        if system is not None:
            initial.insert(0, {"role": "system", "content": system})
        conversation = AsyncConversation.from_messages(
            initial,
            model=self.serialized_route["model"],
            protocol=self.protocol,
            metadata=metadata,
        )
        conversation.default_route = self.serialized_route
        conversation._binding = self
        return conversation

    async def send(
        self, conversation: Conversation, content: Any, *, append_input=True, **options
    ) -> ModelResponse:
        if not conversation._busy.acquire(blocking=False):
            from .v2_models import ConversationBusyError

            raise ConversationBusyError(
                "This conversation already has an active writer"
            )
        try:
            builder = ConversationBuilder.from_conversation(conversation)
            if append_input:
                input_message = builder.add_message("user", content)
                input_ids = [input_message.id]
            else:
                input_ids = [m.id for m in builder._conversation.pending_messages]
            merged = _merge_options(self.options, options)
            builder.begin_operation(self._request_spec(merged), input_ids)
            try:
                result = await self.client._execute_async(
                    self, builder._conversation.messages, merged, builder
                )
                operation = builder.complete_operation(result)
            except asyncio.CancelledError:
                error = ErrorInfo(
                    category="interrupted",
                    message="Async request was cancelled",
                    retryable=True,
                )
                operation = builder.interrupt_operation(error)
                updated = builder.build()
                _replace_conversation(conversation, updated)
                raise
            updated = builder.build()
            _replace_conversation(conversation, updated)
            return replace(result, operation=operation)
        finally:
            conversation._busy.release()


class AsyncConversation(Conversation):
    async def send(self, content: Any, **options: Any) -> ModelResponse:
        if self._binding is None:
            from .v2_models import UnboundConversationError

            raise UnboundConversationError(
                "Conversation has no runtime binding. Call conversation.bind(client) before sending."
            )
        if not isinstance(self._binding, AsyncModel):
            raise TypeError("AsyncConversation must be bound to an AsyncClient")
        return await self._binding.send(self, content, **options)

    async def send_pending(self, **options: Any) -> ModelResponse:
        if self._binding is None:
            from .v2_models import UnboundConversationError

            raise UnboundConversationError(
                "Conversation has no runtime binding. Call conversation.bind(client) before sending."
            )
        if not self.pending_messages:
            raise ValueError(
                "Conversation has no pending user or tool messages to send"
            )
        if not isinstance(self._binding, AsyncModel):
            raise TypeError("AsyncConversation must be bound to an AsyncClient")
        return await self._binding.send(self, None, append_input=False, **options)


class Client:
    def __init__(
        self,
        *,
        api_keys: dict[str, str] | None = None,
        local_endpoint: str | None = None,
        timeout: float = 60,
        max_retries: int = 0,
        transport: httpx.BaseTransport | None = None,
        auth: dict[str, Any] | None = None,
    ):
        self.api_keys = dict(api_keys or {})
        self.local_endpoint = local_endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.auth = dict(auth or {})
        self._http = httpx.Client(timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def model(
        self, model_ref: str, *, protocol=None, options=None, endpoint=None
    ) -> Model:
        options = _provider_default_options(model_ref, options)
        return Model(
            self, model_ref, protocol=protocol, options=options, endpoint=endpoint
        )

    def generate(
        self,
        content=None,
        *,
        model: str,
        protocol=None,
        messages=None,
        system=None,
        **options,
    ):
        return self.model(model, protocol=protocol).generate(
            content, messages=messages, system=system, **options
        )

    def load_conversation(
        self, data: str | dict[str, Any], *, local_endpoint=None
    ) -> Conversation:
        conversation = (
            Conversation.from_json(data)
            if isinstance(data, str)
            else Conversation.from_dict(data)
        )
        return conversation.bind(self, local_endpoint=local_endpoint)

    def _bind_conversation(
        self, conversation: Conversation, *, local_endpoint=None
    ) -> Model:
        route = conversation.default_route
        model_ref = route.get("model")
        if not model_ref:
            raise ValueError("Conversation does not define a default model route")
        endpoint = local_endpoint
        if model_ref.startswith("local/unset/"):
            endpoint = endpoint or self.local_endpoint
            if not endpoint:
                raise ValueError(
                    f"Conversation uses {model_ref} and has no bound local endpoint. "
                    "Pass local_endpoint=... or configure Client(local_endpoint=...)."
                )
        return self.model(
            model_ref,
            protocol=route.get("protocol"),
            options=route.get("options"),
            endpoint=endpoint,
        )

    def _endpoint_for(self, provider: str) -> str | None:
        if provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        if provider == "openai":
            return "https://api.openai.com/v1"
        if provider == "codex":
            return "https://chatgpt.com/backend-api/codex"
        if provider == "local":
            return self.local_endpoint
        return None

    def _api_key(self, provider: str) -> str | None:
        env = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "local": "LOCAL_LLM_API_KEY",
        }.get(provider)
        return self.api_keys.get(provider) or (os.getenv(env) if env else None)

    def _auth_headers(self, provider: str) -> dict[str, str]:
        if provider in self.auth:
            return self.auth[provider].get_headers()
        key = self._api_key(provider)
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _execute(
        self, model: Model, messages: list[Message], options, builder
    ) -> ModelResponse:
        return _execute_loop(
            model,
            messages,
            options,
            builder,
            self._http.request,
            self.max_retries,
            self._auth_headers(model.provider),
        )


class AsyncClient(Client):
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None, **kwargs):
        api_keys = kwargs.pop("api_keys", None)
        local_endpoint = kwargs.pop("local_endpoint", None)
        auth = kwargs.pop("auth", None)
        timeout = kwargs.pop("timeout", 60)
        max_retries = kwargs.pop("max_retries", 0)
        if kwargs:
            raise TypeError(f"Unexpected AsyncClient options: {sorted(kwargs)}")
        self.api_keys = dict(api_keys or {})
        self.local_endpoint = local_endpoint
        self.auth = dict(auth or {})
        self.timeout = timeout
        self.max_retries = max_retries
        self._http = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    def model(
        self, model_ref: str, *, protocol=None, options=None, endpoint=None
    ) -> AsyncModel:
        options = _provider_default_options(model_ref, options)
        return AsyncModel(
            self, model_ref, protocol=protocol, options=options, endpoint=endpoint
        )

    async def generate(
        self,
        content=None,
        *,
        model: str,
        protocol=None,
        messages=None,
        system=None,
        **options,
    ):
        return await self.model(model, protocol=protocol).generate(
            content, messages=messages, system=system, **options
        )

    def load_conversation(
        self, data: str | dict[str, Any], *, local_endpoint=None
    ) -> AsyncConversation:
        payload = json.loads(data) if isinstance(data, str) else data
        conversation = AsyncConversation.from_dict(payload)
        conversation.bind(self, local_endpoint=local_endpoint)
        return conversation

    async def _execute_async(
        self, model: Model, messages, options, builder
    ) -> ModelResponse:
        # Token refresh uses process-level file locks and a synchronous endpoint.
        auth_headers = await asyncio.to_thread(self._auth_headers, model.provider)
        return await _execute_loop_async(
            model,
            messages,
            options,
            builder,
            self._http.request,
            self.max_retries,
            auth_headers,
        )


def _execute_loop(
    model, messages, options, builder, request, max_retries, auth_headers
):
    for number in range(1, max_retries + 2):
        attempt, result = _one_request(
            model, messages, options, request, number, auth_headers
        )
        builder.add_attempt(attempt)
        if (
            result.ok
            or not (result.error and result.error.retryable)
            or number > max_retries
        ):
            _attach_outputs(builder, result)
            _attach_notices(builder, result)
            return result
    raise AssertionError("unreachable")


async def _execute_loop_async(
    model, messages, options, builder, request, max_retries, auth_headers
):
    for number in range(1, max_retries + 2):
        attempt, result = await _one_request_async(
            model, messages, options, request, number, auth_headers
        )
        builder.add_attempt(attempt)
        if (
            result.ok
            or not (result.error and result.error.retryable)
            or number > max_retries
        ):
            _attach_outputs(builder, result)
            _attach_notices(builder, result)
            return result
    raise AssertionError("unreachable")


def _one_request(model, messages, options, request, number, auth_headers):
    url, headers, body = _build_request(model, messages, options, auth_headers)
    started = time.monotonic()
    try:
        response = request("POST", url, headers=headers, json=body)
        return _consume_response(
            model, url, headers, body, response, number, started, auth_headers
        )
    except httpx.HTTPError as exc:
        return _transport_failure(
            model, url, headers, body, exc, number, started, auth_headers
        )


async def _one_request_async(model, messages, options, request, number, auth_headers):
    url, headers, body = _build_request(model, messages, options, auth_headers)
    started = time.monotonic()
    try:
        response = await request("POST", url, headers=headers, json=body)
        return _consume_response(
            model, url, headers, body, response, number, started, auth_headers
        )
    except httpx.HTTPError as exc:
        return _transport_failure(
            model, url, headers, body, exc, number, started, auth_headers
        )


def _build_request(model, messages, options, auth_headers):
    endpoint = model.endpoint
    if not endpoint:
        raise ValueError(f"No endpoint configured for provider {model.provider}")
    path = {
        "chat_completions": "/chat/completions",
        "messages": "/messages",
        "responses": "/responses",
    }[model.protocol]
    url = endpoint.rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    headers.update(auth_headers)
    extra_headers = deepcopy(options.get("extra_headers", {}))
    headers.update(extra_headers)
    if model.provider == "openrouter" and options.get("openrouter_metadata", False):
        headers.setdefault("X-OpenRouter-Metadata", "enabled")
    body = _protocol_body(model.protocol, model.model_id, messages, options)
    return url, headers, body


def _protocol_body(protocol, model_id, messages, options):
    reserved = {
        "extra_headers",
        "extra_body",
        "provider_options",
        "openrouter_metadata",
        "timeout",
        "max_retries",
    }
    common = {k: deepcopy(v) for k, v in options.items() if k not in reserved}
    if protocol == "chat_completions":
        body = {
            "model": model_id,
            "messages": [m.to_standard() for m in messages],
            **common,
        }
    elif protocol == "messages":
        system = [m.text for m in messages if m.role in {"system", "developer"}]
        body = {
            "model": model_id,
            "messages": [
                m.to_standard()
                for m in messages
                if m.role not in {"system", "developer"}
            ],
            **common,
        }
        if system:
            body["system"] = "\n\n".join(system)
        body.setdefault("max_tokens", 4096)
    elif protocol == "responses":
        instructions = [m.text for m in messages if m.role in {"system", "developer"}]
        items = []
        for message in messages:
            if message.role in {"system", "developer"}:
                continue
            if (
                message.provider_state
                and message.provider_state.protocol == "responses"
            ):
                items.extend(deepcopy(message.provider_state.replay))
            elif message.role in {"user", "assistant"}:
                items.append({"role": message.role, "content": message.text})
        body = {"model": model_id, "input": items, **common}
        if instructions:
            body["instructions"] = "\n\n".join(instructions)
        if "max_tokens" in body:
            body["max_output_tokens"] = body.pop("max_tokens")
    else:
        body = {"model": model_id, **common}
    provider_options = options.get("provider_options")
    if provider_options:
        body["provider"] = deepcopy(provider_options)
    extra_body = options.get("extra_body", {})
    conflicts = set(body) & set(extra_body)
    if conflicts:
        raise ValueError(
            f"extra_body conflicts with resolved request fields: {sorted(conflicts)}"
        )
    body.update(deepcopy(extra_body))
    return body


def _consume_response(
    model, url, request_headers, request_body, response, number, started, auth_headers
):
    elapsed = time.monotonic() - started
    stream_events = ()
    if "text/event-stream" in response.headers.get("content-type", ""):
        stream_events = tuple(_parse_sse(response.text))
        raw = _aggregate_stream(model.protocol, stream_events)
    else:
        try:
            raw = response.json()
        except ValueError:
            raw = {"raw_text": response.text}
    safe_url = _safe_url(model, url)
    safe_headers = _safe_headers(model, request_headers)
    wire = WireRecord(
        request={
            "method": "POST",
            "url": safe_url,
            "headers": safe_headers,
            "body": deepcopy(request_body),
        },
        response={
            "status_code": response.status_code,
            "headers": _safe_response_headers(model, response.headers),
            "body": deepcopy(raw),
        },
        stream_events=stream_events,
        timing={"elapsed_seconds": elapsed},
    )
    result = _normalize(model, raw, response.status_code)
    _warn_unknown(model, raw, result)
    attempt = RequestAttempt(
        provider=model.provider,
        model=model.model_id,
        protocol=model.protocol,
        number=number,
        status="succeeded" if result.ok else "failed",
        retryable=result.error.retryable if result.error else False,
        error=result.error,
        wire=wire,
        completed_at=now_iso(),
    )
    return attempt, result


def _transport_failure(model, url, headers, body, exc, number, started, auth_headers):
    message = type(exc).__name__
    error = ErrorInfo(
        category="transport",
        message=message,
        retryable=True,
        raw={"exception_type": message},
    )
    wire = WireRecord(
        request={
            "method": "POST",
            "url": _safe_url(model, url),
            "headers": _safe_headers(model, headers),
            "body": deepcopy(body),
        },
        timing={"elapsed_seconds": time.monotonic() - started},
        transport_error=error.raw,
    )
    attempt = RequestAttempt(
        provider=model.provider,
        model=model.model_id,
        protocol=model.protocol,
        number=number,
        status="failed",
        retryable=True,
        error=error,
        wire=wire,
        completed_at=now_iso(),
    )
    return attempt, ModelResponse(ok=False, error=error)


def _normalize(model, raw, status):
    if status < 200 or status >= 300 or (isinstance(raw, dict) and raw.get("error")):
        error_raw = raw.get("error", raw) if isinstance(raw, dict) else raw
        message = (
            error_raw.get("message", f"Provider returned HTTP {status}")
            if isinstance(error_raw, dict)
            else str(error_raw)
        )
        code = error_raw.get("code") if isinstance(error_raw, dict) else None
        category = _error_category(status, code, message)
        error = ErrorInfo(
            category=category,
            message=message,
            retryable=status in RETRYABLE_STATUS,
            status_code=status,
            provider_code=code,
            provider_type=error_raw.get("type")
            if isinstance(error_raw, dict)
            else None,
            raw=deepcopy(error_raw),
        )
        return ModelResponse(ok=False, error=error, raw=deepcopy(raw))
    if model.protocol == "chat_completions":
        choice = (raw.get("choices") or [{}])[0]
        message_raw = choice.get("message") or {}
        content = message_raw.get("content") or ""
        native = choice.get("native_finish_reason", choice.get("finish_reason"))
        finish = _normalize_finish(choice.get("finish_reason"))
        message = Message(
            role="assistant",
            content=({"type": "text", "text": content},),
            metadata={"provider": {model.provider: {"choice": deepcopy(choice)}}},
        )
    elif model.protocol == "messages":
        blocks = raw.get("content") or []
        content = "".join(
            block.get("text", "") for block in blocks if block.get("type") == "text"
        )
        native = raw.get("stop_reason")
        finish = _normalize_finish(native)
        replay = tuple(
            block
            for block in blocks
            if block.get("type") == "thinking" and block.get("signature")
        )
        state = (
            ProviderState(model.provider, "messages", replay=replay) if replay else None
        )
        message = Message(
            role="assistant", content=tuple(deepcopy(blocks)), provider_state=state
        )
    else:
        output = raw.get("output") or []
        content = raw.get("output_text") or _responses_text(output)
        status_value = raw.get("status")
        native = status_value
        finish = (
            "stop"
            if status_value in {None, "completed"}
            else _normalize_finish(status_value)
        )
        replay = tuple(deepcopy(output))
        state = ProviderState(
            model.provider,
            "responses",
            replay=replay,
            metadata={"response_id": raw.get("id")},
        )
        message = Message(
            role="assistant",
            content=({"type": "text", "text": content},),
            provider_state=state,
        )
    provider_metadata = {}
    if model.provider == "openrouter" and isinstance(raw, dict):
        if raw.get("provider") is not None:
            provider_metadata["selected_backend"] = deepcopy(raw["provider"])
        if raw.get("service_tier") is not None:
            provider_metadata["service_tier"] = deepcopy(raw["service_tier"])
    metadata = {"provider": {model.provider: provider_metadata}}
    if isinstance(raw, dict) and "openrouter_metadata" in raw:
        provider_metadata.update(deepcopy(raw["openrouter_metadata"]))
    error = _finish_error(finish, native, raw)
    return ModelResponse(
        ok=error is None,
        content=content,
        finish_reason=finish,
        native_finish_reason=native,
        usage=deepcopy(raw.get("usage", {})),
        messages=(message,) if content or message.provider_state else (),
        error=error,
        raw=deepcopy(raw),
        metadata=metadata,
    )


def _attach_outputs(builder, result):
    for message in result.messages:
        builder.add_output_message(message)
    if result.finish_reason is not None:
        builder.add_normalization_evidence(
            "finish_reason",
            {
                "source": "wire.response.body",
                "value": result.native_finish_reason,
                "normalized": result.finish_reason,
            },
        )


def _attach_notices(builder, result):
    for notice in result.metadata.get("llm_client", {}).get(
        "normalization_notices", []
    ):
        builder.add_normalization_notice(notice)


def _warn_unknown(model, raw, result):
    if not isinstance(raw, dict):
        return
    known = KNOWN_TOP_LEVEL.get(model.protocol, set())
    notices = result.metadata.setdefault("llm_client", {}).setdefault(
        "normalization_notices", []
    )
    for field in sorted(set(raw) - known):
        key = (model.provider, model.protocol, field)
        notices.append(
            {
                "kind": "unknown_provider_field",
                "provider": model.provider,
                "protocol": model.protocol,
                "direction": "response",
                "path": field,
                "observed_type": type(raw[field]).__name__,
                "action": "preserved",
            }
        )
        with _UNKNOWN_FIELDS_LOCK:
            should_warn = key not in _UNKNOWN_FIELDS
            _UNKNOWN_FIELDS.add(key)
        if should_warn:
            warnings.warn(
                f"Unknown {model.provider} {model.protocol} response field {field!r}; preserved in raw data",
                UnknownProviderFieldWarning,
                stacklevel=4,
            )


def _responses_text(output):
    parts = []
    for item in output:
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") in {"output_text", "text"}:
                parts.append(block.get("text", ""))
    return "".join(parts)


def _parse_sse(text):
    event_name = None
    data_lines = []
    for line in text.splitlines() + [""]:
        if not line:
            if data_lines:
                payload_text = "\n".join(data_lines)
                if payload_text != "[DONE]":
                    try:
                        payload = json.loads(payload_text)
                    except ValueError:
                        payload = {"raw_data": payload_text}
                    if event_name and isinstance(payload, dict):
                        payload.setdefault("_event", event_name)
                    yield payload
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


def _aggregate_stream(protocol, events):
    if protocol == "responses":
        text = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.output_text.delta"
        )
        reasoning = "".join(
            event.get("delta", "")
            for event in events
            if event.get("type") == "response.reasoning_summary_text.delta"
        )
        for event in reversed(events):
            if event.get("type") == "response.completed" and isinstance(
                event.get("response"), dict
            ):
                response = deepcopy(event["response"])
                if text and not response.get("output_text"):
                    response["output_text"] = text
                if reasoning and not any(
                    isinstance(item, dict) and item.get("type") == "reasoning"
                    for item in response.get("output") or []
                ):
                    response.setdefault("output", []).insert(
                        0,
                        {
                            "type": "reasoning",
                            "summary": [
                                {"type": "summary_text", "text": reasoning}
                            ],
                        },
                    )
                return response
        return {
            "object": "response",
            "status": "completed",
            "output_text": text,
            "output": (
                [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": reasoning}],
                    }
                ]
                if reasoning
                else []
            ),
            "usage": {},
        }
    if protocol == "messages":
        text = "".join(
            event.get("delta", {}).get("text", "")
            for event in events
            if event.get("type") == "content_block_delta"
        )
        stop = next(
            (
                event.get("delta", {}).get("stop_reason")
                for event in reversed(events)
                if event.get("type") == "message_delta"
            ),
            "end_turn",
        )
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop,
            "usage": {},
        }
    text_parts = []
    finish = None
    last = {}
    for event in events:
        last = event
        choices = event.get("choices") or []
        if choices:
            choice = choices[0]
            text_parts.append((choice.get("delta") or {}).get("content") or "")
            finish = choice.get("finish_reason") or finish
    return {
        "id": last.get("id"),
        "object": "chat.completion",
        "model": last.get("model"),
        "choices": [
            {
                "message": {"role": "assistant", "content": "".join(text_parts)},
                "finish_reason": finish or "stop",
            }
        ],
        "usage": last.get("usage", {}),
    }


def _error_category(status, code, message):
    text = f"{code or ''} {message}".lower()
    if status == 401 or "auth" in text:
        return "authentication"
    if status == 403:
        return "authorization"
    if status == 429:
        return "rate_limit"
    if "content_filter" in text or "moderation" in text or "safety" in text:
        return "moderation"
    if status in {408, 504}:
        return "timeout"
    if status == 400:
        return "invalid_request"
    return "provider"


def _normalize_finish(value):
    if value is None:
        return None
    value = str(value)
    mapping = {
        "end_turn": "stop",
        "completed": "stop",
        "STOP": "stop",
        "eos": "stop",
        "eos_token": "stop",
        "max_tokens": "length",
        "max_output_tokens": "length",
        "MAX_TOKENS": "length",
        "safety": "content_filter",
        "refusal": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "RECITATION": "content_filter",
        "sensitive": "content_filter",
        "engine_overloaded": "error",
    }
    return mapping.get(
        value,
        value
        if value in {"stop", "length", "tool_calls", "content_filter", "error"}
        else "unknown",
    )


def _finish_error(finish, native, raw):
    if finish is None:
        return ErrorInfo(
            category="invalid_response",
            message="Provider response omitted required finish metadata",
            retryable=None,
            native_finish_reason=native,
            raw=deepcopy(raw),
        )
    if finish == "content_filter":
        return ErrorInfo(
            category="moderation",
            message="Provider ended generation for moderation",
            retryable=False,
            finish_reason=finish,
            native_finish_reason=native,
            raw=deepcopy(raw),
        )
    if finish == "length":
        return ErrorInfo(
            category="truncation",
            message="Provider output reached its generation limit",
            retryable=False,
            finish_reason=finish,
            native_finish_reason=native,
            raw=deepcopy(raw),
        )
    if finish == "unknown":
        return ErrorInfo(
            category="invalid_response",
            message=f"Unknown provider finish reason: {native}",
            retryable=None,
            finish_reason=finish,
            native_finish_reason=native,
            raw=deepcopy(raw),
        )
    if finish == "error":
        return ErrorInfo(
            category="provider",
            message="Provider returned an error finish reason",
            retryable=None,
            finish_reason=finish,
            native_finish_reason=native,
            raw=deepcopy(raw),
        )
    return None


def _parse_model_ref(ref):
    parts = ref.split("/")
    if len(parts) < 2:
        raise ValueError("Model reference must include provider/model")
    provider = parts[0].lower()
    if provider == "local":
        if len(parts) < 3:
            raise ValueError("Local model reference must be local/endpoint/model")
        host = parts[1]
        model_id = "/".join(parts[2:])
        endpoint = (
            None
            if host == "unset"
            else (host if "://" in host else f"http://{host}/v1")
        )
        return provider, model_id, endpoint
    return provider, "/".join(parts[1:]), None


def _default_protocol(provider):
    return {"codex": "responses", "anthropic": "messages"}.get(
        provider, "chat_completions"
    )


def _normalize_protocol(protocol):
    aliases = {
        "chat_completion": "chat_completions",
        "anthropic_messages": "messages",
        "response": "responses",
    }
    normalized = aliases.get(
        str(protocol).replace("-", "_"), str(protocol).replace("-", "_")
    )
    if normalized not in PROTOCOLS:
        raise ValueError(f"Unsupported protocol: {protocol}")
    return normalized


def _merge_options(base, overrides):
    result = deepcopy(base)
    result.update(deepcopy(overrides))
    return result


def _provider_default_options(model_ref, options):
    result = {}
    if model_ref.lower().startswith("codex/"):
        result = {
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }
    result.update(deepcopy(options or {}))
    return result


def _replace_conversation(target, source):
    target.messages = source.messages
    target.operations = source.operations
    target.default_route = source.default_route
    target.metadata = source.metadata


def _safe_headers(model, headers):
    result = {}
    for key, value in headers.items():
        if key.lower() in {
            "authorization",
            "proxy-authorization",
            "cookie",
            "set-cookie",
            "x-api-key",
            "x-goog-api-key",
            "chatgpt-account-id",
        }:
            result[key] = "[REDACTED]"
        elif model.provider == "local" and key.lower() == "host":
            result[key] = "[REDACTED]"
        else:
            result[key] = value
    return result


def _safe_response_headers(model, headers):
    return _safe_headers(model, dict(headers))


def _safe_url(model, url):
    if model.provider == "local":
        suffix = "/" + url.split("/", 3)[3] if url.count("/") >= 3 else ""
        return "local://unset" + suffix
    return url


def generate(content=None, *, model: str, client: Client | None = None, **kwargs):
    owned = client is None
    client = client or Client()
    try:
        return client.generate(content, model=model, **kwargs)
    finally:
        if owned:
            client.close()
