"""Canonical, serializable records for the llm_client V2 API."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Iterable, Iterator
from uuid import uuid4


SCHEMA_VERSION = 1
REDACTED = "[REDACTED]"
SECRET_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "chatgpt-account-id",
}
SECRET_FIELDS = {
    "access_token",
    "refresh_token",
    "client_secret",
    "api_key",
    "apikey",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _redact(value: Any, secrets: Iterable[str] = (), *, key: str | None = None) -> Any:
    if key and key.lower() in SECRET_HEADERS | SECRET_FIELDS:
        return REDACTED
    if isinstance(value, dict):
        return {str(k): _redact(v, secrets, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secrets) for v in value]
    if isinstance(value, tuple):
        return [_redact(v, secrets) for v in value]
    if isinstance(value, str):
        result = value
        for secret in secrets:
            if secret:
                result = result.replace(secret, REDACTED)
        return result
    return value


@dataclass(frozen=True)
class ProviderState:
    provider: str
    protocol: str
    replay: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "protocol": self.protocol,
            "schema_version": self.schema_version,
            "replay": deepcopy(list(self.replay)),
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderState:
        return cls(
            provider=data["provider"],
            protocol=data["protocol"],
            schema_version=int(data.get("schema_version", 1)),
            replay=tuple(deepcopy(data.get("replay", []))),
            metadata=deepcopy(data.get("metadata", {})),
        )


def normalize_content(content: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(content, str):
        return ({"type": "text", "text": content},)
    if content is None:
        return ()
    if isinstance(content, dict):
        return (deepcopy(content),)
    if isinstance(content, (list, tuple)):
        result = []
        for part in content:
            if isinstance(part, str):
                result.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                result.append(deepcopy(part))
            else:
                raise TypeError(
                    f"Message content parts must be strings or mappings, got {type(part).__name__}"
                )
        return tuple(result)
    raise TypeError(
        f"Message content must be a string, mapping, or sequence, got {type(content).__name__}"
    )


@dataclass(frozen=True)
class Message:
    role: str
    content: tuple[dict[str, Any], ...]
    id: str = field(default_factory=lambda: new_id("msg"))
    status: str = "complete"
    provider_state: ProviderState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(
            str(part.get("text", ""))
            for part in self.content
            if part.get("type") in {"text", "output_text", "input_text"}
        )

    def to_standard(self) -> dict[str, Any]:
        if len(self.content) == 1 and self.content[0].get("type") == "text":
            content: Any = self.content[0].get("text", "")
        else:
            content = deepcopy(list(self.content))
        return {"role": self.role, "content": content}

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "role": self.role,
            "content": deepcopy(list(self.content)),
            "status": self.status,
            "metadata": deepcopy(self.metadata),
        }
        if self.provider_state is not None:
            data["provider_state"] = self.provider_state.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        known = {"id", "role", "content", "status", "provider_state", "metadata"}
        metadata = deepcopy(data.get("metadata", {}))
        extras = {k: deepcopy(v) for k, v in data.items() if k not in known}
        if extras:
            metadata.setdefault("migration", {})["unmapped_message_fields"] = extras
        state = data.get("provider_state")
        return cls(
            id=data.get("id", new_id("msg")),
            role=data["role"],
            content=normalize_content(data.get("content")),
            status=data.get("status", "complete"),
            provider_state=ProviderState.from_dict(state) if state else None,
            metadata=metadata,
        )

    @classmethod
    def from_standard(cls, data: dict[str, Any]) -> Message:
        return cls.from_dict(data)


@dataclass(frozen=True)
class ErrorInfo:
    category: str
    message: str
    retryable: bool | None = None
    status_code: int | None = None
    provider_code: str | int | None = None
    provider_type: str | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {k: deepcopy(v) for k, v in vars(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorInfo:
        return cls(
            **{k: deepcopy(v) for k, v in data.items() if k in cls.__dataclass_fields__}
        )


@dataclass(frozen=True)
class WireRecord:
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    stream_events: tuple[dict[str, Any], ...] = ()
    timing: dict[str, Any] = field(default_factory=dict)
    transport_error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": deepcopy(self.request),
            "response": deepcopy(self.response),
            "stream_events": deepcopy(list(self.stream_events)),
            "timing": deepcopy(self.timing),
            "transport_error": deepcopy(self.transport_error),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WireRecord:
        return cls(
            request=deepcopy(data.get("request")),
            response=deepcopy(data.get("response")),
            stream_events=tuple(deepcopy(data.get("stream_events", []))),
            timing=deepcopy(data.get("timing", {})),
            transport_error=deepcopy(data.get("transport_error")),
        )


@dataclass(frozen=True)
class RequestAttempt:
    provider: str
    model: str
    protocol: str
    number: int
    status: str
    wire: WireRecord
    retryable: bool | None = None
    error: ErrorInfo | None = None
    id: str = field(default_factory=lambda: new_id("attempt"))
    started_at: str = field(default_factory=now_iso)
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            k: deepcopy(v)
            for k, v in vars(self).items()
            if k not in {"wire", "error"} and v is not None
        }
        data["wire"] = self.wire.to_dict()
        if self.error:
            data["error"] = self.error.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestAttempt:
        return cls(
            id=data.get("id", new_id("attempt")),
            provider=data["provider"],
            model=data["model"],
            protocol=data["protocol"],
            number=int(data["number"]),
            status=data["status"],
            retryable=data.get("retryable"),
            started_at=data.get("started_at", now_iso()),
            completed_at=data.get("completed_at"),
            wire=WireRecord.from_dict(data.get("wire", {})),
            error=ErrorInfo.from_dict(data["error"]) if data.get("error") else None,
        )


@dataclass(frozen=True)
class ModelResponse:
    ok: bool
    content: str = ""
    reasoning: str | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    messages: tuple[Message, ...] = ()
    error: ErrorInfo | None = None
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    operation: Any = field(default=None, compare=False, repr=False)

    def raise_for_error(self) -> ModelResponse:
        if not self.ok:
            message = self.error.message if self.error else "Model request failed"
            raise RuntimeError(message)
        return self

    @property
    def standardized_response(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "native_finish_reason": self.native_finish_reason,
            "usage": deepcopy(self.usage),
        }

    @property
    def raw_provider_response(self) -> Any:
        return self.raw

    @property
    def error_info(self) -> dict[str, Any] | None:
        return self.error.to_dict() if self.error else None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "ok": self.ok,
            "content": self.content,
            "usage": deepcopy(self.usage),
            "messages": [message.to_dict() for message in self.messages],
            "raw": deepcopy(self.raw),
            "metadata": deepcopy(self.metadata),
        }
        for key in ("reasoning", "finish_reason", "native_finish_reason"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.error:
            data["error"] = self.error.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelResponse:
        return cls(
            ok=bool(data["ok"]),
            content=data.get("content", ""),
            reasoning=data.get("reasoning"),
            finish_reason=data.get("finish_reason"),
            native_finish_reason=data.get("native_finish_reason"),
            usage=deepcopy(data.get("usage", {})),
            messages=tuple(
                Message.from_dict(item) for item in data.get("messages", [])
            ),
            error=ErrorInfo.from_dict(data["error"]) if data.get("error") else None,
            raw=deepcopy(data.get("raw")),
            metadata=deepcopy(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ReplyOperation:
    request_spec: dict[str, Any]
    status: str
    input_message_ids: tuple[str, ...] = ()
    output_message_ids: tuple[str, ...] = ()
    attempts: tuple[RequestAttempt, ...] = ()
    result: ModelResponse | None = None
    error: ErrorInfo | None = None
    id: str = field(default_factory=lambda: new_id("op"))
    started_at: str = field(default_factory=now_iso)
    completed_at: str | None = None
    normalization: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "request_spec": deepcopy(self.request_spec),
            "status": self.status,
            "input_message_ids": list(self.input_message_ids),
            "output_message_ids": list(self.output_message_ids),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "started_at": self.started_at,
            "normalization": deepcopy(self.normalization),
            "metadata": deepcopy(self.metadata),
        }
        if self.completed_at is not None:
            data["completed_at"] = self.completed_at
        if self.result:
            data["result"] = self.result.to_dict()
        if self.error:
            data["error"] = self.error.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplyOperation:
        status = "interrupted" if data.get("status") == "running" else data["status"]
        return cls(
            id=data.get("id", new_id("op")),
            request_spec=deepcopy(data.get("request_spec", {})),
            status=status,
            input_message_ids=tuple(data.get("input_message_ids", [])),
            output_message_ids=tuple(data.get("output_message_ids", [])),
            attempts=tuple(
                RequestAttempt.from_dict(item) for item in data.get("attempts", [])
            ),
            result=ModelResponse.from_dict(data["result"])
            if data.get("result")
            else None,
            error=ErrorInfo.from_dict(data["error"]) if data.get("error") else None,
            started_at=data.get("started_at", now_iso()),
            completed_at=data.get("completed_at"),
            normalization=deepcopy(data.get("normalization", {})),
            metadata=deepcopy(data.get("metadata", {})),
        )


class ConversationBusyError(RuntimeError):
    """Raised when two writers attempt to mutate one conversation."""


class UnboundConversationError(RuntimeError):
    """Raised when execution is attempted without a usable runtime binding."""


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    operations: list[ReplyOperation] = field(default_factory=list)
    default_route: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    _binding: Any = field(default=None, repr=False, compare=False)
    _busy: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._busy is None:
            import threading

            self._busy = threading.Lock()
        self._validate()

    def _validate(self) -> None:
        message_ids = [message.id for message in self.messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("Conversation message IDs must be unique")
        operation_ids = [operation.id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("Conversation operation IDs must be unique")
        known = set(message_ids)
        for operation in self.operations:
            missing = (
                set(operation.input_message_ids + operation.output_message_ids) - known
            )
            if missing:
                raise ValueError(
                    f"Operation {operation.id} references unknown messages: {sorted(missing)}"
                )

    @property
    def last_message(self) -> Message | None:
        return self.messages[-1] if self.messages else None

    @property
    def last_operation(self) -> ReplyOperation | None:
        return self.operations[-1] if self.operations else None

    @property
    def is_bound(self) -> bool:
        return self._binding is not None

    @property
    def pending_messages(self) -> list[Message]:
        last_answer = max(
            (
                index
                for index, message in enumerate(self.messages)
                if message.role == "assistant"
            ),
            default=-1,
        )
        return [
            message
            for message in self.messages[last_answer + 1 :]
            if message.role in {"user", "tool"}
        ]

    def bind(self, client: Any, *, local_endpoint: str | None = None) -> Conversation:
        self._binding = client._bind_conversation(self, local_endpoint=local_endpoint)
        return self

    def send(self, content: Any, **options: Any) -> ModelResponse:
        if self._binding is None:
            raise UnboundConversationError(
                "Conversation has no runtime binding. Call conversation.bind(client) before sending."
            )
        if not self._busy.acquire(blocking=False):
            raise ConversationBusyError(
                "This conversation already has an active writer"
            )
        try:
            return self._binding._send_conversation(self, content, options)
        finally:
            self._busy.release()

    def send_pending(self, **options: Any) -> ModelResponse:
        """Submit imported trailing user/tool messages without appending input."""
        if self._binding is None:
            raise UnboundConversationError(
                "Conversation has no runtime binding. Call conversation.bind(client) before sending."
            )
        if not self.pending_messages:
            raise ValueError(
                "Conversation has no pending user or tool messages to send"
            )
        if not self._busy.acquire(blocking=False):
            raise ConversationBusyError(
                "This conversation already has an active writer"
            )
        try:
            return self._binding._send_conversation(
                self, None, options, append_input=False
            )
        finally:
            self._busy.release()

    def rebind(self, model: Any, *, provider_state: str = "error") -> Conversation:
        states = [
            message.provider_state
            for message in self.messages
            if message.provider_state
        ]
        if states and provider_state != "drop":
            raise ValueError(
                "Conversation contains provider replay state; pass provider_state='drop' to rebind"
            )
        if provider_state == "drop":
            self.messages = [
                Message(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    status=m.status,
                    metadata=deepcopy(m.metadata),
                )
                for m in self.messages
            ]
        self.default_route = model.serialized_route
        self._binding = model
        self.metadata.setdefault("llm_client", {}).setdefault("rebinds", []).append(
            {
                "at": now_iso(),
                "route": deepcopy(self.default_route),
                "provider_state": provider_state,
            }
        )
        return self

    def fork(self) -> Conversation:
        return Conversation(
            messages=list(self.messages),
            operations=list(self.operations),
            default_route=deepcopy(self.default_route),
            metadata=deepcopy(self.metadata),
            schema_version=self.schema_version,
            _binding=self._binding,
        )

    def tail(self, count: int) -> Conversation:
        return self.with_messages(self.messages[-count:])

    def with_messages(self, messages: Iterable[Message]) -> Conversation:
        selected = list(messages)
        selected_ids = {message.id for message in selected}
        operations = [
            op
            for op in self.operations
            if set(op.input_message_ids + op.output_message_ids) <= selected_ids
        ]
        return Conversation(
            messages=selected,
            operations=operations,
            default_route=deepcopy(self.default_route),
            metadata=deepcopy(self.metadata),
            schema_version=self.schema_version,
            _binding=self._binding,
        )

    def to_dict(self, *, secrets: Iterable[str] = ()) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "default_route": deepcopy(self.default_route),
            "messages": [message.to_dict() for message in self.messages],
            "operations": [operation.to_dict() for operation in self.operations],
            "metadata": deepcopy(self.metadata),
        }
        return _redact(data, secrets)

    def to_json(self, *, secrets: Iterable[str] = (), indent: int | None = None) -> str:
        return json.dumps(self.to_dict(secrets=secrets), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, client: Any = None) -> Conversation:
        version = int(data.get("schema_version", 1))
        if version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported conversation schema_version: {version}")
        known = {
            "schema_version",
            "default_route",
            "messages",
            "operations",
            "metadata",
        }
        metadata = deepcopy(data.get("metadata", {}))
        extras = {
            key: deepcopy(value) for key, value in data.items() if key not in known
        }
        if extras:
            metadata.setdefault("migration", {})["unmapped_conversation_fields"] = (
                extras
            )
        conversation = cls(
            schema_version=version,
            default_route=deepcopy(data.get("default_route", {})),
            messages=[Message.from_dict(item) for item in data.get("messages", [])],
            operations=[
                ReplyOperation.from_dict(item) for item in data.get("operations", [])
            ],
            metadata=metadata,
        )
        if client is not None:
            conversation.bind(client)
        return conversation

    @classmethod
    def from_json(cls, data: str, *, client: Any = None) -> Conversation:
        return cls.from_dict(json.loads(data), client=client)

    @classmethod
    def from_messages(
        cls,
        messages: Iterable[dict[str, Any] | Message],
        *,
        model: str | None = None,
        protocol: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        normalized = [
            item if isinstance(item, Message) else Message.from_standard(item)
            for item in messages
        ]
        route = {"model": model, "protocol": protocol} if model else {}
        return cls(
            messages=normalized, default_route=route, metadata=deepcopy(metadata or {})
        )

    @classmethod
    def import_legacy(
        cls,
        response: Any,
        *,
        messages: Iterable[dict[str, Any]] = (),
        model: str = "legacy/unset",
        preserve_source: str = "full",
    ) -> Conversation:
        from .v2_builder import ConversationBuilder

        return ConversationBuilder.import_legacy_response(
            response,
            messages=messages,
            model=model,
            preserve_source=preserve_source,
        )

    def __copy__(self) -> Conversation:
        return self.fork()

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)
