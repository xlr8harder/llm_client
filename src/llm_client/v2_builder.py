"""Canonical assembly pipeline used by execution, imports, and restoration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .base import LLMResponse
from .v2_models import (
    Conversation,
    ErrorInfo,
    Message,
    ModelResponse,
    ProviderState,
    ReplyOperation,
    RequestAttempt,
    new_id,
    now_iso,
)


class ConversationBuilder:
    def __init__(self, conversation: Conversation | None = None):
        self._conversation = conversation.fork() if conversation else Conversation()
        self._operation: dict[str, Any] | None = None

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationBuilder:
        return cls(conversation)

    def add_message(
        self,
        role: str,
        content: Any,
        *,
        id: str | None = None,
        status: str = "complete",
        provider_state: ProviderState | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role}")
        message = Message.from_dict(
            {
                "id": id or new_id("msg"),
                "role": role,
                "content": content,
                "status": status,
                "provider_state": provider_state.to_dict() if provider_state else None,
                "metadata": deepcopy(metadata or {}),
            }
        )
        self._conversation.messages.append(message)
        return message

    def begin_operation(
        self, request_spec: dict[str, Any], input_message_ids: Iterable[str]
    ) -> str:
        if self._operation is not None:
            raise RuntimeError("A builder operation is already active")
        operation_id = new_id("op")
        self._operation = {
            "id": operation_id,
            "request_spec": deepcopy(request_spec),
            "status": "running",
            "input_message_ids": tuple(input_message_ids),
            "output_message_ids": [],
            "attempts": [],
            "started_at": now_iso(),
            "normalization": {},
            "metadata": {},
        }
        return operation_id

    def add_attempt(self, attempt: RequestAttempt) -> None:
        if self._operation is None:
            raise RuntimeError("No active operation")
        expected = len(self._operation["attempts"]) + 1
        if attempt.number != expected:
            raise ValueError(f"Attempt number must be {expected}, got {attempt.number}")
        self._operation["attempts"].append(attempt)

    def add_output_message(self, message: Message) -> None:
        if self._operation is None:
            raise RuntimeError("No active operation")
        if message.id not in {item.id for item in self._conversation.messages}:
            self._conversation.messages.append(message)
        self._operation["output_message_ids"].append(message.id)

    def add_normalization_evidence(self, field: str, evidence: dict[str, Any]) -> None:
        if self._operation is None:
            raise RuntimeError("No active operation")
        self._operation["normalization"].setdefault("evidence", {})[field] = deepcopy(
            evidence
        )

    def add_normalization_notice(self, notice: dict[str, Any]) -> None:
        if self._operation is None:
            raise RuntimeError("No active operation")
        self._operation["normalization"].setdefault("notices", []).append(
            deepcopy(notice)
        )

    def complete_operation(
        self, result: ModelResponse, *, status: str | None = None
    ) -> ReplyOperation:
        if self._operation is None:
            raise RuntimeError("No active operation")
        data = self._operation
        operation = ReplyOperation(
            id=data["id"],
            request_spec=data["request_spec"],
            status=status
            or ("succeeded" if result.ok else _status_for_error(result.error)),
            input_message_ids=tuple(data["input_message_ids"]),
            output_message_ids=tuple(data["output_message_ids"]),
            attempts=tuple(data["attempts"]),
            result=result,
            error=result.error,
            started_at=data["started_at"],
            completed_at=now_iso(),
            normalization=deepcopy(data["normalization"]),
            metadata=deepcopy(data["metadata"]),
        )
        self._conversation.operations.append(operation)
        self._operation = None
        return operation

    def interrupt_operation(self, error: ErrorInfo) -> ReplyOperation:
        result = ModelResponse(ok=False, error=error)
        return self.complete_operation(result, status="interrupted")

    def build(self) -> Conversation:
        if self._operation is not None:
            raise RuntimeError("Cannot build while an operation is active")
        self._conversation._validate()
        return self._conversation

    @classmethod
    def import_legacy_response(
        cls,
        response: LLMResponse,
        *,
        messages: Iterable[dict[str, Any]] = (),
        model: str = "legacy/unset",
        preserve_source: str = "full",
    ) -> Conversation:
        builder = cls(Conversation.from_messages(messages, model=model))
        input_ids = [
            message.id
            for message in builder._conversation.messages
            if message.role in {"user", "tool"}
        ]
        request_spec = {
            "requested": {"model": model},
            "resolved": {"protocol": response.request_format or "chat_completions"},
        }
        builder.begin_operation(request_spec, input_ids)
        error = _legacy_error(response)
        raw = deepcopy(response.raw_provider_response)
        standard = deepcopy(response.standardized_response or {})
        outputs: list[Message] = []
        content = standard.get("content")
        if content is not None:
            message = builder.add_message(
                "assistant",
                str(content),
                metadata={"migration": {"source": "LLMResponse.standardized_response"}},
            )
            builder.add_output_message(message)
            outputs.append(message)
        result = ModelResponse(
            ok=bool(response.success),
            content=str(content or ""),
            reasoning=standard.get("reasoning"),
            finish_reason=standard.get("finish_reason"),
            native_finish_reason=standard.get("native_finish_reason"),
            usage=deepcopy(standard.get("usage", {})),
            messages=tuple(outputs),
            error=error,
            raw=raw,
            metadata={
                "migration": {
                    "source_format": "llm_client.LLMResponse",
                    "raw_response_format": response.raw_response_format,
                    "context": deepcopy(response.context),
                    "source_record": _legacy_source(response)
                    if preserve_source == "full"
                    else None,
                }
            },
        )
        builder.complete_operation(result)
        return builder.build()


def _status_for_error(error: ErrorInfo | None) -> str:
    if not error:
        return "failed"
    return {
        "moderation": "moderated",
        "interrupted": "interrupted",
        "truncation": "truncated",
    }.get(error.category, "failed")


def _legacy_error(response: LLMResponse) -> ErrorInfo | None:
    if response.success and not response.error_info:
        return None
    info = deepcopy(response.error_info or {})
    return ErrorInfo(
        category=_legacy_category(info),
        message=str(info.get("message", "Legacy request failed")),
        retryable=bool(response.is_retryable),
        status_code=info.get("status_code"),
        provider_code=info.get("code"),
        provider_type=info.get("type"),
        finish_reason=info.get("finish_reason"),
        native_finish_reason=info.get("native_finish_reason"),
        raw=info,
    )


def _legacy_category(info: dict[str, Any]) -> str:
    value = str(info.get("type", "")).lower()
    if value in {"content_filter", "moderation"}:
        return "moderation"
    if "auth" in value:
        return "authentication"
    if "rate" in value:
        return "rate_limit"
    if "timeout" in value:
        return "timeout"
    return "provider"


def _legacy_source(response: LLMResponse) -> dict[str, Any]:
    return {
        "success": response.success,
        "standardized_response": deepcopy(response.standardized_response),
        "error_info": deepcopy(response.error_info),
        "raw_provider_response": deepcopy(response.raw_provider_response),
        "request_format": response.request_format,
        "raw_response_format": response.raw_response_format,
        "is_retryable": response.is_retryable,
        "context": deepcopy(response.context),
    }
