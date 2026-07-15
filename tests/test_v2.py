import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import json
import threading
import warnings

import httpx
import pytest

from llm_client import (
    AsyncClient,
    Client,
    Conversation,
    ConversationBuilder,
    ConversationBusyError,
    LLMResponse,
    UnknownProviderFieldWarning,
)


def chat_response(text="hello", **extra):
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "model": "test/model",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "native_finish_reason": "end_turn",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        **extra,
    }


def responses_response(text="hello", **extra):
    return {
        "id": "resp-1",
        "object": "response",
        "status": "completed",
        "model": "test/model",
        "output": [
            {"type": "reasoning", "encrypted_content": "opaque"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        **extra,
    }


def test_simple_chat_path_and_round_trip():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response("simple"))

    with Client(
        api_keys={"openrouter": "secret"}, transport=httpx.MockTransport(handler)
    ) as client:
        conversation = client.model("openrouter/test/model").conversation(
            system="Be concise"
        )
        response = conversation.send("Question")

    assert response.ok
    assert response.content == "simple"
    assert response.operation.status == "succeeded"
    assert seen[0]["messages"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Question"},
    ]
    restored = Conversation.from_json(conversation.to_json())
    assert restored == conversation
    assert restored.to_dict() == conversation.to_dict()


def test_generate_rejects_content_and_messages_together():
    with Client(transport=httpx.MockTransport(lambda request: None)) as client:
        with pytest.raises(ValueError, match="either content or messages"):
            client.generate(
                "duplicate",
                model="openrouter/openai/test",
                messages=[{"role": "user", "content": "already supplied"}],
            )


def test_send_pending_submits_imported_messages_without_duplication():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response("answer"))

    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model("openrouter/openai/test").conversation(
            messages=[{"role": "user", "content": "imported"}]
        )
        response = conversation.send_pending(max_tokens=20)

    assert response.content == "answer"
    assert seen[0]["messages"] == [{"role": "user", "content": "imported"}]
    assert conversation.last_operation.input_message_ids == (
        conversation.messages[0].id,
    )


def test_send_pending_requires_pending_input():
    with Client() as client:
        conversation = client.model("openrouter/openai/test").conversation()
        with pytest.raises(ValueError, match="no pending"):
            conversation.send_pending()


def test_unimplemented_protocol_is_rejected():
    with Client() as client:
        with pytest.raises(ValueError, match="Unsupported protocol"):
            client.model("openrouter/openai/test", protocol="completions")


def test_responses_replays_provider_state_on_followup():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses_response(f"answer-{len(requests)}"))

    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model(
            "openrouter/test/model", protocol="responses"
        ).conversation()
        assert conversation.send("first").content == "answer-1"
        assert conversation.send("second").content == "answer-2"

    second_input = requests[1]["input"]
    assert {"type": "reasoning", "encrypted_content": "opaque"} in second_input
    assert any(item.get("type") == "message" for item in second_input)
    restored = Conversation.from_json(conversation.to_json())
    assert restored == conversation


def test_local_route_and_wire_hide_endpoint_and_require_rebinding():
    endpoint = "http://192.168.44.9:8123/v1"

    def handler(request):
        return httpx.Response(200, json=chat_response("local"))

    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model("local/192.168.44.9:8123/qwen3-4b").conversation()
        conversation.send("hello")
        payload = conversation.to_json()

    assert "192.168.44.9" not in payload
    assert "8123" not in payload
    assert "local/unset/qwen3-4b" in payload
    restored = Conversation.from_json(payload)
    with pytest.raises(ValueError, match="no bound local endpoint"):
        restored.bind(Client(transport=httpx.MockTransport(handler)))
    rebound_client = Client(
        local_endpoint=endpoint, transport=httpx.MockTransport(handler)
    )
    restored.bind(rebound_client)
    assert restored.send("again").content == "local"
    rebound_client.close()


def test_local_unset_route_preserves_slash_qualified_model_id():
    with Client(local_endpoint="http://127.0.0.1:8000/v1") as client:
        model = client.model("local/unset/Qwen/Qwen3-4B")
        assert model.model_id == "Qwen/Qwen3-4B"
        assert model.serialized_route["model"] == "local/unset/Qwen/Qwen3-4B"


def test_wire_redacts_auth_and_preserves_openrouter_metadata_and_unknown_fields():
    raw = chat_response(
        "ok",
        openrouter_metadata={"attempts": [{"provider": "DeepInfra", "status": 200}]},
        future_router_field={"enabled": True},
    )

    def handler(request):
        return httpx.Response(200, headers={"x-generation-id": "gen-1"}, json=raw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Client(
            api_keys={"openrouter": "top-secret"},
            transport=httpx.MockTransport(handler),
        ) as client:
            conversation = client.model("openrouter/test/model").conversation()
            response = conversation.send("hello", openrouter_metadata=True)

    wire = response.operation.attempts[0].wire
    assert wire.request["headers"]["Authorization"] == "[REDACTED]"
    assert wire.response["body"] == raw
    assert (
        response.metadata["provider"]["openrouter"]["attempts"][0]["provider"]
        == "DeepInfra"
    )
    assert (
        response.operation.normalization["notices"][0]["path"] == "future_router_field"
    )
    assert any(isinstance(item.message, UnknownProviderFieldWarning) for item in caught)


def test_openrouter_metadata_enrichment_is_opt_in_and_expected_fields_are_normalized():
    seen = []
    raw = chat_response("ok", provider="OpenAI", service_tier="default")

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=raw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with Client(transport=httpx.MockTransport(handler)) as client:
            response = client.generate("hello", model="openrouter/test/model")

    assert "X-OpenRouter-Metadata" not in seen[0].headers
    assert response.raw == raw
    assert response.metadata["provider"]["openrouter"] == {
        "selected_backend": "OpenAI",
        "service_tier": "default",
    }
    assert not caught


def test_provider_error_is_operation_not_message_and_can_round_trip():
    def handler(request):
        return httpx.Response(
            400,
            json={
                "error": {"code": "content_filter", "message": "blocked by moderation"}
            },
        )

    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model("openrouter/test/model").conversation()
        response = conversation.send("blocked")
    assert not response.ok
    assert response.error.category == "moderation"
    assert conversation.last_message.role == "user"
    assert conversation.last_operation.status == "moderated"
    assert conversation.pending_messages[-1].text == "blocked"
    assert Conversation.from_json(conversation.to_json()) == conversation


@pytest.mark.parametrize(
    ("native", "category", "status"),
    [
        ("max_tokens", "truncation", "truncated"),
        ("safety", "moderation", "moderated"),
        ("new_reason", "invalid_response", "failed"),
    ],
)
def test_finish_reason_classification_preserves_output_and_native_reason(
    native, category, status
):
    raw = chat_response("partial")
    raw["choices"][0]["finish_reason"] = native
    raw["choices"][0].pop("native_finish_reason", None)

    def handler(request):
        return httpx.Response(200, json=raw)

    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model("openrouter/test/model").conversation()
        response = conversation.send("hello")
    assert not response.ok
    assert response.content == "partial"
    assert response.native_finish_reason == native
    assert response.error.category == category
    assert response.operation.status == status
    assert conversation.last_message.role == "assistant"


@pytest.mark.parametrize(
    ("native", "normalized", "category"),
    [
        ("STOP", "stop", None),
        ("completed", "stop", None),
        ("end_turn", "stop", None),
        ("eos_token", "stop", None),
        ("eos", "stop", None),
        ("PROHIBITED_CONTENT", "content_filter", "moderation"),
        ("RECITATION", "content_filter", "moderation"),
        ("refusal", "content_filter", "moderation"),
        ("sensitive", "content_filter", "moderation"),
        ("MAX_TOKENS", "length", "truncation"),
        ("max_output_tokens", "length", "truncation"),
        ("max_tokens", "length", "truncation"),
        ("engine_overloaded", "error", "provider"),
        ("OTHER", "unknown", "invalid_response"),
    ],
)
def test_llm_compliance_finish_reason_taxonomy(native, normalized, category):
    raw = chat_response("result")
    raw["choices"][0]["finish_reason"] = native
    raw["choices"][0].pop("native_finish_reason", None)

    with Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))
    ) as client:
        response = client.generate("test", model="openrouter/test/model")

    assert response.finish_reason == normalized
    assert response.native_finish_reason == native
    assert (response.error.category if response.error else None) == category


def test_missing_finish_reason_is_invalid_response():
    raw = chat_response("result")
    del raw["choices"][0]["finish_reason"]
    raw["choices"][0].pop("native_finish_reason", None)

    with Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=raw))
    ) as client:
        response = client.generate("test", model="openrouter/test/model")

    assert response.ok is False
    assert response.error.category == "invalid_response"


def test_messages_protocol_system_thinking_and_raw_metadata():
    seen = []
    raw = {
        "id": "msg-1",
        "type": "message",
        "role": "assistant",
        "model": "anthropic/test",
        "content": [
            {"type": "thinking", "thinking": "summary", "signature": "signed"},
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=raw)

    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model(
            "openrouter/anthropic/test", protocol="messages"
        ).conversation(system="rules")
        response = conversation.send(
            "question", provider_options={"only": ["anthropic"]}
        )
    assert response.content == "answer"
    assert response.messages[0].provider_state.replay[0]["signature"] == "signed"
    assert seen[0]["system"] == "rules"
    assert seen[0]["provider"] == {"only": ["anthropic"]}
    assert response.raw == raw


def test_responses_sse_is_aggregated_and_events_are_preserved():
    final = responses_response("streamed")
    body = (
        "\n\n".join(
            [
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"stream"}',
                'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"ed"}',
                f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'response': final})}",
                "data: [DONE]",
            ]
        )
        + "\n\n"
    )

    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=body
        )

    with Client(transport=httpx.MockTransport(handler)) as client:
        response = client.generate(
            "hello", model="openrouter/test/model", protocol="responses", stream=True
        )
    assert response.content == "streamed"
    events = response.operation.attempts[0].wire.stream_events
    assert len(events) == 3
    assert events[0]["_event"] == "response.output_text.delta"


def test_retries_are_retained_as_attempts():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": {"message": "warming"}})
        return httpx.Response(200, json=chat_response("ready"))

    with Client(max_retries=1, transport=httpx.MockTransport(handler)) as client:
        response = client.generate("hello", model="openrouter/test/model")
    assert response.content == "ready"
    assert [
        attempt.wire.response["status_code"] for attempt in response.operation.attempts
    ] == [503, 200]


def test_legacy_import_is_lossless_and_round_trips():
    legacy = LLMResponse(
        success=False,
        standardized_response=None,
        error_info={
            "type": "content_filter",
            "message": "blocked",
            "native_finish_reason": "RECITATION",
        },
        raw_provider_response={"error": {"message": "blocked", "provider": "upstream"}},
        request_format="chat_completions",
        raw_response_format="openrouter.chat_completions",
        is_retryable=False,
        context={"question_id": "q1"},
    )
    conversation = ConversationBuilder.import_legacy_response(
        legacy,
        messages=[{"role": "user", "content": "prompt"}],
        model="openrouter/test/model",
    )
    source = conversation.last_operation.result.metadata["migration"]["source_record"]
    assert source["error_info"] == legacy.error_info
    assert source["raw_provider_response"] == legacy.raw_provider_response
    assert source["context"] == legacy.context
    assert Conversation.from_json(conversation.to_json()) == conversation


def test_copy_and_rebind_are_explicit():
    conversation = Conversation.from_messages(
        [{"role": "user", "content": "x"}], model="local/unset/qwen3"
    )
    fork = copy(conversation)
    assert fork.messages is not conversation.messages
    assert fork.messages[0] is conversation.messages[0]


def test_imported_completed_history_is_not_pending():
    completed = Conversation.from_messages(
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
        model="openrouter/test/model",
    )
    assert completed.pending_messages == []
    pending = Conversation.from_messages(
        [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ],
        model="openrouter/test/model",
    )
    assert [message.text for message in pending.pending_messages] == ["new question"]


def test_thread_parallelism_shares_client_without_state_leakage():
    lock = threading.Lock()
    observed = []

    def handler(request):
        body = json.loads(request.content)
        text = body["messages"][-1]["content"]
        with lock:
            observed.append(text)
        return httpx.Response(200, json=chat_response(f"reply:{text}"))

    with Client(transport=httpx.MockTransport(handler)) as client:
        model = client.model("openrouter/test/model")
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(
                pool.map(
                    lambda i: model.conversation().send(f"p{i}").content, range(50)
                )
            )
    assert sorted(results) == sorted(f"reply:p{i}" for i in range(50))
    assert sorted(observed) == sorted(f"p{i}" for i in range(50))


def test_same_conversation_rejects_concurrent_writer():
    entered = threading.Event()
    release = threading.Event()

    def handler(request):
        entered.set()
        release.wait(2)
        return httpx.Response(200, json=chat_response())

    with Client(transport=httpx.MockTransport(handler)) as client:
        conversation = client.model("openrouter/test/model").conversation()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(conversation.send, "one")
            assert entered.wait(1)
            with pytest.raises(ConversationBusyError):
                conversation.send("two")
            release.set()
            future.result()


def test_native_async_client_and_round_trip():
    async def run():
        async def handler(request):
            await asyncio.sleep(0)
            return httpx.Response(200, json=responses_response("async"))

        async with AsyncClient(transport=httpx.MockTransport(handler)) as client:
            conversation = client.model(
                "openrouter/test/model", protocol="responses"
            ).conversation()
            response = await conversation.send("hello")
        assert response.content == "async"
        assert (
            Conversation.from_json(conversation.to_json()).to_dict()
            == conversation.to_dict()
        )

    asyncio.run(run())
