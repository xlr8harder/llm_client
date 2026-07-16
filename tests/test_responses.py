from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx

from llm_client.providers.codex import CodexProvider
from llm_client.providers.openai_responses import OpenAIResponsesStyleProvider
from llm_client.providers.openrouter import OpenRouterProvider


class StubResponsesProvider(OpenAIResponsesStyleProvider):
    api_base = "https://responses.example/v1"
    api_key_env_var = "RESPONSES_TEST_KEY"
    provider_name = "responses_test"


class StubAuthManager:
    def get_headers(self):
        return {
            "Authorization": "Bearer access-token",
            "chatgpt-account-id": "acct_test",
        }


def test_chat_messages_translate_to_nonstreaming_response():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "created_at": 123,
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "brief thought"}
                        ],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    with patch.dict(os.environ, {"RESPONSES_TEST_KEY": "test-key"}):
        response = StubResponsesProvider(
            transport=httpx.MockTransport(handler)
        ).make_chat_completion_request(
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Again"},
            ],
            model_id="test-model",
            max_tokens=55,
            temperature=0.2,
        )

    assert response.success
    assert response.standardized_response["content"] == "hello"
    assert response.standardized_response["reasoning"] == "brief thought"
    assert response.request_format == "responses"
    request = seen[0]
    assert str(request.url) == "https://responses.example/v1/responses"
    payload = json.loads(request.content)
    assert payload["instructions"] == "Be terse."
    assert [item["role"] for item in payload["input"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert payload["max_output_tokens"] == 55
    assert payload["temperature"] == 0.2


def test_streaming_responses_are_aggregated():
    stream = "".join(
        [
            'data: {"type":"response.output_text.delta","delta":"hel"}\n\n',
            'data: {"type":"response.output_text.delta","delta":"lo"}\n\n',
            'data: {"type":"response.reasoning_summary_text.delta","delta":"why"}\n\n',
            'data: {"type":"response.completed","response":{"id":"resp_2","model":"test-model","status":"completed","output":[],"usage":{}}}\n\n',
        ]
    )

    def handler(request):
        return httpx.Response(
            200,
            text=stream,
            headers={"content-type": "text/event-stream"},
        )

    with patch.dict(os.environ, {"RESPONSES_TEST_KEY": "test-key"}):
        response = StubResponsesProvider(
            transport=httpx.MockTransport(handler)
        ).make_responses_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="test-model",
            transport="stream",
        )

    assert response.success
    assert response.standardized_response["content"] == "hello"
    assert response.standardized_response["reasoning"] == "why"


def test_top_k_fails_before_network():
    called = False

    def handler(request):
        nonlocal called
        called = True
        return httpx.Response(500)

    with patch.dict(os.environ, {"RESPONSES_TEST_KEY": "test-key"}):
        response = StubResponsesProvider(
            transport=httpx.MockTransport(handler)
        ).make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="test-model",
            top_k=10,
        )
    assert not response.success
    assert response.error_info["type"] == "invalid_option"
    assert "Remove --top-k" in response.error_info["message"]
    assert not called


def test_direct_stream_flag_retains_legacy_validation():
    with patch.dict(os.environ, {"RESPONSES_TEST_KEY": "test-key"}):
        response = StubResponsesProvider().make_responses_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="test-model",
            stream=True,
        )
    assert not response.success
    assert response.error_info["type"] == "invalid_option"
    assert "transport='stream'" in response.error_info["message"]


def test_openrouter_explicit_responses_format():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_or",
                "model": "openai/gpt-5",
                "status": "completed",
                "output_text": "router response",
            },
        )

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}):
        response = OpenRouterProvider(
            transport=httpx.MockTransport(handler)
        ).make_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="openai/gpt-5",
            request_format="responses",
            only=["openai"],
        )
    assert response.success
    request = seen[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/responses"
    payload = json.loads(request.content)
    assert payload["provider"]["order"] == ["openai"]
    assert payload["provider"]["allow_fallbacks"] is False


def test_codex_provider_uses_responses_and_oauth_headers():
    seen = []
    stream = "".join(
        [
            'data: {"type":"response.output_text.delta","delta":"ok"}\n\n',
            'data: {"type":"response.completed","response":{"id":"resp_c","model":"gpt-5.4","status":"completed","output":[]}}\n\n',
        ]
    )

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            text=stream,
            headers={"content-type": "text/event-stream"},
        )

    response = CodexProvider(
        StubAuthManager(), transport=httpx.MockTransport(handler)
    ).make_chat_completion_request(
        messages=[{"role": "user", "content": "Hi"}],
        model_id="codex/gpt-5.4",
    )
    assert response.success
    request = seen[0]
    assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
    assert request.headers["authorization"] == "Bearer access-token"
    assert request.headers["chatgpt-account-id"] == "acct_test"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-5.4"
    assert payload["stream"] is True
    assert payload["store"] is False


def test_codex_without_oauth_configuration_fails_clearly():
    with patch.dict(os.environ, {}, clear=True):
        response = CodexProvider().make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="codex/gpt-5.4",
        )
    assert not response.success
    assert response.error_info["type"] == "auth_error"
    assert "LLM_CLIENT_CODEX_CLIENT_ID" in response.error_info["message"]
