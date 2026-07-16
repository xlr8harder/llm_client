from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from llm_client import get_provider
from llm_client.providers.codex import CodexAuthManager, CodexProvider
from llm_client.providers.openai_responses import OpenAIResponsesStyleProvider


def fake_jwt(*, exp: float, account_id: str = "acct_test") -> str:
    def encode(value):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return ".".join(
        [
            encode({"alg": "none"}),
            encode(
                {
                    "exp": exp,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": account_id
                    },
                }
            ),
            "signature",
        ]
    )


class FakeResponse:
    def __init__(self, payload, *, status=200, chunks=None):
        self.status = status
        self.data = json.dumps(payload).encode("utf-8")
        self._chunks = chunks or []
        self.closed = False

    def stream(self, decode_content=True):
        yield from self._chunks

    def close(self):
        self.closed = True


class StubResponsesProvider(OpenAIResponsesStyleProvider):
    api_base = "https://responses.example/v1"
    api_key_env_var = "RESPONSES_TEST_KEY"
    provider_name = "responses_test"


class StubAuthManager:
    def get_access(self):
        return "access-token", "acct_test"


class TestResponsesTransport(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RESPONSES_TEST_KEY": "test-key"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    @patch("urllib3.PoolManager.request")
    def test_chat_messages_translate_to_nonstreaming_response(self, request):
        request.return_value = FakeResponse(
            {
                "id": "resp_1",
                "created_at": 123,
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "brief thought"}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        )

        response = StubResponsesProvider().make_chat_completion_request(
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

        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "hello")
        self.assertEqual(response.standardized_response["reasoning"], "brief thought")
        self.assertEqual(response.request_format, "responses")
        args, kwargs = request.call_args
        self.assertEqual(args[:2], ("POST", "https://responses.example/v1/responses"))
        payload = json.loads(kwargs["body"])
        self.assertEqual(payload["instructions"], "Be terse.")
        self.assertEqual([item["role"] for item in payload["input"]], ["user", "assistant", "user"])
        self.assertEqual(payload["max_output_tokens"], 55)
        self.assertEqual(payload["temperature"], 0.2)

    @patch("urllib3.PoolManager.request")
    def test_streaming_responses_are_aggregated(self, request):
        chunks = [
            b"event: response.output_text.delta\n",
            b'data: {"type":"response.output_text.delta","delta":"hel"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"lo"}\n\n',
            b'data: {"type":"response.reasoning_summary_text.delta","delta":"why"}\n\n',
            b'data: {"type":"response.completed","response":{"id":"resp_2","model":"test-model","status":"completed","output":[],"usage":{}}}\n\n',
        ]
        request.return_value = FakeResponse({}, chunks=chunks)

        response = StubResponsesProvider().make_responses_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="test-model",
            transport="stream",
        )

        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "hello")
        self.assertEqual(response.standardized_response["reasoning"], "why")

    @patch("urllib3.PoolManager.request")
    def test_top_k_fails_before_network(self, request):
        response = StubResponsesProvider().make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="test-model",
            top_k=10,
        )
        self.assertFalse(response.success)
        self.assertEqual(response.error_info["type"], "invalid_option")
        self.assertIn("Remove --top-k", response.error_info["message"])
        request.assert_not_called()

    @patch("urllib3.PoolManager.request")
    def test_openrouter_explicit_responses_format(self, request):
        request.return_value = FakeResponse(
            {
                "id": "resp_or",
                "model": "openai/gpt-5",
                "status": "completed",
                "output_text": "router response",
            }
        )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}):
            response = get_provider("openrouter").make_request(
                messages=[{"role": "user", "content": "Hi"}],
                model_id="openai/gpt-5",
                request_format="responses",
                only=["openai"],
            )
        self.assertTrue(response.success)
        args, kwargs = request.call_args
        self.assertEqual(args[1], "https://openrouter.ai/api/v1/responses")
        payload = json.loads(kwargs["body"])
        self.assertEqual(payload["provider"]["order"], ["openai"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])

    @patch("urllib3.PoolManager.request")
    def test_codex_provider_uses_responses_and_oauth_headers(self, request):
        request.return_value = FakeResponse(
            {},
            chunks=[
                b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n',
                b'data: {"type":"response.completed","response":{"id":"resp_c","model":"gpt-5.4","status":"completed","output":[]}}\n\n',
            ],
        )
        response = CodexProvider(StubAuthManager()).make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="codex/gpt-5.4",
        )
        self.assertTrue(response.success)
        args, kwargs = request.call_args
        self.assertEqual(args[1], "https://chatgpt.com/backend-api/codex/responses")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer access-token")
        self.assertEqual(kwargs["headers"]["chatgpt-account-id"], "acct_test")
        payload = json.loads(kwargs["body"])
        self.assertEqual(payload["model"], "gpt-5.4")
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["store"])


class TestCodexAuthManager(unittest.TestCase):
    def write_auth(self, path: Path, access: str, refresh="refresh-old"):
        path.write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "tokens": {
                        "access_token": access,
                        "refresh_token": refresh,
                        "account_id": "acct_fallback",
                    },
                }
            ),
            encoding="utf-8",
        )
        os.chmod(path, 0o600)

    def test_fresh_token_is_returned_without_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            access = fake_jwt(exp=time.time() + 3600)
            self.write_auth(path, access)
            with patch("requests.post") as post:
                result = CodexAuthManager(path).get_access()
            self.assertEqual(result, (access, "acct_test"))
            post.assert_not_called()

    @patch("requests.post")
    def test_expired_token_refreshes_and_updates_auth_atomically(self, post):
        class RefreshResponse:
            status_code = 200

            def json(self):
                return {
                    "access_token": fake_jwt(exp=time.time() + 3600, account_id="acct_new"),
                    "refresh_token": "refresh-new",
                }

        post.return_value = RefreshResponse()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            self.write_auth(path, fake_jwt(exp=time.time() - 60))
            access, account = CodexAuthManager(path).get_access()
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved_mode = oct(os.stat(path).st_mode & 0o777)

        self.assertEqual(account, "acct_new")
        self.assertEqual(saved["tokens"]["access_token"], access)
        self.assertEqual(saved["tokens"]["refresh_token"], "refresh-new")
        self.assertEqual(saved_mode, "0o600")

    @patch("requests.post")
    def test_parallel_callers_perform_one_refresh(self, post):
        new_access = fake_jwt(exp=time.time() + 3600, account_id="acct_new")

        class RefreshResponse:
            status_code = 200

            def json(self):
                return {"access_token": new_access, "refresh_token": "refresh-new"}

        def refresh(*args, **kwargs):
            time.sleep(0.05)
            return RefreshResponse()

        post.side_effect = refresh
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "auth.json"
            self.write_auth(path, fake_jwt(exp=time.time() - 60))
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(lambda _: CodexAuthManager(path).get_access(), range(8))
                )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(results, [(new_access, "acct_new")] * 8)


if __name__ == "__main__":
    unittest.main()
