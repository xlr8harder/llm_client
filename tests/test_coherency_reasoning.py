import unittest
from unittest.mock import patch

from llm_client.base import LLMResponse
from llm_client.testing.coherency import CoherencyTester


def make_response(raw, content="ok", finish_reason="stop"):
    return LLMResponse(
        success=True,
        standardized_response={"content": content, "finish_reason": finish_reason},
        raw_provider_response=raw,
        is_retryable=False,
    )


class TestCoherencyReasoning(unittest.TestCase):
    def test_reasoning_enabled_requires_thinking(self):
        # Mock retry_request to return a response WITHOUT reasoning fields
        def fake_retry_request(
            provider, messages, model_id, max_retries=4, context=None, **options
        ):
            raw = {
                "id": "x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            return make_response(raw)

        with patch("llm_client.testing.coherency.retry_request", fake_retry_request):
            tester = CoherencyTester(
                target_provider_name="openai",
                target_model_id="gpt-fake",
                test_prompts=[{"id": "t1", "prompt": "hi"}],
                num_workers=1,
                request_overrides={"reasoning": {"enabled": True}},
            )

            result = tester.test_model(
                {"id": "t1", "prompt": "hi"}, provider_filter=None
            )
            self.assertFalse(result["success"])
            self.assertIn("Reasoning expected", result.get("error", ""))

    def test_reasoning_disabled_disallows_thinking(self):
        # Mock retry_request to return a response WITH reasoning fields
        def fake_retry_request(
            provider, messages, model_id, max_retries=4, context=None, **options
        ):
            raw = {
                "id": "x",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello",
                            "reasoning": "I am thinking...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            return make_response(raw)

        with patch("llm_client.testing.coherency.retry_request", fake_retry_request):
            tester = CoherencyTester(
                target_provider_name="openai",
                target_model_id="gpt-fake",
                test_prompts=[{"id": "t1", "prompt": "hi"}],
                num_workers=1,
                request_overrides={"reasoning": {"enabled": False}},
            )

            result = tester.test_model(
                {"id": "t1", "prompt": "hi"}, provider_filter=None
            )
            self.assertFalse(result["success"])
            self.assertIn("Reasoning not expected", result.get("error", ""))

    def test_reasoning_enabled_accepts_token_only(self):
        # When reasoning is enabled, having reasoning tokens in usage should count
        def fake_retry_request(
            provider, messages, model_id, max_retries=4, context=None, **options
        ):
            if model_id == "judge-yes":
                # Judge says the answer is coherent
                return make_response(
                    {"id": "judge", "choices": [], "usage": {}}, content="YES"
                )
            # Target provider: no explicit message-level reasoning, but tokens recorded
            raw = {
                "id": "x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 7},
                },
            }
            return make_response(raw)

        with patch("llm_client.testing.coherency.retry_request", fake_retry_request):
            tester = CoherencyTester(
                target_provider_name="openai",
                target_model_id="gpt-fake",
                judge_provider_name="openai",
                judge_model_id="judge-yes",
                test_prompts=[{"id": "t1", "prompt": "hi"}],
                num_workers=1,
                request_overrides={"reasoning": {"enabled": True}},
            )

            result = tester.test_model(
                {"id": "t1", "prompt": "hi"}, provider_filter=None
            )
            self.assertTrue(
                result["success"]
            )  # passes reasoning gate and judge returns YES

    def test_reasoning_disabled_rejects_token_only(self):
        # When reasoning is disabled, having reasoning tokens should cause a failure
        def fake_retry_request(
            provider, messages, model_id, max_retries=4, context=None, **options
        ):
            raw = {
                "id": "x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            }
            return make_response(raw)

        with patch("llm_client.testing.coherency.retry_request", fake_retry_request):
            tester = CoherencyTester(
                target_provider_name="openai",
                target_model_id="gpt-fake",
                test_prompts=[{"id": "t1", "prompt": "hi"}],
                num_workers=1,
                request_overrides={"reasoning": {"enabled": False}},
            )

            result = tester.test_model(
                {"id": "t1", "prompt": "hi"}, provider_filter=None
            )
            self.assertFalse(result["success"])
            self.assertIn("Reasoning not expected", result.get("error", ""))


if __name__ == "__main__":
    unittest.main()
