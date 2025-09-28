import unittest
from unittest.mock import patch

from llm_client.base import LLMResponse
from llm_client.testing.coherency import CoherencyTester


def make_failed_response(err_message, raw=None, status_code=404):
    return LLMResponse(
        success=False,
        standardized_response=None,
        error_info={"message": err_message, "status_code": status_code},
        raw_provider_response=raw or {"error": {"message": err_message, "code": status_code}},
        is_retryable=False,
    )


class TestForcedSubprovider(unittest.TestCase):
    def test_allowed_subproviders_requires_openrouter(self):
        # Using allowed_subproviders with non-OpenRouter must raise
        with self.assertRaises(ValueError):
            CoherencyTester(
                target_provider_name="openai",
                target_model_id="gpt-fake",
                test_prompts=[{"id": "t1", "prompt": "hi"}],
                num_workers=1,
                allowed_subproviders=["deepseek"],
            )

    def test_forced_subprovider_failure_marks_provider_failed(self):
        # Avoid network during provider listing
        with patch(
            "llm_client.providers.openrouter.OpenRouterProvider.get_available_providers",
            return_value=[],
        ):
            # Force a subprovider and make the request fail as OpenRouter would when endpoints are missing
            def fake_retry_request(provider, messages, model_id, max_retries=4, context=None, **options):
                return make_failed_response(
                    "No endpoints found for deepseek/deepseek-chat-v3.1.",
                    raw={"error": {"message": "No endpoints found for deepseek/deepseek-chat-v3.1.", "code": 404}},
                    status_code=404,
                )

            with patch("llm_client.testing.coherency.retry_request", fake_retry_request):
                tester = CoherencyTester(
                    target_provider_name="openrouter",
                    target_model_id="deepseek/deepseek-chat-v3.1",
                    test_prompts=[{"id": "t1", "prompt": "hello"}],
                    num_workers=1,
                    allowed_subproviders=["deepseek"],
                    request_overrides={"reasoning": {"enabled": True}},
                    verbose=True,
                )

                results = tester.run_tests()

                # The forced provider should appear as failed and nothing should pass
                self.assertFalse(results.get("all_passed", True))
                self.assertIn("deepseek", results.get("failed_providers", []))
                self.assertEqual(results.get("passed_providers", []), [])

    def test_forced_subprovider_list_is_used_verbatim(self):
        # If OpenRouter reports no endpoints, the tester should still schedule the forced providers
        with patch(
            "llm_client.providers.openrouter.OpenRouterProvider.get_available_providers",
            return_value=[],
        ):
            # Don't run full tests; just construct and inspect internal list
            tester = CoherencyTester(
                target_provider_name="openrouter",
                target_model_id="some/model",
                test_prompts=[{"id": "t1", "prompt": "hi"}],
                num_workers=1,
                allowed_subproviders=["deepseek", "another"],
            )
            self.assertEqual(tester.openrouter_providers, ["deepseek", "another"])


if __name__ == "__main__":
    unittest.main()
