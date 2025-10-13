#!/usr/bin/env python3
"""
Tests for the LLM client library providers.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import sys
import json
import urllib3

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import get_provider, LLMResponse
from llm_client.providers import (
    OpenAIProvider,
    OpenRouterProvider,
    GoogleProvider,
    TNGTechProvider,
    XAIProvider,
    MoonshotProvider,
)
from llm_client.providers.openai_style import OpenAIStyleProvider


class TestProviders(unittest.TestCase):
    """Tests for the provider implementations"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'OPENAI_API_KEY': 'mock-openai-key',
            'OPENROUTER_API_KEY': 'mock-openrouter-key',
            'FIREWORKS_API_KEY': 'mock-fireworks-key',
            'CHUTES_API_TOKEN': 'mock-chutes-token',
            'GEMINI_API_KEY': 'mock-gemini-key',
            'TNGTECH_API_KEY': 'mock-tngtech-key',
            'XAI_API_KEY': 'mock-xai-key',
            'MOONSHOT_API_KEY': 'mock-moonshot-key',
        })
        self.env_patcher.start()
        
    def tearDown(self):
        """Clean up test environment"""
        self.env_patcher.stop()
        
    def test_provider_factory(self):
        """Test the provider factory function"""
        # Test all providers
        provider_names = [
            "openai",
            "openrouter",
            "fireworks",
            "chutes",
            "google",
            "tngtech",
            "xai",
            "moonshot",
        ]
        for name in provider_names:
            provider = get_provider(name)
            self.assertIsNotNone(provider)
            
        # Test case insensitivity
        provider = get_provider("OpenAI")
        self.assertIsInstance(provider, OpenAIProvider)
        provider = get_provider("Moonshot")
        self.assertIsInstance(provider, MoonshotProvider)
        
        # Test invalid provider
        with self.assertRaises(ValueError):
            get_provider("invalid_provider")
            
    def test_api_key_retrieval(self):
        """Test API key retrieval"""
        # Test each provider gets its key correctly
        openai_provider = get_provider("openai")
        self.assertEqual(openai_provider.get_api_key(), "mock-openai-key")

        moonshot_provider = get_provider("moonshot")
        self.assertEqual(moonshot_provider.get_api_key(), "mock-moonshot-key")

        # Test missing API key with a fresh provider instance
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            # Create a fresh provider instance to avoid cached keys
            fresh_provider = OpenAIProvider()
            with self.assertRaises(ValueError):
                fresh_provider.get_api_key()

    @patch('urllib3.PoolManager.request')
    def test_xai_successful_request(self, mock_request):
        """Test successful X.AI request (OpenAI-compatible)"""
        # Mock the response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "id": "chatcmpl-xai-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "grok-2-latest",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from X.AI"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10
            }
        })
        mock_request.return_value = mock_response

        provider = get_provider("xai")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="grok-2-latest"
        )

        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "Hello from X.AI")
        self.assertEqual(response.standardized_response["model"], "grok-2-latest")
        self.assertEqual(response.standardized_response["provider"], "xai")

    @patch('urllib3.PoolManager.request')
    def test_moonshot_successful_request(self, mock_request):
        """Test successful Moonshot request (OpenAI-compatible)"""
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "id": "chatcmpl-moonshot-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "moonshot-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from Moonshot"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10
            }
        })
        mock_request.return_value = mock_response

        provider = get_provider("moonshot")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="moonshot-v1"
        )

        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "Hello from Moonshot")
        self.assertEqual(response.standardized_response["model"], "moonshot-v1")
        self.assertEqual(response.standardized_response["provider"], "moonshot")

    @patch('urllib3.PoolManager.request')
    def test_tngtech_successful_request(self, mock_request):
        """Test successful TNGTech request"""
        # Mock the response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
        mock_request.return_value = mock_response
        
        # Make the request
        provider = get_provider("tngtech")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="gpt-4o-2024-08-06"
        )
        
        # Verify the response
        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "This is a test response.")
        self.assertEqual(response.standardized_response["model"], "gpt-4o-2024-08-06")
        self.assertEqual(response.standardized_response["provider"], "tngtech")
                
    @patch('urllib3.PoolManager.request')
    def test_openai_successful_request(self, mock_request):
        """Test successful OpenAI request"""
        # Mock the response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
        mock_request.return_value = mock_response
        
        # Make the request
        provider = get_provider("openai")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="gpt-4o-2024-08-06"
        )
        
        # Verify the response
        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "This is a test response.")
        self.assertEqual(response.standardized_response["model"], "gpt-4o-2024-08-06")
        self.assertEqual(response.standardized_response["provider"], "openai")
        
    @patch('urllib3.PoolManager.request')
    def test_openai_error_request(self, mock_request):
        """Test OpenAI request with error"""
        # Mock the error response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(400, {
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        })
        mock_request.return_value = mock_response
        
        # Make the request
        provider = get_provider("openai")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="gpt-4o-2024-08-06"
        )
        
        # Verify the response
        self.assertFalse(response.success)
        self.assertFalse(response.is_retryable)  # Client errors are not retryable
        self.assertEqual(response.error_info["type"], "api_error")
        self.assertEqual(response.error_info["status_code"], 400)

    @patch('urllib3.PoolManager.request')
    def test_openai_null_content_returns_error(self, mock_request):
        """If provider returns null content, treat as non-retryable error."""
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        # Simulate a 200 response with choices but message.content is null
        mock_response = U3Resp(200, {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2
            }
        })
        mock_request.return_value = mock_response

        provider = get_provider("openai")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="gpt-4o-2024-08-06"
        )

        self.assertFalse(res.success)
        self.assertEqual(res.error_info.get("type"), "content_filter")
        self.assertIn("no content", res.error_info.get("message", "").lower())
        
    @patch('urllib3.PoolManager.request')
    def test_openai_network_error(self, mock_request):
        """Test OpenAI request with network error"""
        # Mock a network error
        mock_request.side_effect = Exception("Connection refused")
        
        # Make the request
        provider = get_provider("openai")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="gpt-4o-2024-08-06"
        )
        
        # Verify the response
        self.assertFalse(response.success)
        self.assertTrue(response.is_retryable)  # Network errors are retryable
        self.assertEqual(response.error_info["type"], "network_error")
        
    @patch('urllib3.PoolManager.request')
    def test_openai_timeout_error(self, mock_request):
        """Test OpenAI request with timeout"""
        # Mock a timeout
        from urllib3 import exceptions as u3e
        mock_request.side_effect = u3e.ReadTimeoutError(None, 'https://api.openai.com/v1/chat/completions', 'Request timed out')
        
        # Make the request
        provider = get_provider("openai")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="gpt-4o-2024-08-06"
        )
        
        # Verify the response
        self.assertFalse(response.success)
        self.assertTrue(response.is_retryable)  # Timeouts are retryable
        self.assertEqual(response.error_info["type"], "timeout")

    @patch('urllib3.PoolManager.request')
    def test_openai_http200_top_level_error_is_failure(self, mock_request):
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {"error": {"message": "Bad request in body", "code": 123}})
        mock_request.return_value = mock_response

        provider = get_provider("openai")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="gpt-4o-2024-08-06"
        )

        self.assertFalse(res.success)
        self.assertEqual(res.error_info.get("type"), "api_error")
        self.assertIn("Bad request", res.error_info.get("message", ""))

    @patch('urllib3.PoolManager.request')
    def test_openai_style_customizable_subclass(self, mock_request):
        """Ensure subclasses can override headers and defaults."""

        class DemoProvider(OpenAIStyleProvider):
            api_base = "https://example.test/v1"
            api_key_env_var = "DEMO_API_KEY"
            provider_name = "demo"
            default_max_tokens = 77

            def _build_request_headers(self):
                headers = super()._build_request_headers()
                headers["X-Custom"] = "demo"
                return headers

        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "id": "chatcmpl-demo-1",
            "object": "chat.completion",
            "created": 123,
            "model": "demo-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "demo reply"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        mock_request.return_value = mock_response

        with patch.dict('os.environ', {"DEMO_API_KEY": "demo-key"}):
            provider = DemoProvider()
            response = provider.make_chat_completion_request(
                messages=[{"role": "user", "content": "hello"}],
                model_id="demo-model",
            )

        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "demo reply")
        self.assertEqual(response.standardized_response["provider"], "demo")

        args, kwargs = mock_request.call_args
        # args[0] is method, args[1] is URL
        self.assertEqual(args[0], 'POST')
        self.assertEqual(args[1], "https://example.test/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["X-Custom"], "demo")

        payload = json.loads(kwargs["body"].decode('utf-8'))
        self.assertEqual(payload["max_tokens"], 77)
        self.assertEqual(payload["model"], "demo-model")
        
    @patch('urllib3.PoolManager.request')
    def test_openrouter_provider_filtering(self, mock_request):
        """Test OpenRouter provider filtering"""
        # Mock the response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "openai/gpt-4o-2024-08-06",
            "_provider_used": "openai",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        })
        mock_request.return_value = mock_response
        
        # Make the request with allow_list
        provider = get_provider("openrouter")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="openai/gpt-4o-2024-08-06",
            allow_list=["openai"]
        )
        
        # Check that the request was made with the right parameters
        args, kwargs = mock_request.call_args
        data = json.loads(kwargs["body"].decode('utf-8'))
        self.assertIn("provider", data)
        self.assertIn("order", data["provider"])
        self.assertEqual(data["provider"]["order"], ["openai"])
        self.assertFalse(data["provider"]["allow_fallbacks"])
        
        # Make the request with ignore_list
        provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="openai/gpt-4o-2024-08-06",
            ignore_list=["deepinfra"]
        )
        
        # Check that the request was made with the right parameters
        args, kwargs = mock_request.call_args
        data = json.loads(kwargs["body"].decode('utf-8'))
        self.assertIn("provider", data)
        self.assertIn("ignore", data["provider"])
        self.assertEqual(data["provider"]["ignore"], ["deepinfra"])
        
    @patch('urllib3.PoolManager.request')
    def test_google_request(self, mock_request):
        """Test Google provider request"""
        # Mock the response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "This is a test response from Google."
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "safetyRatings": []
                }
            ],
            "promptFeedback": {
                "safetyRatings": []
            },
            "modelVersion": "gemini-1.5-pro",
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30
            }
        })
        mock_request.return_value = mock_response
        
        # Make the request
        provider = get_provider("google")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="gemini-1.5-pro"
        )
        
        # Verify the response
        self.assertTrue(response.success)
        self.assertEqual(response.standardized_response["content"], "This is a test response from Google.")
        self.assertEqual(response.standardized_response["model"], "gemini-1.5-pro")
        self.assertEqual(response.standardized_response["provider"], "google")
        self.assertEqual(response.standardized_response["finish_reason"], "stop")  # Should be standardized
        
    @patch('urllib3.PoolManager.request')
    def test_google_content_filter(self, mock_request):
        """Test Google content filter handling"""
        # Mock the content filter response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "Partial response before filtering."
                            }
                        ],
                        "role": "model"
                    },
                    "finishReason": "SAFETY",
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "HIGH"
                        }
                    ]
                }
            ],
            "promptFeedback": {
                "safetyRatings": []
            }
        })
        mock_request.return_value = mock_response
        
        # Make the request
        provider = get_provider("google")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Generate harmful content"}],
            model_id="gemini-1.5-pro"
        )
        
        # Verify the response
        self.assertFalse(response.success)
        self.assertFalse(response.is_retryable)  # Content filter errors are not retryable
        self.assertEqual(response.error_info["type"], "content_filter")
        self.assertIn("SAFETY", response.error_info["message"])
        
    @patch('urllib3.PoolManager.request')
    def test_google_prompt_blocked(self, mock_request):
        """Test Google prompt blocked handling"""
        # Mock the prompt blocked response
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "HIGH"
                    }
                ]
            }
        })
        mock_request.return_value = mock_response
        
        # Make the request
        provider = get_provider("google")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Generate harmful content"}],
            model_id="gemini-1.5-pro"
        )
        
        # Verify the response
        self.assertFalse(response.success)
        self.assertFalse(response.is_retryable)  # Prompt blocks are not retryable
        self.assertEqual(response.error_info["type"], "content_filter")
        self.assertIn("Prompt blocked", response.error_info["message"])
        
    def test_openrouter_provider_methods(self):
        """Test OpenRouter provider-specific methods"""
        provider = get_provider("openrouter")
        
        # Test get_available_providers with mocked response
        with patch('urllib3.PoolManager.request') as mock_request:
            class U3Resp:
                def __init__(self, status, data):
                    self.status = status
                    self.data = json.dumps(data).encode('utf-8')

            mock_response = U3Resp(200, {
                "data": [
                    {"provider_name": "openai"},
                    {"provider_name": "anthropic"},
                    {"provider_name": "openai"},  # Duplicate to test deduplication
                ]
            })
            mock_request.return_value = mock_response
            
            providers = provider.get_available_providers("openai/gpt-4o-2024-08-06")
            self.assertEqual(sorted(providers), ["anthropic", "openai"])
            
        # Test is_model_available with mocked response
        with patch.object(provider, 'get_available_providers') as mock_get_providers:
            # Test available
            mock_get_providers.return_value = ["openai", "anthropic"]
            self.assertTrue(provider.is_model_available("openai/gpt-4o-2024-08-06"))
            
            # Test not available
            mock_get_providers.return_value = []
            self.assertFalse(provider.is_model_available("openai/gpt-4o-2024-08-06"))

    @patch('urllib3.PoolManager.request')
    def test_openrouter_http200_top_level_error_is_failure(self, mock_request):
        class U3Resp:
            def __init__(self, status, data):
                self.status = status
                self.data = json.dumps(data).encode('utf-8')

        mock_response = U3Resp(200, {"error": {"message": "No endpoints found"}})
        mock_request.return_value = mock_response

        provider = get_provider("openrouter")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="openai/gpt-4o-mini"
        )

        self.assertFalse(res.success)
        self.assertEqual(res.error_info.get("type"), "api_error")
        self.assertIn("endpoints", res.error_info.get("message", ""))


class TestLLMResponse(unittest.TestCase):
    """Tests for the LLMResponse class"""
    
    def test_llm_response_creation(self):
        """Test creating an LLMResponse object"""
        # Test successful response
        successful_response = LLMResponse(
            success=True,
            standardized_response={"content": "Test response"},
            raw_provider_response={"choices": [{"message": {"content": "Test response"}}]},
            context={"test_id": "123"}
        )
        
        self.assertTrue(successful_response.success)
        self.assertEqual(successful_response.standardized_response["content"], "Test response")
        self.assertEqual(successful_response.context["test_id"], "123")
        
        # Test error response
        error_response = LLMResponse(
            success=False,
            error_info={"type": "api_error", "message": "Invalid API key"},
            raw_provider_response={"error": {"message": "Invalid API key"}},
            is_retryable=False,
            context={"test_id": "456"}
        )
        
        self.assertFalse(error_response.success)
        self.assertEqual(error_response.error_info["type"], "api_error")
        self.assertEqual(error_response.error_info["message"], "Invalid API key")
        self.assertFalse(error_response.is_retryable)
        self.assertEqual(error_response.context["test_id"], "456")


if __name__ == "__main__":
    unittest.main()
