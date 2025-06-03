#!/usr/bin/env python3
"""
Tests for the LLM client library providers.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import sys
import json
import requests

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import get_provider, LLMResponse
from llm_client.providers import OpenAIProvider, OpenRouterProvider, GoogleProvider, TNGTechProvider


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
        })
        self.env_patcher.start()
        
    def tearDown(self):
        """Clean up test environment"""
        self.env_patcher.stop()
        
    def test_provider_factory(self):
        """Test the provider factory function"""
        # Test all providers
        provider_names = ["openai", "openrouter", "fireworks", "chutes", "google", "tngtech"]
        for name in provider_names:
            provider = get_provider(name)
            self.assertIsNotNone(provider)
            
        # Test case insensitivity
        provider = get_provider("OpenAI")
        self.assertIsInstance(provider, OpenAIProvider)
        
        # Test invalid provider
        with self.assertRaises(ValueError):
            get_provider("invalid_provider")
            
    def test_api_key_retrieval(self):
        """Test API key retrieval"""
        # Test each provider gets its key correctly
        openai_provider = get_provider("openai")
        self.assertEqual(openai_provider.get_api_key(), "mock-openai-key")
        
        # Test missing API key with a fresh provider instance
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            # Create a fresh provider instance to avoid cached keys
            fresh_provider = OpenAIProvider()
            with self.assertRaises(ValueError):
                fresh_provider.get_api_key()

    @patch('requests.post')
    def test_tngtech_successful_request(self, mock_post):
        """Test successful TNGTech request"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        }
        mock_post.return_value = mock_response
        
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
        self.assertEqual(response.standardized_response["provider"], "openai")
                
    @patch('requests.post')
    def test_openai_successful_request(self, mock_post):
        """Test successful OpenAI request"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        }
        mock_post.return_value = mock_response
        
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
        
    @patch('requests.post')
    def test_openai_error_request(self, mock_post):
        """Test OpenAI request with error"""
        # Mock the error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        }
        mock_post.return_value = mock_response
        
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
        
    @patch('requests.post')
    def test_openai_network_error(self, mock_post):
        """Test OpenAI request with network error"""
        # Mock a network error
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
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
        
    @patch('requests.post')
    def test_openai_timeout_error(self, mock_post):
        """Test OpenAI request with timeout"""
        # Mock a timeout
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
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
        
    @patch('requests.post')
    def test_openrouter_provider_filtering(self, mock_post):
        """Test OpenRouter provider filtering"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        }
        mock_post.return_value = mock_response
        
        # Make the request with allow_list
        provider = get_provider("openrouter")
        response = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Test message"}],
            model_id="openai/gpt-4o-2024-08-06",
            allow_list=["openai"]
        )
        
        # Check that the request was made with the right parameters
        args, kwargs = mock_post.call_args
        data = json.loads(kwargs["data"])
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
        args, kwargs = mock_post.call_args
        data = json.loads(kwargs["data"])
        self.assertIn("provider", data)
        self.assertIn("ignore", data["provider"])
        self.assertEqual(data["provider"]["ignore"], ["deepinfra"])
        
    @patch('requests.post')
    def test_google_request(self, mock_post):
        """Test Google provider request"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        }
        mock_post.return_value = mock_response
        
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
        
    @patch('requests.post')
    def test_google_content_filter(self, mock_post):
        """Test Google content filter handling"""
        # Mock the content filter response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        }
        mock_post.return_value = mock_response
        
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
        
    @patch('requests.post')
    def test_google_prompt_blocked(self, mock_post):
        """Test Google prompt blocked handling"""
        # Mock the prompt blocked response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "HIGH"
                    }
                ]
            }
        }
        mock_post.return_value = mock_response
        
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
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"provider_name": "openai"},
                    {"provider_name": "anthropic"},
                    {"provider_name": "openai"},  # Duplicate to test deduplication
                ]
            }
            mock_get.return_value = mock_response
            
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
            
            # Test error
            mock_get_providers.return_value = None
            self.assertFalse(provider.is_model_available("openai/gpt-4o-2024-08-06"))


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
