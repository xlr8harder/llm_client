#!/usr/bin/env python3
"""
Tests for the LLM client retry mechanism.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import sys
import time

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import retry_request, LLMResponse
from llm_client.base import LLMProvider


class MockProvider(LLMProvider):
    """Mock provider for testing the retry mechanism"""
    
    def __init__(self, responses=None):
        """
        Initialize with a list of responses to return in sequence.
        
        Args:
            responses: List of LLMResponse objects to return
        """
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
        
    def _get_api_key_env_var(self):
        return 'MOCK_API_KEY'
        
    def make_chat_completion_request(self, messages, model_id, context=None, **options):
        """Return the next response in the sequence"""
        self.call_count += 1
        if self.call_count <= len(self.responses):
            return self.responses[self.call_count - 1]
        return LLMResponse(
            success=True,
            standardized_response={"content": "Default response"},
            context=context
        )


class TestRetryMechanism(unittest.TestCase):
    """Tests for the retry mechanism"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock time.sleep to avoid waiting during tests
        self.sleep_patcher = patch('time.sleep')
        self.mock_sleep = self.sleep_patcher.start()
    
    def tearDown(self):
        """Clean up test environment"""
        self.sleep_patcher.stop()
    
    def test_retry_success_first_try(self):
        """Test scenario where request succeeds on first try"""
        # Create a mock provider that returns success immediately
        successful_response = LLMResponse(
            success=True,
            standardized_response={"content": "Success on first try"},
            context={"test": "first_try"}
        )
        provider = MockProvider(responses=[successful_response])
        
        # Make the request
        result = retry_request(
            provider=provider,
            messages=[{"role": "user", "content": "Test"}],
            model_id="test-model",
            max_retries=3,
            context={"test": "first_try"}
        )
        
        # Verify results
        self.assertTrue(result.success)
        self.assertEqual(result.standardized_response["content"], "Success on first try")
        self.assertEqual(provider.call_count, 1)  # Should only be called once
        self.mock_sleep.assert_not_called()  # Sleep should not be called
    
    def test_retry_success_after_retries(self):
        """Test scenario where request succeeds after several retries"""
        # Create responses - two failures followed by success
        responses = [
            LLMResponse(
                success=False,
                error_info={"type": "network_error", "message": "Connection error"},
                is_retryable=True,
                context={"test": "multiple_tries"}
            ),
            LLMResponse(
                success=False,
                error_info={"type": "timeout", "message": "Request timed out"},
                is_retryable=True,
                context={"test": "multiple_tries"}
            ),
            LLMResponse(
                success=True,
                standardized_response={"content": "Success after retries"},
                context={"test": "multiple_tries"}
            )
        ]
        provider = MockProvider(responses=responses)
        
        # Make the request
        result = retry_request(
            provider=provider,
            messages=[{"role": "user", "content": "Test"}],
            model_id="test-model",
            max_retries=3,
            initial_delay=1,
            context={"test": "multiple_tries"}
        )
        
        # Verify results
        self.assertTrue(result.success)
        self.assertEqual(result.standardized_response["content"], "Success after retries")
        self.assertEqual(provider.call_count, 3)  # Should be called 3 times
        self.assertEqual(self.mock_sleep.call_count, 2)  # Sleep should be called twice
    
    def test_retry_permanent_failure(self):
        """Test scenario with a permanent (non-retryable) failure"""
        # Create a permanent failure response
        permanent_failure = LLMResponse(
            success=False,
            error_info={"type": "api_error", "message": "Invalid API key"},
            is_retryable=False,
            context={"test": "permanent_failure"}
        )
        provider = MockProvider(responses=[permanent_failure])
        
        # Make the request
        result = retry_request(
            provider=provider,
            messages=[{"role": "user", "content": "Test"}],
            model_id="test-model",
            max_retries=3,
            context={"test": "permanent_failure"}
        )
        
        # Verify results
        self.assertFalse(result.success)
        self.assertEqual(result.error_info["message"], "Invalid API key")
        self.assertEqual(provider.call_count, 1)  # Should only be called once
        self.mock_sleep.assert_not_called()  # Sleep should not be called
    
    def test_retry_exhausted_retries(self):
        """Test scenario where retries are exhausted without success"""
        # Create a series of retryable failures
        responses = [
            LLMResponse(
                success=False,
                error_info={"type": "network_error", "message": "Connection error 1"},
                is_retryable=True,
                context={"test": "exhausted"}
            ),
            LLMResponse(
                success=False,
                error_info={"type": "network_error", "message": "Connection error 2"},
                is_retryable=True,
                context={"test": "exhausted"}
            ),
            LLMResponse(
                success=False,
                error_info={"type": "network_error", "message": "Connection error 3"},
                is_retryable=True,
                context={"test": "exhausted"}
            )
        ]
        provider = MockProvider(responses=responses)
        
        # Make the request with only 2 retries (3 total attempts)
        result = retry_request(
            provider=provider,
            messages=[{"role": "user", "content": "Test"}],
            model_id="test-model",
            max_retries=2,  # Only 2 retries (3 total attempts)
            initial_delay=1,
            context={"test": "exhausted"}
        )
        
        # Verify results
        self.assertFalse(result.success)
        self.assertEqual(result.error_info["message"], "Connection error 3")
        self.assertFalse(result.is_retryable)  # Should be marked as no longer retryable
        self.assertTrue(result.error_info.get("max_retries_exceeded", False))
        self.assertEqual(provider.call_count, 3)  # Should be called 3 times
        self.assertEqual(self.mock_sleep.call_count, 2)  # Sleep should be called twice
    
    def test_retry_backoff_calculation(self):
        """Test that backoff delay is calculated correctly"""
        # Create retryable failures
        responses = [
            LLMResponse(success=False, error_info={"message": "Error 1"}, is_retryable=True),
            LLMResponse(success=False, error_info={"message": "Error 2"}, is_retryable=True),
            LLMResponse(success=True, standardized_response={"content": "Success"})
        ]
        provider = MockProvider(responses=responses)
        
        # Make the request
        retry_request(
            provider=provider,
            messages=[{"role": "user", "content": "Test"}],
            model_id="test-model",
            max_retries=5,
            initial_delay=2,
            backoff_factor=3,  # Use a distinctive backoff factor
            jitter=0  # Disable jitter for deterministic testing
        )
        
        # Verify sleep was called with correct delays
        self.assertEqual(self.mock_sleep.call_count, 2)
        calls = self.mock_sleep.call_args_list
        
        # First retry should use initial_delay (2)
        self.assertAlmostEqual(calls[0][0][0], 2, places=1)
        
        # Second retry should use initial_delay * backoff_factor (2 * 3 = 6)
        self.assertAlmostEqual(calls[1][0][0], 6, places=1)


if __name__ == "__main__":
    unittest.main()
