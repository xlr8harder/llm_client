#!/usr/bin/env python3
"""
Tests for urllib3 exception mapping to retryable/non-retryable and error types.
"""
import os
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import get_provider
from urllib3 import exceptions as u3e


class TestErrorMapping(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', {
            'OPENAI_API_KEY': 'k',
            'OPENROUTER_API_KEY': 'k',
            'GEMINI_API_KEY': 'k',
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()

    @patch('urllib3.PoolManager.request')
    def test_timeout_is_retryable(self, mock_request):
        # Simulate a read timeout from urllib3
        mock_request.side_effect = u3e.ReadTimeoutError(None, 'https://example', 'Read timed out')
        p = get_provider('openai')
        res = p.make_chat_completion_request(
            messages=[{"role": "user", "content": "hi"}],
            model_id='gpt-4o-2024-08-06'
        )
        self.assertFalse(res.success)
        self.assertTrue(res.is_retryable)
        self.assertEqual(res.error_info.get('type'), 'timeout')

    @patch('urllib3.PoolManager.request')
    def test_ssl_is_non_retryable(self, mock_request):
        # Simulate SSL certificate error
        mock_request.side_effect = u3e.SSLError(None, 'https://example', 'CERTIFICATE_VERIFY_FAILED')
        p = get_provider('openai')
        res = p.make_chat_completion_request(
            messages=[{"role": "user", "content": "hi"}],
            model_id='gpt-4o-2024-08-06'
        )
        self.assertFalse(res.success)
        self.assertFalse(res.is_retryable)
        self.assertEqual(res.error_info.get('type'), 'network_error')

    @patch('urllib3.PoolManager.request')
    def test_location_parse_is_non_retryable(self, mock_request):
        # Malformed URL scenario
        mock_request.side_effect = u3e.LocationParseError('://bad-url')
        p = get_provider('openai')
        res = p.make_chat_completion_request(
            messages=[{"role": "user", "content": "hi"}],
            model_id='gpt-4o-2024-08-06'
        )
        self.assertFalse(res.success)
        self.assertFalse(res.is_retryable)
        self.assertEqual(res.error_info.get('type'), 'network_error')


if __name__ == '__main__':
    unittest.main()
