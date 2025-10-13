#!/usr/bin/env python3
"""
Streaming transport tests: interface validation and aggregation behavior.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_client import get_provider


class TestStreamingTransport(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict('os.environ', {
            'OPENAI_API_KEY': 'mock-openai-key',
            'OPENROUTER_API_KEY': 'mock-openrouter-key',
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch('requests.post')
    def test_openai_rejects_stream_flag_without_transport(self, mock_post):
        provider = get_provider("openai")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="gpt-4o-2024-08-06",
            stream=True,  # invalid in this client without transport='stream'
        )
        self.assertFalse(res.success)
        self.assertEqual(res.error_info.get('type'), 'invalid_option')
        self.assertIn('transport', res.error_info.get('message', ''))
        mock_post.assert_not_called()

    @patch('requests.post')
    def test_openrouter_rejects_stream_flag_without_transport(self, mock_post):
        provider = get_provider("openrouter")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="openai/gpt-4o-2024-08-06",
            stream=True,  # invalid in this client without transport='stream'
        )
        self.assertFalse(res.success)
        self.assertEqual(res.error_info.get('type'), 'invalid_option')
        self.assertIn('transport', res.error_info.get('message', ''))
        mock_post.assert_not_called()

    @patch('urllib3.PoolManager.request')
    def test_openai_streaming_aggregates_chunks(self, mock_request):
        # Mock a streaming SSE response via urllib3
        class FakeU3Resp:
            status = 200
            def stream(self, amt=65536, decode_content=True):
                lines = [
                    b'data: {"id":"s1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-2024-08-06","choices":[{"delta":{"content":"Hello "},"index":0}]}\n',
                    b'data: {"id":"s1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-2024-08-06","choices":[{"delta":{"content":"world"},"index":0,"finish_reason":"stop"}]}\n',
                    b'data: [DONE]\n',
                ]
                for b in lines:
                    yield b
            def close(self):
                return None
        mock_request.return_value = FakeU3Resp()

        provider = get_provider("openai")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="gpt-4o-2024-08-06",
            transport='stream',
        )

        self.assertTrue(res.success)
        self.assertEqual(res.standardized_response.get('content'), 'Hello world')

    @patch('urllib3.PoolManager.request')
    def test_openrouter_streaming_aggregates_chunks(self, mock_request):
        class FakeU3Resp:
            status = 200
            def stream(self, amt=65536, decode_content=True):
                lines = [
                    b'data: {"id":"s2","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"delta":{"content":"The ","role":"assistant"},"index":0}]}\n',
                    b'data: {"id":"s2","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"delta":{"content":"answer"},"index":0}]}\n',
                    b'data: {"id":"s2","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n',
                    b'data: [DONE]\n',
                ]
                for b in lines:
                    yield b
            def close(self):
                return None
        mock_request.return_value = FakeU3Resp()

        provider = get_provider("openrouter")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="openai/gpt-4o-mini",
            transport='stream',
        )

        self.assertTrue(res.success)
        self.assertEqual(res.standardized_response.get('content'), 'The answer')

    @patch('urllib3.PoolManager.request')
    def test_openai_streaming_error_event_is_failure(self, mock_request):
        class FakeU3Resp:
            status = 200
            def stream(self, amt=65536, decode_content=True):
                lines = [
                    b'data: {"error": {"message": "upstream failure"}}\n',
                    b'data: [DONE]\n',
                ]
                for b in lines:
                    yield b
            def close(self):
                return None
        mock_request.return_value = FakeU3Resp()

        provider = get_provider("openai")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="gpt-4o-2024-08-06",
            transport='stream',
        )

        self.assertFalse(res.success)
        self.assertEqual(res.error_info.get('type'), 'api_error')
        self.assertIn('upstream', res.error_info.get('message', ''))

    @patch('urllib3.PoolManager.request')
    def test_openai_streaming_empty_content_returns_error(self, mock_request):
        # Stream finishes without emitting any content tokens
        class FakeU3Resp:
            status = 200
            def stream(self, amt=65536, decode_content=True):
                lines = [
                    b'data: {"id":"s3","object":"chat.completion.chunk","created":1,"model":"gpt-4o-2024-08-06","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}\n',
                    b'data: [DONE]\n',
                ]
                for b in lines:
                    yield b
            def close(self):
                return None
        mock_request.return_value = FakeU3Resp()

        provider = get_provider("openai")
        res = provider.make_chat_completion_request(
            messages=[{"role": "user", "content": "Hi"}],
            model_id="gpt-4o-2024-08-06",
            transport='stream',
        )

        self.assertFalse(res.success)
        self.assertEqual(res.error_info.get('type'), 'content_filter')
        self.assertIn('no content', res.error_info.get('message', '').lower())


if __name__ == '__main__':
    unittest.main()
