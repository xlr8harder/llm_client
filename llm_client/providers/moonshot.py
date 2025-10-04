"""
Moonshot provider implementation for LLM client.
"""
import json
import requests

from ..base import LLMProvider, LLMResponse

MOONSHOT_API_BASE = "https://api.moonshot.ai/v1"


class MoonshotProvider(LLMProvider):
    """Provider implementation for the Moonshot API (OpenAI-compatible)"""

    def _get_api_key_env_var(self):
        return 'MOONSHOT_API_KEY'

    def make_chat_completion_request(self, messages, model_id, context=None, **options):
        """Make a request to the Moonshot API"""
        try:
            url = f"{MOONSHOT_API_BASE}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.get_api_key()}",
                "Content-Type": "application/json"
            }

            timeout = options.pop('timeout', 60)

            data = {
                "model": model_id,
                "messages": messages,
                "max_tokens": options.pop('max_tokens', 4096)
            }

            data.update(options)

            response = requests.post(
                url=url,
                headers=headers,
                data=json.dumps(data),
                timeout=timeout
            )

            if response.status_code != 200:
                return self._handle_error_response(response, context)

            raw_response = response.json()
            standardized_response = self._standardize_response(raw_response)

            if self._has_content_filter_error(raw_response):
                error_info = self._extract_content_filter_error(raw_response)
                return LLMResponse(
                    success=False,
                    error_info=error_info,
                    raw_provider_response=raw_response,
                    is_retryable=False,
                    context=context
                )

            return LLMResponse(
                success=True,
                standardized_response=standardized_response,
                raw_provider_response=raw_response,
                is_retryable=False,
                context=context
            )

        except requests.exceptions.Timeout as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "timeout",
                    "message": f"Request timed out after {timeout} seconds: {str(e)}",
                    "exception": str(e)
                },
                is_retryable=True,
                context=context
            )
        except requests.exceptions.RequestException as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "network_error",
                    "message": str(e),
                    "exception": str(e),
                    "status_code": e.response.status_code if hasattr(e, 'response') and e.response else None
                },
                is_retryable=True,
                context=context
            )
        except Exception as e:
            return LLMResponse(
                success=False,
                error_info={
                    "type": "unexpected_error",
                    "message": str(e),
                    "exception": str(e)
                },
                is_retryable=False,
                context=context
            )

    def _handle_error_response(self, response, context):
        """Process Moonshot API error responses"""
        status_code = response.status_code
        is_retryable = status_code in [408, 425, 429, 500, 502, 503, 504]

        try:
            error_json = response.json()
        except Exception:
            error_json = None

        error_message = self._extract_error_message(error_json, response)

        error_info = {
            "type": "api_error",
            "status_code": status_code,
            "message": error_message,
            "raw_response": response.text
        }

        return LLMResponse(
            success=False,
            error_info=error_info,
            raw_provider_response=error_json,
            is_retryable=is_retryable,
            context=context
        )

    def _extract_error_message(self, error_json, response):
        """Extract error message from Moonshot responses"""
        if error_json and 'error' in error_json:
            error_obj = error_json['error']
            if isinstance(error_obj, dict) and 'message' in error_obj:
                return error_obj['message']
            return str(error_obj)
        return f"Error (HTTP {response.status_code}): {response.text[:200]}"

    def _has_content_filter_error(self, response):
        """Detect content filter errors"""
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'content_filter' or 'error' in choice:
                return True
        return False

    def _extract_content_filter_error(self, response):
        """Extract content filter error details"""
        choice = response['choices'][0]
        if 'error' in choice:
            error_obj = choice['error']
            message = error_obj.get('message', 'Content filtered')
        else:
            message = "Response stopped due to content filter"

        return {
            "type": "content_filter",
            "message": message
        }

    def _standardize_response(self, provider_response):
        """Convert Moonshot response to standardized format"""
        standardized = {
            "id": provider_response.get("id"),
            "created": provider_response.get("created"),
            "model": provider_response.get("model"),
            "provider": "moonshot",
            "content": None,
            "usage": provider_response.get("usage", {})
        }

        if "choices" in provider_response and provider_response["choices"]:
            choice = provider_response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                standardized["content"] = choice["message"]["content"]
            standardized["finish_reason"] = choice.get("finish_reason")

        return standardized
