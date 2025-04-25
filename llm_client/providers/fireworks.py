"""
Fireworks AI provider implementation for LLM client.
"""
import json
import requests

from ..base import LLMProvider, LLMResponse

# API Endpoint Constants
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"

class FireworksProvider(LLMProvider):
    """Provider implementation for Fireworks AI API"""
    
    def _get_api_key_env_var(self):
        return 'FIREWORKS_API_KEY'
    
    def make_chat_completion_request(self, messages, model_id, context=None, **options):
        """
        Make a request to the Fireworks AI API
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            model_id: Fireworks model identifier
            context: Optional context object to include in the response
            **options: Additional options for the request
        
        Returns:
            LLMResponse object with standardized result
        """
        try:
            # Prepare request data
            url = f"{FIREWORKS_API_BASE}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.get_api_key()}", 
                "Content-Type": "application/json"
            }
            
            # Extract timeout if provided, otherwise default to 60 seconds
            timeout = options.pop('timeout', 60)
            
            # Prepare the data payload
            data = {
                "model": model_id,
                "messages": messages,
                "max_tokens": options.pop('max_tokens', 4096)
            }
            
            # Add any remaining options to the payload
            data.update(options)
            
            # Make the request
            response = requests.post(
                url=url, 
                headers=headers, 
                data=json.dumps(data), 
                timeout=timeout
            )
            
            # Handle non-200 responses
            if response.status_code != 200:
                return self._handle_error_response(response, context)
                
            # Process successful response
            raw_response = response.json()
            standardized_response = self._standardize_response(raw_response)
            
            # Check if the response contains a content filter error
            if self._has_content_filter_error(raw_response):
                error_info = self._extract_content_filter_error(raw_response)
                return LLMResponse(
                    success=False,
                    error_info=error_info,
                    raw_provider_response=raw_response,
                    is_retryable=False,  # Content filter errors are not retryable
                    context=context
                )
            
            return LLMResponse(
                success=True,
                standardized_response=standardized_response,
                raw_provider_response=raw_response,
                is_retryable=False,  # Not needed for success but included for consistency
                context=context
            )
            
        except requests.exceptions.Timeout as e:
            # Handle timeout errors
            return LLMResponse(
                success=False,
                error_info={
                    "type": "timeout",
                    "message": f"Request timed out after {timeout} seconds: {str(e)}",
                    "exception": str(e)
                },
                is_retryable=True,  # Timeouts are retryable
                context=context
            )
        except requests.exceptions.RequestException as e:
            # Handle network errors
            return LLMResponse(
                success=False,
                error_info={
                    "type": "network_error",
                    "message": str(e),
                    "exception": str(e),
                    "status_code": e.response.status_code if hasattr(e, 'response') and e.response else None
                },
                is_retryable=True,  # Network errors are usually retryable
                context=context
            )
        except Exception as e:
            # Handle unexpected errors
            return LLMResponse(
                success=False,
                error_info={
                    "type": "unexpected_error",
                    "message": str(e),
                    "exception": str(e)
                },
                is_retryable=False,  # Be conservative with unexpected errors
                context=context
            )
    
    def _handle_error_response(self, response, context):
        """Process error responses from the API"""
        status_code = response.status_code
        is_retryable = status_code in [408, 425, 429, 500, 502, 503, 504]
        
        try:
            error_json = response.json()
        except:
            error_json = None
            
        error_message = self._extract_error_message(error_json, response.text)
        
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
    
    def _extract_error_message(self, error_json, response_text):
        """Extract error message from response"""
        if error_json and 'error' in error_json:
            error_obj = error_json['error']
            if isinstance(error_obj, dict) and 'message' in error_obj:
                return error_obj['message']
            return str(error_obj)
        return f"Error: {response_text[:200]}"
    
    def _has_content_filter_error(self, response):
        """Check if the response contains a content filter error"""
        if 'choices' in response and len(response['choices']) > 0:
            choice = response['choices'][0]
            if choice.get('finish_reason') == 'content_filter' or 'error' in choice:
                return True
        return False
    
    def _extract_content_filter_error(self, response):
        """Extract content filter error from response"""
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
        """Convert Fireworks response to standardized format"""
        standardized = {
            "id": provider_response.get("id"),
            "created": provider_response.get("created"),
            "model": provider_response.get("model"),
            "provider": "fireworks",
            "content": None,
            "usage": provider_response.get("usage", {})
        }
        
        # Extract content from the first choice
        if "choices" in provider_response and len(provider_response["choices"]) > 0:
            choice = provider_response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                standardized["content"] = choice["message"]["content"]
            standardized["finish_reason"] = choice.get("finish_reason")
        
        return standardized
