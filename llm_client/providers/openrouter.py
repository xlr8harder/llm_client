"""
OpenRouter provider implementation for LLM client.
"""
import json
import os
import requests

from ..base import LLMProvider, LLMResponse

# API Endpoint Constants
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

class OpenRouterProvider(LLMProvider):
    """Provider implementation for OpenRouter API"""
    
    def _get_api_key_env_var(self):
        return 'OPENROUTER_API_KEY'
    
    def make_chat_completion_request(self, messages, model_id, context=None, **options):
        """
        Make a request to the OpenRouter API
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            model_id: OpenRouter model identifier
            context: Optional context object to include in the response
            **options: Additional options (including allow_list and ignore_list for providers)
        
        Returns:
            LLMResponse object with standardized result
        """
        try:
            # Prepare request data
            url = f"{OPENROUTER_API_BASE}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.get_api_key()}", 
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_TITLE", "LLM Client Library")
            }
            
            # Extract timeout if provided, otherwise default to 60 seconds
            timeout = options.pop('timeout', 60)
            
            # Prepare the data payload
            data = {
                "model": model_id,
                "messages": messages,
                "max_tokens": options.pop('max_tokens', 4096)
            }
            
            # Handle provider routing options
            provider_routing = {}
            
            if 'allow_list' in options:
                provider_routing["order"] = options.pop('allow_list')
                provider_routing["allow_fallbacks"] = False
            elif 'ignore_list' in options:
                provider_routing["ignore"] = options.pop('ignore_list')
                
            if provider_routing:
                data["provider"] = provider_routing
            
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
        """Convert OpenRouter response to standardized format"""
        standardized = {
            "id": provider_response.get("id"),
            "created": provider_response.get("created"),
            "model": provider_response.get("model"),
            "provider": "openrouter",
            "content": None,
            "usage": provider_response.get("usage", {})
        }
        
        # Extract additional OpenRouter metadata
        if "_provider_used" in provider_response:
            standardized["sub_provider"] = provider_response["_provider_used"]
            
        # Extract content from the first choice
        if "choices" in provider_response and len(provider_response["choices"]) > 0:
            choice = provider_response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                standardized["content"] = choice["message"]["content"]
            standardized["finish_reason"] = choice.get("finish_reason")
        
        return standardized
    
    def get_available_providers(self, model_id):
        """
        Get list of providers available for the given model ID
        
        Args:
            model_id: OpenRouter model identifier
            
        Returns:
            List of provider names or None if the request fails
        """
        try:
            endpoints_url = f"{OPENROUTER_API_BASE}/models/{model_id}/endpoints"
            headers = {"Authorization": f"Bearer {self.get_api_key()}"}
            
            response = requests.get(endpoints_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check structure based on user example: data is a list
            if 'data' in data and isinstance(data['data'], list):
                providers = [ep.get('provider_name') for ep in data['data'] if ep.get('provider_name')]
                unique_providers = sorted(list(set(p for p in providers if p)))
                return unique_providers
                
            # Check alternative nested structure
            elif 'data' in data and 'endpoints' in data['data'] and isinstance(data['data']['endpoints'], list):
                providers = [ep.get('provider_name') for ep in data['data']['endpoints'] if ep.get('provider_name')]
                unique_providers = sorted(list(set(p for p in providers if p)))
                return unique_providers
                
            else:
                print(f"ERROR: Unexpected structure in OpenRouter model data for {model_id}")
                return None
                
        except Exception as e:
            print(f"ERROR: Failed to fetch OpenRouter providers for {model_id}: {str(e)}")
            return None
    
    def is_model_available(self, model_id):
        """
        Check if a model is currently available on OpenRouter
        
        Args:
            model_id: OpenRouter model identifier
            
        Returns:
            Boolean indicating if the model is available
        """
        providers = self.get_available_providers(model_id)
        return providers is not None and len(providers) > 0
