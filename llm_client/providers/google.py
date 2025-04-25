"""
Google provider implementation for LLM client.
"""
import json
import requests

from ..base import LLMProvider, LLMResponse

# API Endpoint Constants
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

class GoogleProvider(LLMProvider):
    """Provider implementation for Google Gemini API"""
    
    def _get_api_key_env_var(self):
        return 'GEMINI_API_KEY'
    
    def make_chat_completion_request(self, messages, model_id, context=None, **options):
        """
        Make a request to the Google Gemini API
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            model_id: Google model identifier
            context: Optional context object to include in the response
            **options: Additional options for the request
        
        Returns:
            LLMResponse object with standardized result
        """
        try:
            # Get API key
            api_key = self.get_api_key()
            
            # Prepare request URL
            url = f"{GOOGLE_API_BASE}/models/{model_id}:generateContent?key={api_key}"
            
            # Extract timeout if provided, otherwise default to 60 seconds
            timeout = options.pop('timeout', 60)
            
            # Convert messages to Google's format
            google_messages = self._convert_messages_to_google_format(messages)
            
            # Prepare the data payload
            data = {
                "contents": google_messages
            }
            
            # Add safety settings with maximum permissiveness
            data["safetySettings"] = [ 
                {"category": c, "threshold": "BLOCK_NONE"} 
                for c in [
                    "HARM_CATEGORY_HARASSMENT", 
                    "HARM_CATEGORY_HATE_SPEECH", 
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                    "HARM_CATEGORY_DANGEROUS_CONTENT"
                ]
            ]
            
            # Add any remaining options to the payload
            for key, value in options.items():
                if key not in data:
                    data[key] = value
            
            # Make the request
            response = requests.post(
                url=url, 
                headers={"Content-Type": "application/json"},
                data=json.dumps(data), 
                timeout=timeout
            )
            
            # Handle non-200 responses
            if response.status_code != 200:
                return self._handle_error_response(response, context)
                
            # Process successful response
            raw_response = response.json()
            
            # Check for prompt feedback (indicates blocked input)
            if 'promptFeedback' in raw_response and raw_response['promptFeedback'].get('blockReason'):
                return self._handle_prompt_blocked(raw_response, context)
            
            # Google API can return content filter in different ways
            if self._has_content_filter_error(raw_response):
                error_info = self._extract_content_filter_error(raw_response)
                return LLMResponse(
                    success=False,
                    error_info=error_info,
                    raw_provider_response=raw_response,
                    is_retryable=False,  # Content filter errors are not retryable
                    context=context
                )
            
            standardized_response = self._standardize_response(raw_response)
            
            return LLMResponse(
                success=True,
                standardized_response=standardized_response,
                raw_provider_response=raw_response,
                is_retryable=False,
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
    
    def _convert_messages_to_google_format(self, messages):
        """
        Convert standard chat message format to Google's format
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            
        Returns:
            List of message objects in Google's format
        """
        # For Google, we combine all messages into a single contents array
        google_messages = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Only add non-empty messages
            if content:
                if role == 'assistant':
                    google_role = 'model'
                else:
                    google_role = 'user'
                
                google_messages.append({
                    "role": google_role,
                    "parts": [{"text": content}]
                })
        
        # For the simplest case, if we only have one message, format it differently
        if len(messages) == 1 and messages[0].get('role', 'user') == 'user':
            return [{"parts": [{"text": messages[0].get('content', '')}]}]
            
        return google_messages
    
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
    
    def _handle_prompt_blocked(self, response, context):
        """Handle when Google blocks the prompt"""
        block_reason = response['promptFeedback']['blockReason']
        safety_ratings = response['promptFeedback'].get('safetyRatings', [])
        
        error_info = {
            "type": "content_filter",
            "message": f"Prompt blocked due to: {block_reason}",
            "block_reason": block_reason,
            "safety_ratings": safety_ratings
        }
        
        return LLMResponse(
            success=False,
            error_info=error_info,
            raw_provider_response=response,
            is_retryable=False,  # Prompt blocking is not retryable
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
        # Check for safety issues in candidates
        if 'candidates' in response and len(response['candidates']) > 0:
            candidate = response['candidates'][0]
            finish_reason = candidate.get('finishReason')
            if finish_reason in ['SAFETY', 'RECITATION', 'OTHER']:
                return True
                
        # Check for top-level error
        if 'error' in response:
            return True
            
        return False
    
    def _extract_content_filter_error(self, response):
        """Extract content filter error from response"""
        # Check for candidate-level issues
        if 'candidates' in response and len(response['candidates']) > 0:
            candidate = response['candidates'][0]
            finish_reason = candidate.get('finishReason')
            safety_ratings = candidate.get('safetyRatings', [])
            
            if finish_reason:
                return {
                    "type": "content_filter",
                    "message": f"Response stopped due to: {finish_reason}",
                    "finish_reason": finish_reason,
                    "safety_ratings": safety_ratings
                }
                
        # Check for top-level error
        if 'error' in response:
            error_obj = response['error']
            message = error_obj.get('message', 'Unknown Google API error')
            error_code = error_obj.get('code')
            
            return {
                "type": "api_error",
                "message": message,
                "code": error_code
            }
            
        # Fallback
        return {
            "type": "content_filter",
            "message": "Unknown content filter issue"
        }
    
    def _standardize_response(self, provider_response):
        """Convert Google response to standardized format"""
        standardized = {
            "id": None,  # Google doesn't provide an ID like OpenAI
            "created": None,  # Google doesn't provide a timestamp
            "model": provider_response.get("modelVersion"),
            "provider": "google",
            "content": None,
            "usage": {}
        }
        
        # Extract usage metadata if available
        if 'usageMetadata' in provider_response:
            um = provider_response['usageMetadata']
            standardized['usage'] = {
                'prompt_tokens': um.get('promptTokenCount'),
                'completion_tokens': um.get('candidatesTokenCount'),
                'total_tokens': um.get('totalTokenCount')
            }
        
        # Extract content from the first candidate
        if 'candidates' in provider_response and len(provider_response['candidates']) > 0:
            candidate = provider_response['candidates'][0]
            finish_reason_raw = candidate.get('finishReason')
            
            # Map Google finish reasons to standard format
            reason_map = {
                "STOP": "stop", 
                "MAX_TOKENS": "length", 
                "SAFETY": "content_filter",
                "RECITATION": "content_filter", 
                "OTHER": "error", 
                "UNSPECIFIED": "error"
            }
            standardized['finish_reason'] = reason_map.get(finish_reason_raw, finish_reason_raw.lower() if finish_reason_raw else None)
            
            # Extract text content
            if ('content' in candidate and 'parts' in candidate['content'] and
                isinstance(candidate['content']['parts'], list) and len(candidate['content']['parts']) > 0):
                standardized['content'] = "".join([part.get('text', '') for part in candidate['content']['parts'] if 'text' in part])
        
        return standardized
