"""
Base classes for the LLM client library.
"""

from abc import ABC, abstractmethod
import os


class LLMResponse:
    """
    Standard response object for all LLM provider requests.

    Attributes:
        success (bool): Whether the request was successful
        standardized_response (dict): Standardized response format across all providers
        error_info (dict): Information about any error encountered
        raw_provider_response (dict): The original provider response
        request_format (str): Client request format that was used
        raw_response_format (str): Schema/shape of the raw provider response
        is_retryable (bool): Whether the error can be retried
        context (any): Optional context passed by the consumer
    """

    def __init__(
        self,
        success=False,
        standardized_response=None,
        error_info=None,
        raw_provider_response=None,
        request_format=None,
        raw_response_format=None,
        is_retryable=False,
        context=None,
    ):
        self.success = success
        self.standardized_response = standardized_response
        self.error_info = error_info
        self.raw_provider_response = raw_provider_response
        self.request_format = request_format
        self.raw_response_format = raw_response_format
        self.is_retryable = is_retryable
        self.context = context


class LLMProvider(ABC):
    """Base class for LLM API providers"""

    def __init__(self):
        self._api_key = None

    def get_api_key(self):
        """Get the API key, retrieving from environment if not already cached"""
        if self._api_key is None:
            env_var = self._get_api_key_env_var()
            self._api_key = os.getenv(env_var)
            if not self._api_key:
                raise ValueError(
                    f"Required API key environment variable '{env_var}' is not set"
                )
        return self._api_key

    @abstractmethod
    def _get_api_key_env_var(self):
        """Return the environment variable name for the API key"""
        pass

    @abstractmethod
    def make_chat_completion_request(self, messages, model_id, context=None, **options):
        """
        Make a standardized chat completion request

        Args:
            messages: List of message objects
            model_id: ID of the model to use
            context: Optional context object that will be passed back in the response
            **options: Additional provider-specific options

        Returns:
            LLMResponse object containing success/failure status and data
        """
        pass

    def make_request(
        self,
        messages,
        model_id,
        context=None,
        request_format="chat_completions",
        **options,
    ):
        """
        Make a request using the selected provider request format.

        Providers may override this to support multiple API shapes. The default
        implementation preserves the historic chat-completions behavior.
        """
        normalized_format = (request_format or "chat_completions").replace("-", "_")
        if normalized_format in {"chat_completions", "chat_completion"}:
            response = self.make_chat_completion_request(
                messages=messages,
                model_id=model_id,
                context=context,
                **options,
            )
            if isinstance(response, LLMResponse) and response.request_format is None:
                response.request_format = "chat_completions"
            return response

        return LLMResponse(
            success=False,
            error_info={
                "type": "invalid_option",
                "message": f"Unsupported request_format: {request_format}",
            },
            request_format=normalized_format,
            raw_response_format="llm_client.error",
            is_retryable=False,
            context=context,
        )

    def _standardize_response(self, provider_response):
        """
        Convert provider-specific response to standardized format.
        This method should be implemented by each provider.

        Args:
            provider_response: The raw response from the provider

        Returns:
            dict: Standardized response format
        """
        raise NotImplementedError(
            "_standardize_response must be implemented by provider classes"
        )
