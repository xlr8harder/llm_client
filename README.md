# LLM Client Library

A clean, modular client library for interacting with various LLM providers (OpenAI, OpenRouter, Fireworks, Chutes, Google) with standardized responses and robust error handling.

## Features

- **Unified Interface**: Common API for all supported LLM providers
- **Standardized Responses**: Consistent response format regardless of provider
- **Robust Error Handling**: Clear distinction between retryable and permanent errors
- **Automatic Retries**: Configurable retry mechanism with exponential backoff
- **Context Passing**: Support for passing context through requests for multi-threaded usage
- **Coherency Testing**: Framework for testing model coherency

## Supported Providers

- OpenAI
- OpenRouter (with provider filtering)
- Fireworks AI
- Chutes
- Google (Gemini models)

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/llm-client.git
cd llm-client
```

## Configuration

Set the required API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export OPENROUTER_API_KEY="your-openrouter-key"
export FIREWORKS_API_KEY="your-fireworks-key"
export CHUTES_API_TOKEN="your-chutes-token"
export GEMINI_API_KEY="your-gemini-key"
```

## Basic Usage

```python
from llm_client import get_provider, retry_request

# Get a provider instance
provider = get_provider("openai")

# Prepare your message
messages = [{"role": "user", "content": "Explain quantum computing in simple terms."}]

# Make a request with retry logic
response = retry_request(
    provider=provider,
    messages=messages,
    model_id="gpt-4o-2024-08-06",
    max_retries=3,
    timeout=60
)

# Check for success
if response.success:
    # Access the content from the standardized response
    print(response.standardized_response["content"])
else:
    # Handle error
    print(f"Error: {response.error_info['message']}")
```

## OpenRouter Provider Filtering

```python
from llm_client import get_provider, retry_request

# Get the OpenRouter provider
provider = get_provider("openrouter")

# To specify which providers to allow
response = retry_request(
    provider=provider,
    messages=[{"role": "user", "content": "Hello, world!"}],
    model_id="openai/gpt-4o-2024-08-06",
    allow_list=["openai", "anthropic"]  # Only use these providers
)

# To specify which providers to ignore
response = retry_request(
    provider=provider,
    messages=[{"role": "user", "content": "Hello, world!"}],
    model_id="openai/gpt-4o-2024-08-06",
    ignore_list=["deepinfra", "anyscale"]  # Don't use these providers
)
```

## Coherency Testing

```python
from llm_client.testing import run_coherency_tests

# Run coherency tests
success, failed_providers = run_coherency_tests(
    target_model_id="gpt-4o-2024-08-06",
    target_provider_name="openai",
    num_workers=4  # Run tests in parallel
)

if success:
    print("All coherency tests passed!")
else:
    print("Some coherency tests failed.")
    
# For OpenRouter, get list of providers that failed
if target_provider_name == "openrouter" and failed_providers:
    print(f"These providers failed: {', '.join(failed_providers)}")
```

## Advanced Usage

See the `examples` directory for more complex usage examples.

## Error Handling

The library distinguishes between retryable errors (network issues, timeouts, rate limits) and permanent errors (invalid API keys, malformed requests). The `retry_request` function automatically handles retries for retryable errors.

## Response Format

All provider responses are standardized to a common format:

```python
{
    "id": "response-id",  # Provider-specific ID if available
    "created": 1625097587,  # Timestamp if available
    "model": "gpt-4o-2024-08-06",  # Model name
    "provider": "openai",  # Provider name
    "content": "Response text here",  # The actual response content
    "finish_reason": "stop",  # Reason the generation stopped
    "usage": {  # Token usage information if available
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

## License

[Apache 2.0](LICENSE)
