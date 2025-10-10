# LLM Client Library

## Warning: Vibe-coded.

A clean, modular client library for interacting with various LLM providers (OpenAI, OpenRouter, Fireworks, Chutes, Google, X.AI) with standardized responses and robust error handling.

## Features

- **Unified Interface**: Common API for all supported LLM providers
- **Standardized Responses**: Consistent response format regardless of provider
- **Robust Error Handling**: Clear distinction between retryable and permanent errors
- **Automatic Retries**: Configurable retry mechanism with exponential backoff
- **Context Passing**: Support for passing context through requests for multi-threaded usage
- **Coherency Testing (OpenRouter)**: Sub‑provider gating to automatically exclude failing endpoints; optionally enforces reasoning/"thinking" output

## Supported Providers

- OpenAI
- OpenRouter (with provider filtering)
- Fireworks AI
- Chutes
- Google (Gemini models)
- X.AI (OpenAI-compatible)

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
export XAI_API_KEY="your-xai-key"
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

The coherency tester works with any provider, but it is primarily useful for OpenRouter.

- OpenRouter: a single model can have multiple sub‑providers. The tester enumerates sub‑providers, runs a small prompt suite for each, and reports which sub‑providers passed or failed so you can build an `ignore_list` for production calls.
- Other providers: the tester runs the same prompt suite once and returns a simple pass/fail summary for that provider/model pair (no sub‑provider breakdown).

It also supports optional reasoning enforcement: when you pass a `reasoning` override, the tester will fail providers that don’t return thinking output when enabled, or that return thinking when disabled.

```python
from llm_client import get_provider, retry_request
from llm_client.testing import run_coherency_tests

model_id = "qwen/qwen3-next-80b-a3b-thinking"  # example OpenRouter model

# Optional: enforce reasoning behavior during testing
# Use either max_tokens OR effort (not both)
request_overrides = {
    "reasoning": {
        "enabled": True,
        "max_tokens": 2048,
        # "effort": "medium",  # alternative to max_tokens
    }
}

# Run coherency tests against all available sub‑providers
success, failed_providers = run_coherency_tests(
    target_model_id=model_id,
    target_provider_name="openrouter",
    num_workers=4,
    request_overrides=request_overrides,
    # verbose=True,  # optional: dump raw responses on failures
)

if not success:
    print("No sub‑provider passed all tests; consider retrying later.")

# When making real requests, avoid failing providers
provider = get_provider("openrouter")
messages = [{"role": "user", "content": "Write a limerick about databases."}]

response = retry_request(
    provider=provider,
    messages=messages,
    model_id=model_id,
    ignore_list=failed_providers,    # exclude failing sub‑providers
    **request_overrides,
)

if response.success:
    print(response.standardized_response.get("content", "<no content>"))
else:
    print(f"Error: {response.error_info and response.error_info.get('message')}")
```

## Streaming Transport

Some providers support server-sent events (SSE) streaming for long generations. This library exposes a streaming transport that keeps the HTTP connection active and aggregates streamed chunks into a single final result, preserving the existing `LLMResponse` interface.

- Enable it via the request option `transport='stream'`.
- Do not pass `stream=True` directly; the client will reject it. This library does not expose token-by-token streaming to the caller.

Example:

```python
from llm_client import get_provider, retry_request

provider = get_provider("openrouter")  # or "openai", "xai", etc.
messages = [{"role": "user", "content": "Write a long answer..."}]

resp = retry_request(
    provider=provider,
    messages=messages,
    model_id="openai/gpt-4o-2024-08-06",
    transport="stream",       # use SSE under the hood, but return one final response
    timeout=(10, 300),         # optional: tolerate slow token gaps
)

if resp.success:
    print(resp.standardized_response["content"])  # aggregated content
else:
    print(f"Error: {resp.error_info['message']}")
```

If you accidentally pass `stream=True` without `transport='stream'`, the providers return a non‑retryable error with type `invalid_option`. This ensures callers don’t mistakenly expect token‑level streaming, which the library doesn’t provide.

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
