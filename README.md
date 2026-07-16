# LLM Client Library

## Warning: Vibe-coded.

A clean, modular client library for interacting with LLM providers and local OpenAI-compatible servers with standardized responses and robust error handling.

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
- Google Agent Platform (OpenAI-compatible)
- TNG Tech (OpenAI-compatible)
- X.AI (OpenAI-compatible)
- Moonshot (OpenAI-compatible)
- Stepfun (OpenAI-compatible)
- Tinker (Sampling API via `tinker` + `tinker_cookbook`)
- Local OpenAI-compatible servers such as vLLM, llama.cpp, or llama-cpp-python

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/llm-client.git
cd llm-client
```

### Optional: Tinker provider dependencies (uv)

The `tinker` provider has optional dependencies. With `uv`:

```bash
uv pip install -e '.[tinker]'
```

This will install:
- `tinker` from PyPI
- `tinker_cookbook` from PyPI, with the `inkling` renderer extra

## Configuration

Set the required API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export OPENROUTER_API_KEY="your-openrouter-key"
export FIREWORKS_API_KEY="your-fireworks-key"
export CHUTES_API_TOKEN="your-chutes-token"
export GEMINI_API_KEY="your-gemini-key"
export GOOGLE_API_KEY="your-google-agent-platform-key"
export GOOGLE_AGENT_PLATFORM_PROJECT_ID="your-google-cloud-project-id"
export GOOGLE_AGENT_PLATFORM_LOCATION="global"
export GOOGLE_AGENT_PLATFORM_ENDPOINT="openapi"
export XAI_API_KEY="your-xai-key"
```

For Google Agent Platform, use provider name `google_agent_platform` and a
publisher-qualified OpenAI-compatible model ID, for example
`xai/grok-4.1-fast-non-reasoning`.

For local OpenAI-compatible servers, use provider name `local` or
`openai_compatible`. You can configure the endpoint once with
`LOCAL_LLM_BASE_URL` and pass the server's model name normally:

```bash
export LOCAL_LLM_BASE_URL="http://127.0.0.1:8000/v1"
# Optional; omit this for local servers that do not require Authorization.
export LOCAL_LLM_API_KEY="your-local-server-key"
```

```python
from llm_client import get_provider, retry_request

provider = get_provider("local")
response = retry_request(
    provider=provider,
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    model_id="served-model-name",
    max_retries=1,
    timeout=120,
)
```

For one-off calls and registries such as `mq`, you can also encode the endpoint
directly in `model_id`:

```python
model_id = "127.0.0.1:8000/Qwen/Qwen3-8B"
# optional provider-prefixed spelling:
model_id = "local/127.0.0.1:8000/Qwen/Qwen3-8B"
# full/custom base URL form:
model_id = "http://127.0.0.1:8000/custom/v1::served-model-name"
```

`127.0.0.1:8000/Qwen/Qwen3-8B` calls
`http://127.0.0.1:8000/v1/chat/completions` and sends `Qwen/Qwen3-8B` as the
OpenAI `model` value. Full URLs use `::` so the URL path and model boundary are
explicit; full URL values without `::` fail with a non-retryable
`invalid_option` error. The local provider uses the shared OpenAI-compatible
transport, so options such as `temperature`, `max_tokens`, `timeout`, and
`transport="stream"` behave like they do for other OpenAI-style providers.

## Tinker Provider Usage

Model IDs for the `tinker` provider are encoded to include the base model, renderer, and checkpoint path:

- Recommended: `<base_model>::<renderer>::<tinker_path>`
- Also supported: `<base_model>::<tinker_path>` (renderer from `renderer=` or defaults to `role_colon`)
- Also supported: `<tinker_path>` with `base_model=...` in options

```python
from llm_client import get_provider, retry_request

provider = get_provider("tinker")

response = retry_request(
    provider=provider,
    messages=[{"role": "user", "content": "Hello!"}],
    model_id="Qwen/Qwen3-30B-A3B::qwen3::tinker://your-checkpoint-path",
    max_retries=2,
    timeout=120,
)

print(response.standardized_response["content"])
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

## OpenRouter Anthropic Messages API

OpenRouter also exposes an Anthropic Messages-compatible API. Use `request_format="anthropic_messages"` when a model needs native Anthropic request fields such as adaptive thinking.

```python
from llm_client import get_provider, retry_request

provider = get_provider("openrouter")

response = retry_request(
    provider=provider,
    messages=[{"role": "user", "content": "Solve this carefully."}],
    model_id="anthropic/claude-opus-4.7",
    request_format="anthropic_messages",
    thinking={"type": "adaptive"},
    output_config={"effort": "high"},
    max_tokens=16000,
)

print(response.request_format)       # anthropic_messages
print(response.raw_response_format)  # openrouter.anthropic_messages
print(response.standardized_response["content"])
```

The raw provider payload is preserved unchanged in `raw_provider_response`. The response object carries `request_format` and `raw_response_format` so downstream code can tell which raw schema it received without mutating the raw payload.

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
)

if resp.success:
    print(resp.standardized_response["content"])  # aggregated content
else:
    print(f"Error: {resp.error_info['message']}")
```

If you accidentally pass `stream=True` without `transport='stream'`, the providers return a non‑retryable error with type `invalid_option`. This ensures callers don’t mistakenly expect token‑level streaming, which the library doesn’t provide.

### Direct Provider Call (no retry_request)

You can call the provider directly without the retry helper. Use the same `transport='stream'` option:

```python
from llm_client import get_provider

provider = get_provider("openai")  # or "openrouter", "xai", etc.
messages = [{"role": "user", "content": "Write a long answer..."}]

resp = provider.make_chat_completion_request(
    messages=messages,
    model_id="gpt-4o-2024-08-06",
    transport="stream",   # SSE on the wire; final aggregated LLMResponse
)

if resp.success:
    print(resp.standardized_response["content"])  # aggregated content
else:
    print(f"Error: {resp.error_info and resp.error_info.get('message')}" )
```

Notes:
- `stream=True` is not supported directly and will return an `invalid_option` error. Always use `transport='stream'`.
- This returns a single `LLMResponse` after aggregating streamed chunks; partial tokens are not exposed.
- When using `transport='stream'`, the `timeout` option is treated as an overall request budget (enforced within the streaming layer). You can keep the default or pass a single numeric value; no per-read tuning required.

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
    "native_finish_reason": "STOP",  # Provider-native reason, if available
    "normalization_evidence": {
        "finish_reason": {
            "source": "choices[0].finish_reason",
            "value": "STOP",
            "normalized": "stop",
        }
    },
    "usage": {  # Token usage information if available
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}
```

Content-filter stops from safety, policy, license, or recitation filters are
normalized into `error_info["type"] == "content_filter"` and are non-retryable.
When the provider exposes the triggering field, `error_info` also includes
`native_finish_reason` and `normalization_evidence` so callers can inspect why
the broad bucket was chosen.

## License

[Apache 2.0](LICENSE)
