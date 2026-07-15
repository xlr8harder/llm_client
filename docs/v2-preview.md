# V2 Preview

V2 is an additive preview. Existing provider classes and `LLMResponse` behavior
remain unchanged.

## Available now

- `Client`, `AsyncClient`, configured `Model` routes, and serializable
  `Conversation` objects.
- OpenRouter and local OpenAI-compatible access through Chat Completions.
- Explicit OpenRouter Messages and Responses protocol selection.
- Canonical messages, operations, attempts, normalized errors, finish reasons,
  provider replay state, and raw wire evidence.
- Stable conversation JSON round trips and lossless legacy-response import.
- Credential and sensitive-header redaction; local endpoint hosts are never
  serialized.
- Thread-safe shared sync clients and task-safe async clients. One conversation
  permits one active writer; fork it for parallel continuations.
- Aggregate SSE decoding with ordered raw stream events retained in the operation
  record.
- Generic PKCE OAuth and isolated Codex OAuth foundations, including rotating
  refresh-token single-flight behavior.

## Routes and protocols

The shortest path uses the provider default protocol:

```python
response = llm_client.generate(
    "Explain speculative decoding.",
    model="openrouter/openai/gpt-5.6-sol",
)
```

Override the protocol on a configured route:

```python
model = client.model(
    "openrouter/openai/gpt-5.6-sol",
    protocol="responses",
    options={"reasoning": {"enabled": True}},
)
```

Supported preview protocols are `chat_completions`, `messages`, and `responses`.
Unsupported protocol names fail during model construction.

Local routes require a runtime endpoint:

```python
client = llm_client.Client(local_endpoint="http://127.0.0.1:8000/v1")
model = client.model("local/unset/Qwen/Qwen3-4B")
conversation = model.conversation()
```

Serialized local routes use `local/unset/Qwen/Qwen3-4B`. Restored conversations
fail with a specific error until rebound to a client with `local_endpoint` set.
Binding never silently changes the stored model ID.

## Not yet public

- Incremental caller-visible stream iteration (`model.stream(...)`). SSE is
  currently consumed into one response while preserving every raw event.
- A browser callback listener or login CLI for live Codex OAuth.
- Legacy Completions protocol support in the V2 facade.

The complete target contract and migration rules are in
[V2 Requirements and Design](v2-requirements-design.md). Codex test boundaries
are in [Codex OAuth Testing](codex-oauth-testing.md).
