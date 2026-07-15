# llm_client V2 Requirements and Design

Status: Draft

## 1. Purpose

`llm_client` V2 must preserve the library's original goal: make the common result
easy to obtain, provide a useful normalized view across providers, and retain the
raw evidence needed for later debugging, benchmarking, migration, and revised
analysis.

The shortest useful path must remain short:

```python
response = llm_client.generate(
    "Explain speculative decoding.",
    model="openrouter/openai/gpt-5.6-sol",
)

print(response.content)
```

Advanced facilities must be available without becoming mandatory ceremony.

## 2. Design Principles

1. Existing unchanged code must retain identical behavior.
2. Normalize additively, never destructively.
3. Preserve raw request, response, error, and provider metadata whenever capture
   is enabled.
4. Keep raw, normalized, derived, provider, library, migration, and application
   namespaces separate.
5. Treat ambiguous provider outcomes as ambiguous, not successful.
6. Keep protocol selection separate from model identity.
7. Make route changes explicit and auditable.
8. Keep model catalog policy outside `llm_client`.
9. Use one canonical builder for live requests, imports, migration, and
   deserialization.
10. Make canonical serialization versioned, credential-free, and round-trip
    stable.

## 3. Scope

V2 covers:

- One-shot generation.
- Stateful, serializable conversations.
- Text and extensible multimodal content.
- Multiple wire protocols per provider.
- Provider-specific replay state.
- Structured operations, retries, errors, and raw wire capture.
- Local OpenAI-compatible endpoints.
- Legacy record migration.
- Synchronous, asynchronous, and streaming-compatible data models.

V2 does not become:

- A model catalog.
- An agent or tool-execution framework.
- A prompt-template framework.
- A hidden cross-provider billing or fallback system.
- A server-side conversation service.

The library may represent tool calls and tool results without executing them.

## 4. Compatibility Contract

Existing users must receive identical behavior from unchanged code. This includes:

- Existing imports, functions, methods, and signatures.
- Provider lookup and model ID interpretation.
- Default endpoints and wire protocols.
- Request payloads and relevant headers.
- Option defaults and validation.
- Timeout and retry behavior.
- Error and finish-reason classification.
- `LLMResponse` attributes and dictionary shapes.
- `standardized_response`, `raw_provider_response`, and `error_info`.
- Environment variables and logging behavior.

Legacy calls must not begin selecting Responses, Messages, or another protocol
because V2 considers it preferable. In particular,
`make_chat_completion_request()` remains Chat Completions.

V2 is introduced as a second public facade over shared components:

```text
Legacy facade                 V2 facade
LLMResponse                   Response
retry_request                 Client / Model
make_chat_completion_request  Conversation
            \                /
             shared adapters
             protocols
             transports
             normalization
```

Existing providers may initially feed V2 through an `LLMResponse` importer. A
provider moves to the shared canonical engine only after differential tests prove
that its legacy projection is unchanged.

## 5. Public API

### 5.1 One-shot generation

```python
response = llm_client.generate(
    "Explain speculative decoding.",
    model="openrouter/openai/gpt-5.6-sol",
    system="Be concise.",
)
```

Standard messages are also accepted:

```python
response = llm_client.generate(
    model="openrouter/openai/gpt-5.6-sol",
    messages=[
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain speculative decoding."},
    ],
)
```

Supplying both a direct input and `messages` is an error.

Common values are direct properties:

```python
response.ok
response.content
response.reasoning
response.finish_reason
response.native_finish_reason
response.usage
response.error
response.messages
response.raw
```

Callers must not unpack provider choices or content blocks to obtain ordinary
text. Advanced evidence remains available through `response.operation`.

### 5.2 Client and model access route

```python
client = llm_client.Client()

model = client.model(
    "openrouter/openai/gpt-5.6-sol",
    protocol="responses",
    options={"reasoning": {"enabled": True}},
)

response = model.generate("Explain speculative decoding.")
```

`Model` means a configured access route, not model weights. It identifies a
provider, provider model ID, protocol, and defaults. It contains no conversation
history.

### 5.3 Conversations

```python
conversation = model.conversation(system="Be concise.")
response = conversation.send("What is speculative decoding?")
response = conversation.send("How does it affect latency?")
```

Standard messages import directly:

```python
conversation = Conversation.from_messages(
    messages,
    model="openrouter/openai/gpt-5.6-sol",
)
```

A conversation is primarily serializable data. A runtime binding is optional:

```python
conversation = Conversation.from_json(data)
conversation.messages
conversation.operations
conversation.fork()

conversation.bind(client)
conversation.send("Continue.")
```

Convenience forms use the same deserializer:

```python
Conversation.from_json(data, client=client)
client.load_conversation(data)
```

Clients, credentials, locks, HTTP pools, callbacks, and provider instances are
never serialized.

### 5.4 Sync, async, and streaming

V2 should provide separate sync and async clients over shared schemas and
adapters:

```python
client = llm_client.Client()
async_client = llm_client.AsyncClient()
```

Streaming yields structured events and produces a final ordinary `Response`:

```python
with model.stream("Hello") as stream:
    for event in stream:
        if event.type == "text_delta":
            print(event.text, end="")

response = stream.response
```

### 5.5 Parallelism and concurrency

High parallelism is a primary operating mode, not an optional optimization. V2
must support many simultaneous requests from threads and async tasks without a
process-wide request lock or shared mutable request state.

Concurrency ownership is explicit:

- `Client` is safe to share across threads. `AsyncClient` is safe to share across
  tasks on its owning event loop.
- Configured `Model` access routes and completed canonical records are immutable
  and safe to share.
- Every operation receives isolated builder, request, response, retry, and stream
  aggregation state.
- Provider adapters and normalization registries do not store mutable per-request
  state on shared provider instances.
- Connection pools may be shared when the transport supports it; response bodies
  and live streams are never shared between requests.
- Credential refresh uses single-flight coordination scoped to one auth profile.
  It must not serialize ordinary inference requests or unrelated accounts.
- Unknown-field warning deduplication, ID generation, metrics, and caches are
  thread-safe and do not affect result correctness.
- Optional rate or concurrency limiting is explicit and scoped. The library does
  not impose a hidden global throttle.

A conversation has ordered mutable history and therefore has one writer at a
time. Concurrent `send`, `resume`, `rebind`, or destructive history operations on
the same conversation fail immediately with `ConversationBusyError` rather than
silently racing or choosing an arbitrary order. Independent conversations and
conversation forks remain fully parallel.

```python
shared_model = client.model("openrouter/openai/gpt-5.6-sol")

# Each worker owns its conversation; the client and model are shared.
conversation = shared_model.conversation(messages=worker_messages)
response = conversation.send(worker_prompt)
```

If ordered concurrent submission to one logical conversation is needed, the
application serializes those submissions or uses an explicit future queueing
facility. V2 does not infer ordering from thread or task scheduling.

Async support uses a native async transport rather than permanently wrapping the
synchronous client in a thread executor. Sync and async implementations share
protocol codecs, builders, normalization, error classification, and canonical
serialization. Cancellation records an interrupted attempt and leaves the
conversation in a recoverable state.

## 6. Canonical Data Model

### 6.1 Conversation

A conversation contains:

- `schema_version`.
- Ordered model-visible `messages`.
- Ordered generation `operations`.
- A `default_route` for future requests.
- Namespaced metadata.

A conversation may contain operations from more than one provider, model, or
protocol. Every operation records the route it actually used.

### 6.2 Messages

A message is one model-visible item, not a request/reply pair:

```python
@dataclass(frozen=True)
class Message:
    id: str
    role: str
    content: tuple[ContentPart, ...]
    status: str
    provider_state: ProviderState | None
    metadata: dict
```

Requirements:

- Support `system`, `developer`, `user`, `assistant`, and `tool` roles.
- Permit consecutive messages with the same role.
- Preserve order exactly.
- Support indexing, iteration, and ordinary slicing.
- Keep completed messages immutable.
- Preserve unknown content blocks rather than discarding them.
- Preserve system and developer messages in their original positions.

```python
conversation.messages[-1]
conversation.messages[:10]
conversation.last_message
```

### 6.3 Content

The content model must be ordered and extensible. Initial content kinds include:

- Text.
- Image.
- Audio.
- File.
- Tool call.
- Tool result.
- Reasoning summary.
- Unknown provider content.

`message.text` and `response.content` provide the simple text projection.

### 6.4 Provider state

Opaque replay state is separate from normalized content:

```python
@dataclass(frozen=True)
class ProviderState:
    provider: str
    protocol: str
    schema_version: int
    replay: tuple[dict, ...]
    metadata: dict
```

Examples include:

- OpenAI encrypted reasoning items.
- Anthropic signed thinking blocks.
- Google thought signatures.
- Provider-specific tool state.

Adapters own validation and replay. Provider state must be JSON-compatible and
must never contain credentials.

### 6.5 Response

`Response` is the normalized result of an operation. It exposes common fields
directly while retaining its operation and raw evidence.

Compatibility projections may expose:

- `standardized_response`.
- `raw_provider_response`.
- `error_info`.

These projections must not mutate the canonical object or raw provider payload.

## 7. Operations, Attempts, and Recovery

Messages contain semantic history. Operations contain execution history. Errors
are operation outcomes, not synthetic messages.

One reply operation records:

- Input message IDs.
- Output message IDs.
- Requested and resolved access route.
- Attempts and retries.
- Final outcome.
- Normalized error.
- Raw wire evidence.
- Timing and normalization notices.

Each network/provider attempt records its own provider, model, protocol, timing,
retryability, error, and wire record.

Submitting new input appends the input message before execution. If execution
produces no valid assistant output:

- No artificial assistant message is appended.
- The failed operation is retained.
- The input remains pending.
- The conversation remains valid.

```python
conversation.pending_messages
conversation.last_operation
conversation.resume()
conversation.replace_pending("Rephrased request")
conversation.remove_pending()
```

Terminal errors require an explicit forced retry or changed input/route. A
provider refusal containing valid assistant text may be committed as a message
while the operation remains classified as moderated.

Partial stream output remains operation evidence by default:

```python
conversation.last_operation.partial_output
conversation.resume()
conversation.accept_partial()
conversation.discard_partial()
```

Deserializing a `running` operation converts it to `interrupted` unless the
runtime can prove it remains active.

## 8. Errors and Finish Reasons

V2 must preserve the error and finish-reason behavior developed for reliable
evaluation.

Normalized error categories initially include:

- Authentication.
- Authorization.
- Moderation.
- Rate limiting.
- Timeout.
- Transport failure.
- Invalid request.
- Invalid response.
- Provider failure.
- Interruption.
- Unknown.

An error retains:

- Human-readable message.
- Retryability as `True`, `False`, or unknown.
- HTTP status.
- Provider code and type.
- Normalized and native finish reasons.
- Raw provider error.
- Normalization evidence.

Programming and local validation errors raise before execution. Executed provider
outcomes return a `Response`; callers may opt into exceptions with
`response.raise_for_error()`.

Required classification behavior:

- Native finish reasons always remain accessible.
- Provider errors are not overwritten by misleading secondary metadata.
- Missing required finish metadata is a metadata error.
- Unknown finish reasons remain unknown or fail strict validation.
- Moderation is distinct from transport failure.
- Truncation is distinct from successful completion.
- Retry/fallback outcomes do not erase earlier attempt evidence.
- Every normalized classification records its source evidence.

The legacy and V2 facades use one shared normalization component. Existing
`llm-compliance` fixtures should become `llm_client` compatibility tests before
downstream consumers remove parallel classification code.

## 9. Protocols

Protocol is separate from provider and model identity. Initial normalized names:

- `chat_completions`.
- `messages`.
- `responses`.
- `completions`.

The existing `request_format` option remains a compatibility alias. Conflicting
`protocol` and `request_format` values raise.

Protocol resolution order is:

1. Explicit call override.
2. Protocol required by retained conversation provider state.
3. Configured model access route.
4. Provider compatibility default.

V2 initially avoids hidden model-family protocol inference:

- Codex defaults to Responses.
- Direct Anthropic defaults to Messages.
- Existing OpenRouter defaults to Chat Completions.
- OpenRouter Responses and Messages require an explicit protocol or configured
  model route.
- Generic local OpenAI-compatible access defaults to Chat Completions.

An explicit future `protocol="auto"` may use capability discovery without
changing existing defaults.

## 10. Provider and Model Identity

A route distinguishes:

- Access provider, such as `openrouter`.
- Provider-specific model ID, such as `openai/gpt-5.6-sol`.
- Wire protocol, such as `responses`.
- Actual backend provider, such as `DeepInfra`.

Protocol is not encoded into model slugs.

Different providers may use different IDs for the same conceptual model.
Changing provider, provider model ID, protocol, or endpoint is an explicit
`rebind`, even when application metadata says the canonical model is unchanged.

```python
conversation.rebind(
    client.model("other-provider/provider-model-name"),
)
```

`bind()` supplies runtime resources for the existing route. It never substitutes
a different model. A mismatched model binding raises and instructs the caller to
use `rebind()`.

If replay-dependent provider state exists, rebind fails by default. The caller
must explicitly translate or drop it:

```python
conversation.rebind(new_model, provider_state="drop")
```

Rebind records old and new provider-specific identities and whether state was
translated or dropped. A caller-owned canonical model hint may assist UI and
catalog resolution but never proves provider-state compatibility.

## 11. Local Access

Local OpenAI-compatible endpoints use the same model and protocol abstractions:

```python
model = client.model(
    "local/127.0.0.1:8000/qwen3-4b",
    protocol="chat_completions",
)
```

Local endpoint identity is runtime-private. Canonical serialization stores:

```text
local/unset/qwen3-4b
```

It does not store local endpoint names, hostnames, IP addresses, or ports. A
restored local conversation cannot send until an endpoint is bound:

```python
conversation.bind(client, local_endpoint="http://127.0.0.1:8000/v1")
```

Binding supplies only the endpoint. It does not change `qwen3-4b`. If the server
supports model discovery, binding may validate availability. It never silently
chooses another advertised model.

Serialized local wire records replace endpoint authority with a local sentinel
while preserving method, path, protocol, body, options, response, and timing.

## 12. Request Options

Option ownership is explicit:

```python
model.generate(
    "...",
    max_output_tokens=4000,
    temperature=0.2,
    provider_options={"order": ["deepinfra"]},
    extra_body={"future_field": {}},
    extra_headers={"X-Provider-Metadata": "enabled"},
)
```

- Stable common options use direct normalized arguments.
- Provider controls use `provider_options`.
- Unrestricted protocol payload extensions use `extra_body`.
- Transport extensions use `extra_headers`.
- Explicit pass-through fields are recorded and are not treated as unexpected.
- Conflicting normalized and pass-through fields raise unless an explicit
  override policy is selected.
- Unknown explicit fields are transmitted and preserved rather than dropped.

## 13. Raw Capture and Namespace Ownership

Raw capture may retain:

- Request method, path, headers, and body.
- Response status, headers, and body.
- Ordered stream events.
- Request and generation IDs.
- Timing and retry data.
- Transport exceptions.

Capture policies may include `none`, `response`, `full`, and
`full_with_stream_events`.

Canonical namespace rules:

- `wire.request` is transmitted request evidence, subject only to documented
  privacy transformations.
- `wire.response` is received response evidence, subject only to documented
  redaction.
- `result` contains normalized projections.
- `provider_state` contains opaque replay state.
- `normalization` contains evidence and notices.
- `metadata.llm_client` contains library execution metadata.
- `metadata.provider.<provider>` contains provider enrichment obtained outside
  the original response.
- `metadata.application` is caller-owned.
- `metadata.migration` contains import provenance.

Derived fields are never injected into raw provider payloads. Unknown provider
fields remain at their original raw paths.

Normalization evidence points to source paths where practical:

```json
{
  "finish_reason": {
    "source": "wire.response.body.choices[0].finish_reason",
    "value": "stop",
    "normalized": "stop"
  }
}
```

The legacy `_llm_client` envelope remains an import/export compatibility format,
not part of canonical V2 serialization.

## 14. Secret Redaction and Host Privacy

Normal serialization and logging replace authentication material with the string:

```text
[REDACTED]
```

Redaction covers:

- Known active secret values wherever encountered.
- Authorization, API-key, proxy-authentication, and cookie headers.
- Access tokens, refresh tokens, and client secrets.

Payload value types and structures otherwise remain intact. Unredacted
serialization requires a conspicuous explicit option and is never enabled by a
general debug flag. OAuth exchanges do not enter conversation wire records.

Host privacy requirements:

- Do not intentionally collect local socket addresses.
- Do not serialize proxy configuration or environment variables.
- Do not add local hostnames, usernames, working directories, or IP addresses to
  request headers.
- Sanitize transport exception structures before storage.
- Review every captured request-header source.
- For local providers, remove endpoint authority and `Host` values during normal
  serialization.
- Do not scan and mutate arbitrary model output for IP-like strings.

Wire-capture changes require a privacy review. Documentation must state that
arbitrary provider responses and caller metadata can themselves contain
identifying information.

## 15. Unknown Provider Fields

Unknown fields are preserved, surfaced, and reviewed. Blind preservation is not
considered full support.

Coverage states are:

- `normalized`: understood and represented canonically.
- `preserved`: retained raw but not interpreted.
- `ignored`: intentionally excluded with a documented reason.
- `unknown`: newly encountered and unreviewed.

Unexpected fields emit `UnknownProviderFieldWarning` once per process for each:

```text
(provider, protocol, direction, JSON path)
```

Warnings contain paths and observed types, never values. Every affected operation
also records a durable normalization notice, even when the console warning has
already been deduplicated.

Intentionally open maps, such as tool arguments, application metadata, and
structured output, do not warn for arbitrary keys. Explicit caller pass-through
fields are marked as caller-supplied rather than unexpected.

Modes:

```python
unknown_fields="warn"      # Default: preserve and warn
unknown_fields="preserve"  # Preserve silently
unknown_fields="error"     # Preserve evidence and fail interpretation
```

## 16. OpenRouter

OpenRouter support uses generic provider, protocol, enrichment, option, and field
coverage facilities.

V2 must support OpenRouter's Chat Completions, Messages, Responses, and legacy
Completions routes without changing legacy defaults.

It must preserve, when returned or explicitly fetched:

- Requested and returned models.
- Selected backend provider and endpoint variant.
- Router metadata and fallback attempts.
- Generation, request, and upstream IDs.
- Normalized and native finish reasons.
- Provider error codes and metadata.
- Usage, native token counts, cost, latency, and cache data.
- Streaming terminal metadata.

Provider routing options and future OpenRouter request fields pass through generic
`provider_options`, `extra_body`, and `extra_headers` facilities. The final
transmitted request is retained in the wire record.

Optional generation enrichment may be configured as `never`, `on_error`, or
`always`. Enrichment belongs under `metadata.provider.openrouter`, not inside the
original response payload.

OpenRouter provides a manual and fixture-based cross-protocol test case:

1. Send semantically equivalent requests over Chat Completions, Messages, and
   Responses where supported.
2. Normalize each response.
3. Import raw fixtures through the builder.
4. Compare structural invariants, classifications, and provenance.
5. Verify raw protocol differences remain complete.
6. Verify serialize/deserialize round trips.

Live generated text is not expected to match byte-for-byte. Fixture normalization
must match exactly.

## 17. Request Reproducibility

Every operation records one compact `request_spec` containing:

- Requested model and protocol.
- Resolved provider and provider model ID.
- Resolved protocol.
- Effective common and provider options.
- Routing, fallback, timeout, retry, streaming, and capture settings.
- Resolution source supplied by the caller.
- `llm_client` version.

Every attempt separately records the final redacted wire request. The request spec
explains how a request was resolved; the wire record proves what was transmitted.

Model catalog resolution remains outside `llm_client`. A caller such as
`llm-compliance` resolves canonical model, provider model ID, protocol, and
experiment defaults, then passes concrete values to `llm_client`. Optional catalog
name and fingerprint belong in `metadata.application`.

The serialized operation, not the current external catalog, is authoritative for
reproduction.

## 18. Canonical Builder

One canonical `ConversationBuilder` is used by:

- Live provider execution.
- Conversation methods.
- Standard-message imports.
- Legacy imports.
- Deserialization.
- Retry, resume, and stream aggregation.
- Tests and fixtures.

It provides controlled operations such as:

```python
builder.add_message(...)
builder.begin_operation(...)
builder.add_attempt(...)
builder.attach_wire_request(...)
builder.attach_wire_response(...)
builder.add_stream_event(...)
builder.attach_error(...)
builder.attach_provider_state(...)
builder.attach_metadata(...)
builder.complete_operation(...)
builder.build()
```

Provider adapters do not directly mutate conversation lists or construct
partially valid records.

The builder centrally enforces:

- Unique IDs.
- Valid roles and content.
- Valid references and operation transitions.
- Retry numbering and terminal outcomes.
- Provider-state compatibility.
- JSON compatibility.
- Namespace ownership.
- Redaction.
- Unknown-field notices.
- Error and finish-reason normalization evidence.
- Interrupted-operation restoration.

Operation construction is transactional. Unexpected exceptions leave a recorded
failed or interrupted operation, not an impossible partial object.

The same builder supports strict live validation and preservation-oriented legacy
imports through policy options rather than separate implementations.

## 19. Serialization and Copying

Canonical serialization is:

- Versioned.
- JSON-compatible.
- Self-contained for inspection and editing.
- Deterministic where practical.
- Credential-free by default.
- Free of runtime-only objects.
- Capable of restoring provider state and recovery state.

```python
conversation.to_dict()
conversation.to_json()
Conversation.from_dict(data)
Conversation.from_json(data)
```

Round-trip semantic equality is mandatory:

```python
built = builder.build()
restored = Conversation.from_json(built.to_json())

assert restored == built
assert restored.to_dict() == built.to_dict()
```

Messages and completed records are immutable. Shallow copying duplicates mutable
conversation containers while safely sharing immutable records:

```python
fork = copy.copy(conversation)
assert fork.messages is not conversation.messages
assert fork.messages[0] is conversation.messages[0]
```

History helpers include `fork`, `tail`, `with_messages`, and `clear`. History edits
validate provider-state dependencies.

## 20. Legacy Migration

Built-in importers should cover:

- Standard message lists.
- Existing `LLMResponse` objects and dictionaries.
- Existing serialized `llm_client` records and `_llm_client` envelopes.
- Raw Chat Completions responses.
- Raw Anthropic Messages responses.
- Raw Google responses where practical.

Migration preserves all currently logged data, including wire evidence, raw
responses, errors, retryability, request and response formats, context, caller
metadata, finish reasons, and normalization evidence.

Every source field is either:

1. Mapped into a canonical field, or
2. Retained unchanged as unmapped source data.

No field disappears silently.

```python
result = Conversation.import_legacy(record, format="llm_client.v0")
result.conversation
result.detected_format
result.warnings
result.unmapped_fields
```

Source preservation modes are `full`, `unmapped`, and `none`; `full` is the
recommended benchmark migration default.

Irregular formats use the same canonical builder through a supported importer
hook:

```python
def import_record(row, builder):
    builder.add_message(...)
    builder.attach_metadata(...)
    builder.attach_wire_response(...)

Conversation.import_records(rows, importer=import_record)
```

Migration modes are `strict`, `preserve`, and `best_effort`; `preserve` is the
recommended benchmark default.

## 21. Required Tests

### 21.1 Legacy compatibility

Golden and differential tests cover exact request bodies, relevant redacted
headers, URLs, defaults, streaming, successful responses, HTTP errors, moderation,
finish reasons, retryability, timeouts, and raw provider payloads.

### 21.2 Serialization

Round-trip tests cover:

- Empty and populated conversations.
- All initial content kinds.
- Consecutive same-role and system/developer messages.
- Provider state.
- Success, retries, fallback, moderation, truncation, and transport failure.
- Interrupted streams and partial output.
- Raw requests, responses, and stream events.
- Unknown fields and all metadata namespaces.
- Legacy imported records.
- Redacted records.
- Every built-in provider fixture.

### 21.3 Namespace integrity

Tests prove that:

- Raw payloads are unchanged except documented privacy transformations.
- Derived fields do not appear in raw payloads unless supplied by the provider.
- Provider enrichment remains namespaced.
- Unknown fields remain at original raw paths.
- Application metadata cannot overwrite library or provider metadata.
- Legacy export does not mutate canonical objects.

### 21.4 Privacy

Tests inject credential, hostname, username, filesystem path, LAN IP, public-IP,
and proxy URL sentinels through transport diagnostics. Normal serialization must
remove credentials and must not collect or expose runtime host diagnostics.

### 21.5 Classification

Port existing `llm-compliance` finish-reason and error fixtures into shared
`llm_client` contract tests. Unknown reasons must not silently pass.

### 21.6 Provider protocols

Fixture and optional live tests compare supported protocols and ensure equivalent
normalized structure, complete raw preservation, and stable round trips.

### 21.7 Concurrency

Concurrency tests cover:

- Many threads sharing one `Client` and immutable `Model` route.
- Many tasks sharing one `AsyncClient` and immutable `Model` route.
- Isolation of request options, wire records, retries, errors, and stream events.
- No cross-request context, response, credential, or provider-state leakage.
- Parallel requests continuing while one auth profile performs a coordinated
  refresh.
- Exactly one refresh for concurrent expiry detection on the same auth profile.
- Independent refresh and request progress for different auth profiles.
- Thread-safe unknown-field warning deduplication and ID generation.
- Immediate `ConversationBusyError` for concurrent mutation of one conversation.
- Full parallelism across separate conversations and forks.
- Async cancellation producing a serializable interrupted operation.
- Absence of hidden global request or rate-limit serialization.

## 22. Migration Plan

1. Freeze legacy behavior with golden tests.
2. Introduce canonical records and `ConversationBuilder`.
3. Add import from existing `LLMResponse`.
4. Add V2 `Response`, `Client`, `Model`, and `Conversation` without changing
   legacy execution.
5. Add independently owned Codex OAuth and Responses through V2.
6. Add explicit OpenRouter Messages and Responses protocols.
7. Move `mq` to Conversation.
8. Port `llm-compliance` classification fixtures and add legacy import tests.
9. Differentially migrate provider internals to the shared engine.
10. Make legacy methods compatibility projections only after parity is proven.
11. Deprecate duplicate internals only after downstream migration and a defined
    compatibility period.

## 23. Acceptance Summary

V2 is acceptable when:

- The common call remains as simple as `response.content`.
- Existing unchanged code behaves identically.
- Conversations restore without a client and bind without changing models.
- Route and protocol changes are explicit and auditable.
- Local endpoint identity is absent from canonical serialization.
- Raw evidence and legacy data survive round trips without silent loss.
- Secrets and runtime host diagnostics do not leak through normal serialization.
- Unknown provider fields are preserved and made visible for implementation.
- Existing error and finish-reason rigor is retained.
- Provider-specific functionality does not pollute canonical namespaces.
- Model catalog policy remains outside the library.
- Shared clients and models sustain highly parallel threaded and async workloads
  without cross-request state leakage or hidden global serialization.
