# Codex OAuth Testing

Status: Fake-server coverage implemented; live account validation pending.

## Implemented Foundation

V2 owns its OAuth credentials independently. It does not import or modify Codex
CLI `auth.json`.

The generic OAuth implementation provides:

- Authorization Code flow with PKCE S256 and state validation.
- Canonical configurable authorization and token endpoints.
- Atomic credential replacement with `0600` file permissions.
- Thread and process coordination around rotating refresh tokens.
- Single-flight refresh scoped to one credential store.
- Safe error messages that do not include token endpoint bodies or credentials.
- A provider auth interface that supplies request headers without adding them to
  normalized responses or unredacted wire records.

`CodexOAuthManager` additionally provides:

- The canonical OpenAI scopes `openid profile email offline_access`.
- ChatGPT account ID extraction from the access-token claim.
- `Authorization`, `chatgpt-account-id`, `originator`, and Responses beta headers.
- An independent default credential path under
  `~/.config/llm_client/codex-oauth.json`.
- Codex Responses request defaults: stateless storage, streaming, and encrypted
  reasoning content inclusion.

No OAuth client ID is hard-coded. `client_id` remains explicit until a supported
third-party registration or public-client policy is confirmed.

## Automated Fake-Server Coverage

`tests/test_oauth.py` runs against an in-process HTTP OAuth and inference server.
It verifies:

- PKCE URL and scope construction.
- Callback state rejection.
- Authorization-code exchange.
- Atomic credential persistence and permissions.
- Refresh-token rotation.
- One refresh across 100 concurrent token requests.
- Authenticated inference through the V2 client.
- Access-token, refresh-token, and account-ID redaction.
- Codex-specific request headers and Responses defaults using a fake JWT.

Run:

```bash
uv run pytest -q tests/test_oauth.py
```

## Live Validation Still Required

Live testing requires a user to complete the browser authorization step and must
answer these external-policy questions:

1. Which OAuth client ID is supported for `llm_client`?
2. Which redirect URIs are registered for that client?
3. Is direct external-tool use supported without an intermediary auth service?
4. Does the token response rotate refresh tokens consistently?
5. Does the access token contain the expected ChatGPT account claim?
6. Which Codex models are exposed to the signed-in account?

After those are resolved, live validation should:

1. Construct `CodexOAuthManager.create(...)` with a supported client ID.
2. Call `begin_login()` and open the returned URL.
3. Pass the resulting callback URL to `complete_redirect()`.
4. Create `Client(auth={"codex": manager})`.
5. Send a one-turn request to `codex/<advertised-model>`.
6. Serialize the operation and run credential canary checks.
7. Restore and continue the conversation, verifying encrypted reasoning replay.
8. Force token expiry and verify one refresh under parallel load.
9. Confirm the Codex CLI login remains unaffected.

The browser callback listener and user-facing CLI login command are intentionally
left for the live-validation phase. The protocol, storage, refresh, integration,
and fake-server test boundaries are already isolated so that work does not require
changing the conversation or provider APIs.
