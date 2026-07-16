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

- The upstream Codex scopes `openid profile email offline_access
  api.connectors.read api.connectors.invoke`.
- ChatGPT account ID extraction from the access-token claim.
- `Authorization`, `chatgpt-account-id`, `originator`, and Responses beta headers.
- An independent default credential path under
  `~/.config/llm_client/codex-oauth.json`.
- Codex Responses request defaults: stateless storage, streaming, and encrypted
  reasoning content inclusion.

- The public native Codex client ID and its fixed loopback callback shape.
- A browser callback CLI plus manual callback mode for remote environments.
- Automatic credential discovery for sync and async clients.

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
confirm token rotation, account claims, model availability, and server-enforced
usage limits for that account.

After those are resolved, live validation should:

1. Run `llm-client auth login codex`.
2. Create a default `Client()` and send a one-turn request to an available
   `codex/<model>` route.
6. Serialize the operation and run credential canary checks.
7. Restore and continue the conversation, verifying encrypted reasoning replay.
8. Force token expiry and verify one refresh under parallel load.
9. Confirm the Codex CLI login remains unaffected.

The remaining work is live-account validation; automated coverage does not make
requests against a user's subscription.
