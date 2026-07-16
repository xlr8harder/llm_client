from concurrent.futures import ThreadPoolExecutor
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.parse import parse_qs, urlparse
import httpx
import pytest

from llm_client import (
    Client,
    CodexOAuthManager,
    OAuthConfig,
    OAuthCredentialStore,
    OAuthCredentials,
    OAuthError,
    OAuthManager,
    codex_oauth_config,
)


class FakeOAuthState:
    def __init__(self):
        self.lock = threading.Lock()
        self.code_exchanges = 0
        self.refreshes = 0
        self.generation_auth = []
        self.current_refresh = "refresh-1"


def serve_fake_oauth():
    state = FakeOAuthState()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path == "/oauth/token":
                form = parse_qs(body.decode("utf-8"))
                grant = form.get("grant_type", [""])[0]
                with state.lock:
                    if grant == "authorization_code":
                        state.code_exchanges += 1
                        payload = {
                            "access_token": "access-1",
                            "refresh_token": "refresh-1",
                            "expires_in": 3600,
                        }
                    elif (
                        grant == "refresh_token"
                        and form.get("refresh_token", [""])[0] == state.current_refresh
                    ):
                        state.refreshes += 1
                        next_number = state.refreshes + 1
                        state.current_refresh = f"refresh-{next_number}"
                        payload = {
                            "access_token": f"access-{next_number}",
                            "refresh_token": state.current_refresh,
                            "expires_in": 3600,
                        }
                    else:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b'{"error":"invalid_grant"}')
                        return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
                return
            if self.path == "/v1/chat/completions":
                with state.lock:
                    state.generation_auth.append(self.headers.get("Authorization"))
                payload = {
                    "id": "chat-1",
                    "object": "chat.completion",
                    "model": "fake-model",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "authenticated",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode("utf-8"))
                return
            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


def manager_for(server, tmp_path):
    base = f"http://127.0.0.1:{server.server_port}"
    config = OAuthConfig(
        provider="fake",
        client_id="client-1",
        authorization_url=f"{base}/oauth/authorize",
        token_url=f"{base}/oauth/token",
        scopes=("openid", "offline_access"),
        redirect_uri="http://127.0.0.1:1455/callback",
    )
    return OAuthManager(
        config, OAuthCredentialStore(tmp_path / "fake-oauth.json"), refresh_margin=120
    )


def test_pkce_login_exchange_atomic_store_and_permissions(tmp_path):
    server, state = serve_fake_oauth()
    try:
        manager = manager_for(server, tmp_path)
        login = manager.begin_login(state="fixed-state", verifier="v" * 64)
        query = parse_qs(urlparse(login.url).query)
        assert query["state"] == ["fixed-state"]
        assert query["code_challenge_method"] == ["S256"]
        assert query["scope"] == ["openid offline_access"]
        credentials = manager.exchange_code("code-1", login)
        assert credentials.access_token == "access-1"
        assert manager.store.load() == credentials
        assert manager.store.path.stat().st_mode & 0o777 == 0o600
        assert state.code_exchanges == 1
        rejected = manager.begin_login(state="expected", verifier="x" * 64)
        with pytest.raises(OAuthError, match="state did not match"):
            manager.complete_redirect(
                "http://127.0.0.1:1455/callback?code=code-2&state=wrong", rejected
            )
        manager.close()
    finally:
        server.shutdown()
        server.server_close()


def test_concurrent_refresh_is_single_flight_and_rotates_atomically(tmp_path):
    server, state = serve_fake_oauth()
    try:
        manager = manager_for(server, tmp_path)
        manager.store.save(
            OAuthCredentials("expired-access", "refresh-1", time.time() - 10)
        )
        with ThreadPoolExecutor(max_workers=24) as pool:
            tokens = list(pool.map(lambda _: manager.get_access_token(), range(100)))
        assert set(tokens) == {"access-2"}
        assert state.refreshes == 1
        assert manager.store.load().refresh_token == "refresh-2"
        manager.close()
    finally:
        server.shutdown()
        server.server_close()


def test_oauth_token_drives_inference_but_is_redacted_from_wire(tmp_path):
    server, state = serve_fake_oauth()
    manager = manager_for(server, tmp_path)
    manager.store.save(OAuthCredentials("access-live", "refresh-1", time.time() + 3600))
    base = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        with Client(auth={"openai": manager}) as client:
            response = client.model("openai/fake-model", endpoint=base).generate(
                "hello"
            )
        assert response.content == "authenticated"
        assert state.generation_auth == ["Bearer access-live"]
        serialized = json.dumps(response.operation.to_dict())
        assert "access-live" not in serialized
        assert "[REDACTED]" in serialized
    finally:
        manager.close()
        server.shutdown()
        server.server_close()


def test_codex_headers_and_responses_defaults_with_fake_credentials(tmp_path):
    def encoded(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    access = f"{encoded({'alg': 'none'})}.{encoded({'https://api.openai.com/auth': {'chatgpt_account_id': 'acct-secret'}})}.sig"
    config = codex_oauth_config(
        client_id="codex-test-client", auth_base="http://auth.invalid"
    )
    manager = CodexOAuthManager(config, OAuthCredentialStore(tmp_path / "codex.json"))
    manager.store.save(OAuthCredentials(access, "refresh-secret", time.time() + 3600))
    seen = []
    final = {
        "id": "resp-codex",
        "object": "response",
        "status": "completed",
        "model": "gpt-test",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "codex ok"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    }

    def handler(request):
        seen.append((dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json=final)

    try:
        with Client(
            auth={"codex": manager}, transport=httpx.MockTransport(handler)
        ) as client:
            response = client.model(
                "codex/gpt-test", endpoint="https://fake.invalid/codex"
            ).generate("hello")
        headers, body = seen[0]
        assert headers["authorization"] == f"Bearer {access}"
        assert headers["chatgpt-account-id"] == "acct-secret"
        assert body["store"] is False
        assert body["stream"] is True
        assert body["include"] == ["reasoning.encrypted_content"]
        serialized = json.dumps(response.operation.to_dict())
        assert access not in serialized
        assert "acct-secret" not in serialized
        assert serialized.count("[REDACTED]") >= 2
        assert response.content == "codex ok"
    finally:
        manager.close()


def test_codex_authorize_request_matches_upstream_public_client_shape(tmp_path):
    manager = CodexOAuthManager.create(credential_path=tmp_path / "codex.json")
    try:
        login = manager.begin_login(state="state", verifier="v" * 64)
        query = parse_qs(urlparse(login.url).query)
    finally:
        manager.close()
    assert query["client_id"] == ["app_EMoamEEZ73f0CkXaXp7hrann"]
    assert query["redirect_uri"] == ["http://localhost:1455/auth/callback"]
    assert query["scope"] == [
        "openid profile email offline_access api.connectors.read api.connectors.invoke"
    ]
    assert query["id_token_add_organizations"] == ["true"]
    assert query["codex_cli_simplified_flow"] == ["true"]
    assert query["originator"] == ["llm_client"]


def test_default_client_explains_unconfigured_codex_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CLIENT_CODEX_AUTH_FILE", str(tmp_path / "missing.json"))
    with Client(transport=httpx.MockTransport(lambda _request: None)) as client:
        with pytest.raises(OAuthError, match="llm-client auth login codex"):
            client.model("codex/gpt-test").generate("hello")
