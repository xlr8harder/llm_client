"""Codex-specific OAuth configuration layered over the generic OAuth manager."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from .oauth import OAuthConfig, OAuthCredentialStore, OAuthError, OAuthManager


CODEX_AUTH_BASE = "https://auth.openai.com"
CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_AUTH_CLAIM = "https://api.openai.com/auth"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_SCOPES = (
    "openid",
    "profile",
    "email",
    "offline_access",
    "api.connectors.read",
    "api.connectors.invoke",
)


def codex_oauth_config(
    *,
    client_id: str = CODEX_CLIENT_ID,
    redirect_uri: str = "http://localhost:1455/auth/callback",
    auth_base: str = CODEX_AUTH_BASE,
) -> OAuthConfig:
    if not client_id:
        raise ValueError("Codex OAuth client_id must be provided explicitly")
    return OAuthConfig(
        provider="codex",
        client_id=client_id,
        authorization_url=f"{auth_base.rstrip('/')}/oauth/authorize",
        token_url=f"{auth_base.rstrip('/')}/oauth/token",
        scopes=CODEX_SCOPES,
        redirect_uri=redirect_uri,
        authorization_params={
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "llm_client",
        },
    )


class CodexOAuthManager(OAuthManager):
    """OAuth manager that supplies the headers required by the Codex backend."""

    @classmethod
    def create(
        cls,
        *,
        client_id: str = CODEX_CLIENT_ID,
        credential_path: str | Path | None = None,
        redirect_uri: str = "http://localhost:1455/auth/callback",
        auth_base: str = CODEX_AUTH_BASE,
        transport=None,
        refresh_margin: float = 120,
    ) -> CodexOAuthManager:
        path = credential_path or os.getenv(
            "LLM_CLIENT_CODEX_AUTH_FILE",
            str(Path.home() / ".config" / "llm_client" / "codex-oauth.json"),
        )
        return cls(
            codex_oauth_config(
                client_id=client_id, redirect_uri=redirect_uri, auth_base=auth_base
            ),
            OAuthCredentialStore(path),
            transport=transport,
            refresh_margin=refresh_margin,
        )

    def get_headers(self) -> dict[str, str]:
        if self.store.load() is None:
            raise OAuthError(
                "Codex is not configured. Run `llm-client auth login codex` before using a codex/... model."
            )
        access = self.get_access_token()
        account_id = _account_id(access)
        if not account_id:
            raise OAuthError(
                "Codex OAuth access token does not contain a ChatGPT account ID"
            )
        return {
            "Authorization": f"Bearer {access}",
            "chatgpt-account-id": account_id,
            "originator": "llm_client",
            "OpenAI-Beta": "responses=experimental",
        }


def _account_id(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    encoded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        payload: Any = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    claim = payload.get(CODEX_AUTH_CLAIM) if isinstance(payload, dict) else None
    value = claim.get("chatgpt_account_id") if isinstance(claim, dict) else None
    return value if isinstance(value, str) and value else None
