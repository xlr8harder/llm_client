"""OAuth PKCE and refresh support with independently owned credential storage."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlencode
from urllib.parse import parse_qs, urlparse

import httpx


@dataclass(frozen=True)
class OAuthConfig:
    provider: str
    client_id: str
    authorization_url: str
    token_url: str
    scopes: tuple[str, ...]
    redirect_uri: str


@dataclass(frozen=True)
class OAuthLoginRequest:
    url: str
    state: str
    verifier: str
    redirect_uri: str


@dataclass(frozen=True)
class OAuthCredentials:
    access_token: str
    refresh_token: str
    expires_at: float
    token_type: str = "Bearer"
    id_token: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthCredentials:
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=float(data["expires_at"]),
            token_type=data.get("token_type", "Bearer"),
            id_token=data.get("id_token"),
            metadata=data.get("metadata"),
        )


class OAuthError(RuntimeError):
    pass


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


class OAuthCredentialStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")

    def load(self) -> OAuthCredentials | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return OAuthCredentials.from_dict(data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise OAuthError(f"OAuth credential store is invalid: {self.path}") from exc

    def save(self, credentials: OAuthCredentials) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(credentials.to_dict(), handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_path.touch(mode=0o600, exist_ok=True)
        with _thread_lock(self.lock_path), self.lock_path.open("r+") as handle:
            try:
                import fcntl
            except ImportError:
                fcntl = None
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                yield


class OAuthManager:
    def __init__(
        self,
        config: OAuthConfig,
        store: OAuthCredentialStore,
        *,
        transport: httpx.BaseTransport | None = None,
        refresh_margin: float = 120,
    ):
        self.config = config
        self.store = store
        self.refresh_margin = refresh_margin
        self._http = httpx.Client(transport=transport, timeout=30)

    def close(self):
        self._http.close()

    def begin_login(
        self, *, state: str | None = None, verifier: str | None = None
    ) -> OAuthLoginRequest:
        verifier = verifier or secrets.token_urlsafe(64)
        state = state or secrets.token_urlsafe(32)
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        return OAuthLoginRequest(
            url=f"{self.config.authorization_url}?{urlencode(params)}",
            state=state,
            verifier=verifier,
            redirect_uri=self.config.redirect_uri,
        )

    def exchange_code(self, code: str, login: OAuthLoginRequest) -> OAuthCredentials:
        if not code:
            raise OAuthError("OAuth authorization code is empty")
        credentials = self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.config.client_id,
                "redirect_uri": login.redirect_uri,
                "code_verifier": login.verifier,
            },
            previous=None,
        )
        with self.store.locked():
            self.store.save(credentials)
        return credentials

    def complete_redirect(
        self, redirect_url: str, login: OAuthLoginRequest
    ) -> OAuthCredentials:
        parsed = urlparse(redirect_url)
        query = parse_qs(parsed.query)
        returned_state = query.get("state", [None])[0]
        if not returned_state or not secrets.compare_digest(
            returned_state, login.state
        ):
            raise OAuthError("OAuth callback state did not match the login request")
        oauth_error = query.get("error", [None])[0]
        if oauth_error:
            raise OAuthError(
                f"{self.config.provider} OAuth login was rejected: {oauth_error}"
            )
        code = query.get("code", [None])[0]
        if not code:
            raise OAuthError("OAuth callback did not contain an authorization code")
        return self.exchange_code(code, login)

    def get_access_token(self) -> str:
        credentials = self.store.load()
        if credentials is None:
            raise OAuthError(
                f"No {self.config.provider} OAuth login is stored; run the provider login command"
            )
        if credentials.expires_at - time.time() > self.refresh_margin:
            return credentials.access_token
        with self.store.locked():
            credentials = self.store.load()
            if credentials is None:
                raise OAuthError(f"No {self.config.provider} OAuth login is stored")
            if credentials.expires_at - time.time() > self.refresh_margin:
                return credentials.access_token
            refreshed = self._token_request(
                {
                    "grant_type": "refresh_token",
                    "refresh_token": credentials.refresh_token,
                    "client_id": self.config.client_id,
                },
                previous=credentials,
            )
            self.store.save(refreshed)
            return refreshed.access_token

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.get_access_token()}"}

    def _token_request(
        self, data: dict[str, str], previous: OAuthCredentials | None
    ) -> OAuthCredentials:
        try:
            response = self._http.post(self.config.token_url, data=data)
        except httpx.HTTPError as exc:
            raise OAuthError(
                f"{self.config.provider} OAuth token request failed: {type(exc).__name__}"
            ) from exc
        if response.status_code != 200:
            raise OAuthError(
                f"{self.config.provider} OAuth token request failed with HTTP {response.status_code}; log in again"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OAuthError(
                f"{self.config.provider} OAuth token response was invalid JSON"
            ) from exc
        access = payload.get("access_token")
        refresh = payload.get("refresh_token") or (
            previous.refresh_token if previous else None
        )
        if (
            not isinstance(access, str)
            or not access
            or not isinstance(refresh, str)
            or not refresh
        ):
            raise OAuthError(
                f"{self.config.provider} OAuth response omitted required tokens"
            )
        expires_in = payload.get("expires_in", 3600)
        try:
            expires_at = time.time() + float(expires_in)
        except (TypeError, ValueError) as exc:
            raise OAuthError(
                f"{self.config.provider} OAuth response has invalid expiry"
            ) from exc
        known = {
            "access_token",
            "refresh_token",
            "expires_in",
            "token_type",
            "id_token",
        }
        return OAuthCredentials(
            access_token=access,
            refresh_token=refresh,
            expires_at=expires_at,
            token_type=payload.get("token_type", "Bearer"),
            id_token=payload.get("id_token"),
            metadata={k: v for k, v in payload.items() if k not in known} or None,
        )


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
