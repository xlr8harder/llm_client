"""Codex subscription provider using ChatGPT OAuth and the Responses API."""

from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict

import requests

from .openai_responses import OpenAIResponsesStyleProvider


CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_CLAIM = "https://api.openai.com/auth"

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


class CodexAuthManager:
    """Read and refresh Codex CLI OAuth credentials without exposing them."""

    def __init__(self, auth_path: str | Path | None = None, refresh_margin=120):
        self.auth_path = Path(auth_path).expanduser() if auth_path else self.default_auth_path()
        self.refresh_margin = int(refresh_margin)
        self.lock_path = self.auth_path.with_name(f".{self.auth_path.name}.llm-client.lock")

    @staticmethod
    def default_auth_path() -> Path:
        override = os.getenv("LLM_CLIENT_CODEX_AUTH_FILE")
        if override:
            return Path(override).expanduser()
        codex_home = Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser()
        return codex_home / "auth.json"

    def get_access(self) -> tuple[str, str]:
        if not self.auth_path.exists():
            raise ValueError(
                f"Codex OAuth credentials were not found at {self.auth_path}. Run 'codex login' first."
            )
        self.auth_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with _thread_lock(self.lock_path), self._process_lock():
            auth = self._read_auth()
            access, refresh, account_id = self._credential_fields(auth)
            if self._token_is_fresh(access):
                return access, account_id
            refreshed = self._refresh(refresh)

            # Codex itself does not use this lock. Avoid overwriting a newer login
            # or refresh that landed while the HTTP refresh was in flight.
            current = self._read_auth()
            current_access, current_refresh, current_account = self._credential_fields(current)
            if current_refresh != refresh and self._token_is_fresh(current_access):
                return current_access, current_account

            tokens = dict(current.get("tokens") or {})
            tokens["access_token"] = refreshed["access_token"]
            tokens["refresh_token"] = refreshed["refresh_token"]
            if isinstance(refreshed.get("id_token"), str):
                tokens["id_token"] = refreshed["id_token"]
            new_account = self._account_id(refreshed["access_token"]) or account_id
            tokens["account_id"] = new_account
            current["tokens"] = tokens
            current["last_refresh"] = datetime.now(timezone.utc).isoformat()
            self._write_auth(current)
            return refreshed["access_token"], new_account

    def _read_auth(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Codex auth file is invalid JSON: {self.auth_path}") from error
        except OSError as error:
            raise ValueError(f"Could not read Codex auth file {self.auth_path}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Codex auth file must contain a JSON object: {self.auth_path}")
        return payload

    def _credential_fields(self, auth: dict) -> tuple[str, str, str]:
        if auth.get("auth_mode") not in {None, "chatgpt"}:
            raise ValueError(
                "Codex is not logged in with ChatGPT OAuth. Run 'codex login' and choose ChatGPT."
            )
        tokens = auth.get("tokens")
        if not isinstance(tokens, dict):
            raise ValueError("Codex auth file does not contain OAuth tokens. Run 'codex login'.")
        access = tokens.get("access_token")
        refresh = tokens.get("refresh_token")
        if not isinstance(access, str) or not access:
            raise ValueError("Codex auth file is missing an access token. Run 'codex login'.")
        if not isinstance(refresh, str) or not refresh:
            raise ValueError("Codex auth file is missing a refresh token. Run 'codex login'.")
        account_id = self._account_id(access) or tokens.get("account_id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("Could not determine the ChatGPT account id from Codex OAuth credentials.")
        return access, refresh, account_id

    def _refresh(self, refresh_token: str) -> dict[str, str]:
        try:
            response = requests.post(
                CODEX_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                },
                timeout=30,
            )
        except requests.RequestException as error:
            raise ValueError(f"Codex OAuth refresh failed: {error}") from error
        if response.status_code != 200:
            raise ValueError(
                f"Codex OAuth refresh failed with HTTP {response.status_code}. Run 'codex login' again."
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise ValueError("Codex OAuth refresh returned invalid JSON.") from error
        access = payload.get("access_token")
        refresh = payload.get("refresh_token")
        if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
            raise ValueError("Codex OAuth refresh response was missing access or refresh tokens.")
        return payload

    def _token_is_fresh(self, token: str) -> bool:
        expiry = self._jwt_payload(token).get("exp")
        if not isinstance(expiry, (int, float)):
            return False
        return float(expiry) - datetime.now(timezone.utc).timestamp() > self.refresh_margin

    def _account_id(self, token: str) -> str | None:
        claim = self._jwt_payload(token).get(CODEX_AUTH_CLAIM)
        if not isinstance(claim, dict):
            return None
        account_id = claim.get("chatgpt_account_id")
        return account_id if isinstance(account_id, str) and account_id else None

    def _jwt_payload(self, token: str) -> dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        encoded = parts[1] + "=" * (-len(parts[1]) % 4)
        try:
            payload = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @contextmanager
    def _process_lock(self):
        self.lock_path.touch(mode=0o600, exist_ok=True)
        with self.lock_path.open("r+") as handle:
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
                import msvcrt

                if self.lock_path.stat().st_size == 0:
                    handle.write("0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    def _write_auth(self, payload: dict[str, Any]) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.auth_path.parent,
                prefix=f".{self.auth_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(payload, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.auth_path)
            os.chmod(self.auth_path, 0o600)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()


class CodexProvider(OpenAIResponsesStyleProvider):
    """Direct Codex Responses provider authenticated by the Codex CLI login."""

    api_base = CODEX_API_BASE
    api_key_env_var = "CODEX_ACCESS_TOKEN"
    provider_name = "codex"
    force_stream = True

    def __init__(self, auth_manager: CodexAuthManager | None = None):
        super().__init__()
        self.auth_manager = auth_manager or CodexAuthManager()
        self._account_id: str | None = None

    def get_api_key(self):
        access, account_id = self.auth_manager.get_access()
        self._account_id = account_id
        return access

    def _build_responses_headers(self) -> Dict[str, str]:
        access = self.get_api_key()
        return {
            "Authorization": f"Bearer {access}",
            "chatgpt-account-id": str(self._account_id),
            "originator": "llm_client",
            "User-Agent": "llm_client",
            "OpenAI-Beta": "responses=experimental",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

    def _normalize_responses_model_id(self, model_id: str) -> str:
        return model_id.removeprefix("codex/")

    def _customize_responses_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["store"] = False
        data["stream"] = True
        data.setdefault("instructions", "You are a helpful assistant.")
        data.setdefault("text", {"verbosity": "low"})
        data.setdefault("include", ["reasoning.encrypted_content"])
        return data
