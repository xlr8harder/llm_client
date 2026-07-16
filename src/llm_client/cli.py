from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import time
import webbrowser

from ._version import __version__
from .codex_oauth import CODEX_CLIENT_ID, CodexOAuthManager
from .oauth import OAuthError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-client")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("version", help="Show the installed version")

    auth = commands.add_parser("auth", help="Manage provider authentication")
    auth_commands = auth.add_subparsers(dest="auth_command", required=True)
    for name in ("status", "logout"):
        command = auth_commands.add_parser(name, help=f"{name.title()} provider auth")
        command.add_argument("provider", choices=["codex"])
    login = auth_commands.add_parser("login", help="Log in to a provider")
    login.add_argument("provider", choices=["codex"])
    login.add_argument("--client-id", default=CODEX_CLIENT_ID, help=argparse.SUPPRESS)
    login.add_argument(
        "--manual",
        action="store_true",
        help="Print the URL and paste the callback instead of opening a browser",
    )
    return parser


def _emit(payload: dict, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(payload["message"])


def _manager(*, client_id: str = CODEX_CLIENT_ID, redirect_uri: str | None = None):
    options = {"client_id": client_id}
    if redirect_uri is not None:
        options["redirect_uri"] = redirect_uri
    return CodexOAuthManager.create(**options)


def _auth_status(*, json_output: bool) -> int:
    manager = _manager()
    try:
        credentials = manager.store.load()
    finally:
        manager.close()
    if credentials is None:
        _emit(
            {
                "provider": "codex",
                "configured": False,
                "message": "Codex is not logged in.",
            },
            json_output=json_output,
        )
        return 1
    expired = credentials.expires_at <= time.time()
    _emit(
        {
            "provider": "codex",
            "configured": True,
            "expired": expired,
            "expires_at": credentials.expires_at,
            "message": "Codex login is stored" + (" but expired." if expired else "."),
        },
        json_output=json_output,
    )
    return 0


def _auth_logout(*, json_output: bool) -> int:
    manager = _manager()
    try:
        path = manager.store.path
        if path.exists():
            path.unlink()
    finally:
        manager.close()
    _emit(
        {"provider": "codex", "configured": False, "message": "Codex login removed."},
        json_output=json_output,
    )
    return 0


def _manual_login(*, client_id: str, json_output: bool) -> int:
    manager = _manager(client_id=client_id)
    try:
        login = manager.begin_login()
        print(login.url, file=sys.stderr if json_output else sys.stdout)
        callback = input("Paste the full redirected callback URL: ").strip()
        manager.complete_redirect(callback, login)
    finally:
        manager.close()
    _emit(
        {"provider": "codex", "configured": True, "message": "Codex login stored."},
        json_output=json_output,
    )
    return 0


def _browser_login(*, client_id: str, json_output: bool) -> int:
    callback: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            callback["url"] = f"http://localhost:{self.server.server_port}{self.path}"
            body = b"Codex login received. You can close this window."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = None
    for port in (1455, 1457):
        try:
            server = HTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise OAuthError(
            "Codex login could not bind localhost ports 1455 or 1457. "
            "Stop the process using those ports or run with --manual."
        )
    manager = _manager(
        client_id=client_id,
        redirect_uri=f"http://localhost:{server.server_port}/auth/callback",
    )
    try:
        login = manager.begin_login()
        print(f"Opening {login.url}", file=sys.stderr if json_output else sys.stdout)
        webbrowser.open(login.url)
        server.timeout = 300
        server.handle_request()
        if "url" not in callback:
            raise OAuthError("Codex login timed out waiting for the browser callback")
        manager.complete_redirect(callback["url"], login)
    finally:
        server.server_close()
        manager.close()
    _emit(
        {"provider": "codex", "configured": True, "message": "Codex login stored."},
        json_output=json_output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "version":
            _emit(
                {"version": __version__, "message": __version__}, json_output=args.json
            )
            return 0
        if args.auth_command == "status":
            return _auth_status(json_output=args.json)
        if args.auth_command == "logout":
            return _auth_logout(json_output=args.json)
        if args.manual:
            return _manual_login(client_id=args.client_id, json_output=args.json)
        return _browser_login(client_id=args.client_id, json_output=args.json)
    except (OAuthError, OSError, ValueError) as error:
        if args.json:
            print(json.dumps({"error": str(error)}, separators=(",", ":")))
        else:
            print(f"llm-client: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
