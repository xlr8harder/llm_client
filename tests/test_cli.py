import json
import time

from llm_client.cli import main
from llm_client.codex_oauth import CodexOAuthManager
from llm_client.oauth import OAuthCredentials


def test_auth_status_and_logout_do_not_print_tokens(tmp_path, monkeypatch, capsys):
    path = tmp_path / "codex.json"
    monkeypatch.setenv("LLM_CLIENT_CODEX_AUTH_FILE", str(path))
    manager = CodexOAuthManager.create()
    manager.store.save(
        OAuthCredentials("access-secret", "refresh-secret", time.time() + 3600)
    )
    manager.close()

    assert main(["--json", "auth", "status", "codex"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configured"] is True
    assert "secret" not in json.dumps(payload)

    assert main(["auth", "logout", "codex"]) == 0
    assert not path.exists()
    assert main(["auth", "status", "codex"]) == 1


def test_version_command_supports_json(capsys):
    assert main(["--json", "version"]) == 0
    assert json.loads(capsys.readouterr().out)["version"]
