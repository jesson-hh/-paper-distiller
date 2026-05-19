"""Tests for paper-distiller-chat CLI dispatch.

v1.4: no subcommand → v1.4 conversational agent (AgentLoop).
Legacy slash-command REPL accessible via `legacy-repl` subcommand.
"""


def test_chat_no_subcommand_launches_agent_loop(mocker, tmp_path, monkeypatch):
    """v1.4 default: no subcommand → conversational AgentLoop."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_agent = mocker.patch("paper_distiller.chat.cli.AgentLoop")
    fake_agent.return_value.run.return_value = 0
    # LLMClient does no I/O at init beyond API-key validation, but mock it to
    # avoid pulling in real env config dependencies.
    mocker.patch("paper_distiller.chat.cli.LLMClient")

    from paper_distiller.chat.cli import main
    rc = main(["--vault", str(tmp_path)])

    assert rc == 0
    fake_agent.assert_called_once()
    # vault_path must be threaded through.
    kwargs = fake_agent.call_args.kwargs
    assert kwargs["vault_path"] == str(tmp_path)


def test_chat_no_subcommand_no_vault_returns_error():
    from paper_distiller.chat.cli import main
    rc = main([])  # no --vault, no subcommand
    assert rc == 2


def test_chat_no_subcommand_missing_api_key_returns_error(tmp_path, monkeypatch):
    """Agent loop requires PD_API_KEY / PD_BASE_URL / PD_MODEL."""
    monkeypatch.delenv("PD_API_KEY", raising=False)
    monkeypatch.delenv("PD_BASE_URL", raising=False)
    monkeypatch.delenv("PD_MODEL", raising=False)
    from paper_distiller.chat.cli import main
    rc = main(["--vault", str(tmp_path)])
    assert rc == 2


def test_legacy_repl_subcommand_launches_old_repl(mocker, tmp_path):
    """Pre-v1.4 slash-command REPL is now opt-in via `legacy-repl`."""
    fake_repl = mocker.patch("paper_distiller.chat.cli.REPL")
    fake_repl.return_value.run.return_value = 0
    from paper_distiller.chat.cli import main
    rc = main(["legacy-repl", "--vault", str(tmp_path)])
    assert rc == 0
    fake_repl.assert_called_once_with(vault_path=str(tmp_path))
