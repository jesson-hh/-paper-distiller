"""Tests for REPL.dispatch_one — single-input dispatch logic, no actual stdin/stdout."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_repl_dispatch_quit_returns_quit_sentinel(tmp_path):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    assert r.dispatch_one("/quit") == "QUIT"


def test_repl_dispatch_help_prints_commands(tmp_path, capsys):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("/help")
    captured = capsys.readouterr()
    assert "/distill" in captured.out
    assert "/ask" in captured.out


def test_repl_dispatch_vault_runs_handler(tmp_path, capsys):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("/vault")
    captured = capsys.readouterr()
    assert "articles:" in captured.out


def test_repl_dispatch_unknown_slash_prints_error(tmp_path, capsys):
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("/nosuchcmd")
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).lower()
    assert "unknown" in combined


def test_repl_dispatch_natural_language_uses_router(mocker, tmp_path, capsys, monkeypatch):
    """NL input → IntentRouter.classify → slot-fill prompts → user cancels → no action."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_router_class = mocker.patch("paper_distiller.chat.repl.loop.IntentRouter")
    fake_router_class.return_value.classify.return_value = {
        "command": "ask",
        "params": {"question": "why diffusion?"},
        "missing_params": ["max_rounds", "per_round", "max_cost_cny"],
        "confidence": 8,
    }
    # Mock the slot-fill input() → "" (use default for each slot)
    mocker.patch("builtins.input", return_value="")
    # Mock final confirm to False (cancel before dispatch)
    mocker.patch("paper_distiller.chat.repl.loop._confirm", return_value=False)
    mocker.patch("paper_distiller.chat.repl.loop.LLMClient")
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("why diffusion?")
    captured = capsys.readouterr()
    assert "Intent: ask" in captured.out
    assert "question" in captured.out.lower()
    # Cancellation should be printed
    assert "cancelled" in captured.out.lower()


def test_repl_dispatch_natural_language_slot_filling(mocker, tmp_path, capsys, monkeypatch):
    """NL → router → interactive slot-fill (user types values) → confirm → dispatch."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_router_class = mocker.patch("paper_distiller.chat.repl.loop.IntentRouter")
    fake_router_class.return_value.classify.return_value = {
        "command": "ask",
        "params": {"question": "why diffusion?"},
        "missing_params": ["max_rounds", "per_round", "max_cost_cny"],
        "confidence": 8,
    }
    # User types: 5 (max_rounds), "" use default (per_round=2), 10.0 (max_cost)
    mocker.patch("builtins.input", side_effect=["5", "", "10.0"])
    mocker.patch("paper_distiller.chat.repl.loop._confirm", return_value=True)
    mocker.patch("paper_distiller.chat.repl.loop.LLMClient")
    fake_cli_main = mocker.patch("paper_distiller.chat.cli.main", return_value=0)
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("why diffusion?")
    # cli.main should have been called with the user-supplied values
    fake_cli_main.assert_called_once()
    argv = fake_cli_main.call_args.args[0]
    assert "--max-rounds" in argv
    assert "5" in argv
    assert "--per-round" in argv
    assert "2" in argv  # default used
    assert "--max-cost-cny" in argv
    assert "10.0" in argv


def test_repl_dispatch_nl_show_routes_to_handle_show(mocker, tmp_path, capsys, monkeypatch):
    """NL classified as 'show' should NOT go through cli.main (no show subcommand);
    instead, route directly to handle_show and display the article."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    # Set up a vault with an article so handle_show finds it
    (tmp_path / "articles").mkdir()
    (tmp_path / "articles" / "myslug.md").write_text(
        "---\ntitle: My Article\n---\n\n# My Article\n\nMy content body.",
        encoding="utf-8",
    )

    fake_router_class = mocker.patch("paper_distiller.chat.repl.loop.IntentRouter")
    fake_router_class.return_value.classify.return_value = {
        "command": "show",
        "params": {"slug": "myslug"},
        "missing_params": [],
        "confidence": 9,
    }
    # Confirm = True so we proceed past the confirmation gate
    mocker.patch("paper_distiller.chat.repl.loop._confirm", return_value=True)
    mocker.patch("paper_distiller.chat.repl.loop.LLMClient")
    # Sanity: if the fix is missing, cli.main would be invoked and would SystemExit
    fake_cli_main = mocker.patch("paper_distiller.chat.cli.main")

    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("看看 myslug")

    captured = capsys.readouterr()
    # The article content should be printed
    assert "My Article" in captured.out
    assert "My content body." in captured.out
    # cli.main MUST NOT have been called (the bug was that it WAS called for show)
    fake_cli_main.assert_not_called()
