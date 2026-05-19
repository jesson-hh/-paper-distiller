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
    """NL input → IntentRouter.classify → proposal print → user cancels → no action."""
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
    # Mock the confirmation prompt to return False (cancel)
    mocker.patch("paper_distiller.chat.repl.loop._confirm", return_value=False)
    mocker.patch("paper_distiller.chat.repl.loop.LLMClient")
    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    r.dispatch_one("why diffusion?")
    captured = capsys.readouterr()
    assert "Intent: ask" in captured.out
    assert "question" in captured.out.lower()
