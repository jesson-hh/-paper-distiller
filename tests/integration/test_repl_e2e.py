"""End-to-end test for REPL: feed a sequence of inputs, verify behavior."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_repl_handles_help_then_vault_then_quit(tmp_path, mocker, capsys, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    # Set up a tiny vault
    (tmp_path / "articles").mkdir()
    (tmp_path / "articles" / "a.md").write_text(
        "---\ntitle: A\n---\n", encoding="utf-8",
    )

    from paper_distiller.chat.repl.loop import REPL
    r = REPL(vault_path=tmp_path)
    assert r.dispatch_one("/help") is None
    assert r.dispatch_one("/vault") is None
    assert r.dispatch_one("/quit") == "QUIT"

    captured = capsys.readouterr()
    assert "/distill" in captured.out  # from /help
    assert "articles: 1" in captured.out  # from /vault
