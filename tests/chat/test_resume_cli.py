"""Tests for paper-distiller-chat 'resume' subcommand."""
import json
from unittest.mock import MagicMock

import pytest


def test_resume_cli_parses_args():
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args(["resume", "--vault", "/tmp/v", "--session-id", "sid-abc"])
    assert args.subcommand == "resume"
    assert args.vault == "/tmp/v"
    assert args.session_id == "sid-abc"


def test_resume_cli_dispatches_with_session_id(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    # Pre-seed a real state.json so read_state returns a SessionState
    vault = tmp_path
    sessions_dir = vault / ".paper_distiller" / "qa-sessions" / "sid-abc"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "state.json").write_text(json.dumps({
        "session_id": "sid-abc", "question": "Q?", "config_snapshot": {},
        "started_at": "2026-05-19T10:00:00", "rounds_completed": 1,
        "articles_distilled": [], "articles_seen_ids": [], "history": [],
        "last_reflection": None, "cost_cny": 0.0,
        "tokens_in_total": 0, "tokens_out_total": 0,
        "is_done": False, "stop_reason": "user_quit",
    }), encoding="utf-8")

    fake_run = mocker.patch("paper_distiller.chat.cli.run_qa_loop")
    fake_run.return_value = {
        "session_id": "sid-abc", "stop_reason": "llm_done",
        "rounds_completed": 3, "articles_distilled_count": 5,
        "cost_cny": 1.0, "tokens_in_total": 2000, "tokens_out_total": 800,
    }
    from paper_distiller.chat.cli import main
    rc = main(["resume", "--vault", str(vault), "--session-id", "sid-abc"])
    assert rc == 0
    fake_run.assert_called_once()
    cfg = fake_run.call_args[0][0]
    assert cfg.qa_resume_session_id == "sid-abc"
