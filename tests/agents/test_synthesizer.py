"""Tests for AnswerSynthesizer agent — wraps qa.answer.synthesize + vault write."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.synthesizer import AnswerSynthesizer
from paper_distiller.distill.article import ArticleResult
from paper_distiller.qa.state import SessionState


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _state(articles):
    return SessionState(
        session_id="sid-1", question="why diffusion?",
        config_snapshot={}, started_at="2026-05-19T10:00:00",
        articles_distilled=articles,
        stop_reason="llm_done",
        rounds_completed=2,
        cost_cny=1.5,
        tokens_in_total=1000, tokens_out_total=500,
    )


def _ctx(state, **cfg_overrides):
    cfg = SimpleNamespace(qa_question=state.question, **cfg_overrides)
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"qa_state": state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_synthesizer_calls_synthesize_and_saves(mocker):
    arts = [_article("a"), _article("b")]
    state = _state(arts)
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA: 答案", "body": "# 答案\n\n...",
            "tags": ["qa"], "cited_slugs": ["a", "b"],
        },
    )
    ctx = _ctx(state)
    ctx.vault.save_entry = MagicMock(side_effect=lambda **kw: {"slug": kw["slug"]})
    out = await AnswerSynthesizer().run(ctx)
    ctx.vault.save_entry.assert_called_once()
    call_kw = ctx.vault.save_entry.call_args.kwargs
    assert call_kw["category"] == "surveys"
    assert call_kw["slug"].startswith("qa-")
    assert "answer_survey_slug" in out


@pytest.mark.asyncio
async def test_synthesizer_skipped_when_no_articles():
    state = _state(articles=[])
    ctx = _ctx(state)
    ctx.vault.save_entry = MagicMock()
    out = await AnswerSynthesizer().run(ctx)
    ctx.vault.save_entry.assert_not_called()
    assert out["answer_survey_slug"] is None


@pytest.mark.asyncio
async def test_synthesizer_appends_audit_trail_to_body(mocker):
    """Final body contains the LLM answer + an audit trail section."""
    arts = [_article("a")]
    state = _state(arts)
    state.history = []
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={"title": "QA: x", "body": "# answer", "tags": [], "cited_slugs": ["a"]},
    )
    ctx = _ctx(state)
    saved_body = {}
    def _capture_save(**kw):
        saved_body["body"] = kw["body"]
        return {"slug": kw["slug"]}
    ctx.vault.save_entry = MagicMock(side_effect=_capture_save)
    await AnswerSynthesizer().run(ctx)
    assert "# answer" in saved_body["body"]
    assert "audit trail" in saved_body["body"].lower() or "研究过程" in saved_body["body"]


@pytest.mark.asyncio
async def test_synthesizer_deps():
    assert AnswerSynthesizer().deps == ["vault-writer"]
