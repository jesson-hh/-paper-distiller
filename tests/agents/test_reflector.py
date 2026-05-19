"""Tests for ProgressReflector agent — wraps qa.reflection.reflect."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.reflector import ProgressReflector
from paper_distiller.qa.state import SessionState


def _state():
    return SessionState(
        session_id="sid-1", question="why diffusion?",
        config_snapshot={}, started_at="2026-05-19T10:00:00",
    )


def _ctx(state, **cfg_overrides):
    cfg = SimpleNamespace(qa_max_rounds=3, qa_question="why diffusion?", **cfg_overrides)
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"qa_state": state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_reflector_calls_reflect_with_state_inputs(mocker):
    state = _state()
    state.history = []
    state.articles_distilled = []
    fake_reflect = mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": False, "confidence": 5,
            "what_we_know": "...", "what_is_missing": "...",
            "next_query": "diffusion convergence 2024",
            "next_query_rationale": "...", "suggest_stop": False,
        },
    )
    ctx = _ctx(state)
    out = await ProgressReflector().run(ctx)
    fake_reflect.assert_called_once()
    call = fake_reflect.call_args
    args_and_kw = list(call.args) + list(call.kwargs.values())
    assert "why diffusion?" in args_and_kw  # question passed
    assert 3 in args_and_kw  # max_rounds passed
    assert out["reflection"]["next_query"] == "diffusion convergence 2024"


@pytest.mark.asyncio
async def test_reflector_passes_prior_queries_from_history(mocker):
    state = _state()
    from paper_distiller.qa.state import RoundRecord
    state.history = [
        RoundRecord(round=1, query="q1", rationale="", candidates_found=3,
                    new_articles=2, article_slugs=[], what_we_know="",
                    what_is_missing="", confidence=4, timestamp="..."),
    ]
    fake_reflect = mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": False, "confidence": 6,
            "what_we_know": "", "what_is_missing": "",
            "next_query": "q2", "next_query_rationale": "", "suggest_stop": False,
        },
    )
    ctx = _ctx(state)
    await ProgressReflector().run(ctx)
    call = fake_reflect.call_args
    found = False
    for a in list(call.args) + list(call.kwargs.values()):
        if isinstance(a, list) and "q1" in a:
            found = True
    assert found, "prior_queries should include 'q1'"


@pytest.mark.asyncio
async def test_reflector_deps():
    assert ProgressReflector().deps == []
