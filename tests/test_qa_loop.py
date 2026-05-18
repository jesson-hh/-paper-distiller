"""Integration tests for the QA loop. All subsystems mocked."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.config import Config
from paper_distiller.qa.loop import run
from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _config(tmp_path, max_rounds=5, max_articles=15, max_cost_cny=20.0,
            per_round=2, interactive=False, resume_id=None):
    return Config(
        vault_path=tmp_path / "vault",
        topic=None, author=None,
        top_n=per_round, pool=10, force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
        source="arxiv", ss_api_key=None,
        qa_max_rounds=max_rounds, qa_max_articles=max_articles,
        qa_max_cost_cny=max_cost_cny, qa_confidence_threshold=8,
        qa_per_round=per_round, qa_interactive=interactive,
        qa_resume_session_id=resume_id, qa_question="why diffusion?",
    )


def _paper(i, arxiv_id=None):
    aid = arxiv_id or f"2501.0000{i}"
    return Paper(
        source="arxiv", paper_id=aid, arxiv_id=aid,
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://arxiv.org/pdf/{aid}.pdf",
        published="2025-01-01", categories=[],
    )


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _common_mocks(mocker, reflection_responses,
                   candidates_seq=None, distill_factory=None):
    """Mock all the subsystems used by qa.loop.run."""
    llm_class = mocker.patch("paper_distiller.qa.loop.LLMClient")
    llm_instance = llm_class.return_value
    llm_instance.total_tokens_in = 100
    llm_instance.total_tokens_out = 50
    mocker.patch(
        "paper_distiller.qa.loop.reflect",
        side_effect=list(reflection_responses),
    )
    if candidates_seq is None:
        mocker.patch(
            "paper_distiller.qa.loop.gather_candidates",
            return_value=[_paper(1), _paper(2), _paper(3)],
        )
    else:
        mocker.patch(
            "paper_distiller.qa.loop.gather_candidates",
            side_effect=candidates_seq,
        )
    mocker.patch(
        "paper_distiller.qa.loop.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    mocker.patch(
        "paper_distiller.qa.loop.fetch_with_fallback",
        return_value="x" * 1000,
    )

    if distill_factory is None:
        def distill_factory(paper, full_text, wiki_index, llm):
            return _article(slug=f"a-{paper.arxiv_id}")
    mocker.patch(
        "paper_distiller.qa.loop.distill_article",
        side_effect=distill_factory,
    )
    mocker.patch(
        "paper_distiller.qa.loop.synthesize",
        return_value={
            "title": "QA: Answer",
            "body": "# QA\n\nAnswer body [[a-2501.00001]]",
            "tags": ["qa"],
            "cited_slugs": ["a-2501.00001"],
        },
    )


def test_loop_terminates_on_llm_done(tmp_path, mocker):
    """Loop exits when reflection.is_done=True with confidence >= threshold."""
    cfg = _config(tmp_path)
    cfg.vault_path.mkdir()

    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
        {"is_done": True, "confidence": 9, "what_we_know": "all clear",
         "what_is_missing": "", "next_query": "",
         "next_query_rationale": "", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)
    assert summary["stop_reason"] == "llm_done"
    # Round 1 reflection -> not done, distill 2 papers (rounds_completed=1)
    # Round 2 reflection -> done, stop
    assert summary["rounds_completed"] == 1


def test_loop_terminates_on_max_rounds(tmp_path, mocker):
    """Loop exits cleanly when max_rounds is hit, even if LLM says not done."""
    cfg = _config(tmp_path, max_rounds=2)
    cfg.vault_path.mkdir()

    # Need 3 reflections: round 1, round 2, then round 3 reflection (triggers max_rounds check)
    not_done = {
        "is_done": False, "confidence": 4, "what_we_know": "a",
        "what_is_missing": "...", "next_query_rationale": "...",
        "suggest_stop": False,
    }
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
        {**not_done, "next_query": "q3"},
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)
    assert summary["stop_reason"] == "max_rounds"
    assert summary["rounds_completed"] == 2


def test_loop_terminates_on_no_candidates(tmp_path, mocker):
    """If all candidates were already seen (full dedup), stop with no_candidates."""
    cfg = _config(tmp_path)
    cfg.vault_path.mkdir()

    not_done = {
        "is_done": False, "confidence": 4, "what_we_know": "...",
        "what_is_missing": "...", "next_query_rationale": "...",
        "suggest_stop": False,
    }
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
    ]
    # gather_candidates always returns the same 2 papers
    same_papers = [_paper(1), _paper(2)]
    _common_mocks(
        mocker,
        reflection_seq,
        candidates_seq=[same_papers, same_papers],
    )

    summary = run(cfg)
    assert summary["stop_reason"] == "no_candidates"


def test_loop_terminates_on_max_articles(tmp_path, mocker):
    """Loop exits when total distilled articles reach max_articles."""
    cfg = _config(tmp_path, max_articles=2, per_round=2)
    cfg.vault_path.mkdir()

    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "a",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)
    assert summary["stop_reason"] == "max_articles"
    assert summary["articles_distilled_count"] == 2


def test_loop_persists_state_each_round(tmp_path, mocker):
    """After each round, state.json is written under .paper_distiller/qa-sessions/<sid>/."""
    cfg = _config(tmp_path, max_rounds=1)
    cfg.vault_path.mkdir()

    not_done = {
        "is_done": False, "confidence": 4, "what_we_know": "a",
        "what_is_missing": "...", "next_query_rationale": "...",
        "suggest_stop": False,
    }
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
    ]
    _common_mocks(mocker, reflection_seq)

    summary = run(cfg)

    session_id = summary["session_id"]
    state_path = (cfg.vault_path / ".paper_distiller" / "qa-sessions"
                  / session_id / "state.json")
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["question"] == cfg.qa_question
    assert data["rounds_completed"] >= 1
    assert data["stop_reason"] == "max_rounds"
