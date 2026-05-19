"""Tests for chat.qa_runner — the QA loop driver. All subsystems mocked."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i, arxiv_id=None):
    aid = arxiv_id or f"2501.0000{i}"
    return Paper(
        source="arxiv", paper_id=aid, arxiv_id=aid,
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://x/{aid}.pdf", published="2025-01-01", categories=[],
    )


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _cfg(tmp_path, max_rounds=5, max_articles=15, max_cost=20.0, threshold=8, per_round=2):
    from paper_distiller.config import Config
    return Config(
        vault_path=tmp_path / "vault",
        topic=None, author=None,
        top_n=per_round, pool=10, force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
        source="arxiv", ss_api_key=None,
        qa_max_rounds=max_rounds, qa_max_articles=max_articles,
        qa_max_cost_cny=max_cost, qa_confidence_threshold=threshold,
        qa_per_round=per_round, qa_interactive=False,
        qa_resume_session_id=None, qa_question="why diffusion?",
    )


def _common_mocks(mocker, reflection_responses):
    """Mock all subsystems used by the QA loop's DAGs."""
    # Mock LLMClient at the qa_runner import site so it doesn't try to hit a real API
    fake_llm_class = mocker.patch("paper_distiller.chat.qa_runner.LLMClient")
    llm_instance = fake_llm_class.return_value
    llm_instance.total_tokens_in = 100
    llm_instance.total_tokens_out = 50
    # Reflection
    mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        side_effect=list(reflection_responses),
    )
    # Searchers
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(1), _paper(2), _paper(3)],
    )
    mocker.patch(
        "paper_distiller.agents.searchers.ss_search",
        return_value=[],
    )
    # Ranker
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    # PDF fetch + distill
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    def _make_article(paper, full_text, wiki_index, llm):
        from paper_distiller.distill.article import ArticleResult
        return ArticleResult(
            slug=f"a-{paper.arxiv_id}", title=f"T-{paper.arxiv_id}",
            body="b", tags=[], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=_make_article,
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    # Synthesizer
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA: answer", "body": "# answer\n\n...",
            "tags": ["qa"], "cited_slugs": [],
        },
    )


def test_qa_loop_stops_on_llm_done(tmp_path, mocker):
    """Reflection returns is_done=True with confidence >= threshold → stop."""
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "", "what_is_missing": "",
         "next_query": "q1", "next_query_rationale": "", "suggest_stop": False},
        {"is_done": True, "confidence": 9, "what_we_know": "all clear", "what_is_missing": "",
         "next_query": "", "next_query_rationale": "", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "llm_done"
    assert summary["rounds_completed"] == 1


def test_qa_loop_stops_on_max_rounds(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_rounds=2)
    cfg.vault_path.mkdir()
    not_done = {"is_done": False, "confidence": 4, "what_we_know": "a",
                "what_is_missing": "...", "next_query_rationale": "...", "suggest_stop": False}
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
        {**not_done, "next_query": "q3"},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "max_rounds"
    assert summary["rounds_completed"] == 2


def test_qa_loop_stops_on_no_candidates(tmp_path, mocker):
    """If all candidates were already seen, stop with no_candidates."""
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    not_done = {"is_done": False, "confidence": 4, "what_we_know": "", "what_is_missing": "",
                "next_query_rationale": "", "suggest_stop": False}
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
    ]
    _common_mocks(mocker, reflection_seq)
    # Override arxiv_search: always return same 2 papers (so round 2 sees nothing new)
    same_papers = [_paper(1), _paper(2)]
    mocker.patch("paper_distiller.agents.searchers.arxiv_search", return_value=same_papers)

    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "no_candidates"


def test_qa_loop_stops_on_max_articles(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_articles=2, per_round=2)
    cfg.vault_path.mkdir()
    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "a",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "max_articles"
    assert summary["articles_distilled_count"] == 2


def test_qa_loop_stops_on_llm_brake(tmp_path, mocker):
    """reflection.suggest_stop=True → llm_brake stop reason."""
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    reflection_seq = [
        {"is_done": False, "confidence": 3, "what_we_know": "", "what_is_missing": "",
         "next_query": "q1", "next_query_rationale": "", "suggest_stop": True},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "llm_brake"


def test_qa_loop_persists_state_each_round(tmp_path, mocker):
    """state.json appears under .paper_distiller/qa-sessions/<sid>/ after each round."""
    cfg = _cfg(tmp_path, max_rounds=1)
    cfg.vault_path.mkdir()
    not_done = {"is_done": False, "confidence": 4, "what_we_know": "a",
                "what_is_missing": "...", "next_query_rationale": "...", "suggest_stop": False}
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    sid = summary["session_id"]
    state_path = cfg.vault_path / ".paper_distiller" / "qa-sessions" / sid / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["stop_reason"] == "max_rounds"
