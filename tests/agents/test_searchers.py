"""Tests for ArxivSearcher + SemanticScholarSearcher agents — wrap existing source modules."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.searchers import ArxivSearcher, SemanticScholarSearcher
from paper_distiller.sources.arxiv import Paper


def _paper(arxiv_id):
    return Paper(
        source="arxiv", paper_id=arxiv_id, arxiv_id=arxiv_id,
        title=f"P{arxiv_id}", authors=[], abstract="...",
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx_with_topic(topic="diffusion models"):
    cfg = SimpleNamespace(
        topic=topic, author=None, pool=10, source="both",
        ss_api_key=None,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={}, on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_arxiv_searcher_writes_candidates_arxiv(mocker):
    fake_papers = [_paper("2501.00001"), _paper("2501.00002")]
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=fake_papers,
    )
    ctx = _ctx_with_topic()
    agent = ArxivSearcher()
    out = await agent.run(ctx)
    assert out["candidates_arxiv"] == fake_papers


@pytest.mark.asyncio
async def test_arxiv_searcher_uses_qa_next_query_when_present(mocker):
    """If shared['next_query'] is set (QA mode), it overrides cfg.topic."""
    fake_search = mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[],
    )
    ctx = _ctx_with_topic("ignored_topic")
    ctx.shared["next_query"] = "qa-mode-query"
    await ArxivSearcher().run(ctx)
    fake_search.assert_called_once()
    # Accept either positional or kwarg call style
    call_args = fake_search.call_args
    assert (call_args.kwargs.get("query") == "qa-mode-query") or (
        len(call_args.args) > 0 and call_args.args[0] == "qa-mode-query"
    )


@pytest.mark.asyncio
async def test_ss_searcher_writes_candidates_ss(mocker):
    fake_papers = [_paper("ss-1")]
    mocker.patch(
        "paper_distiller.agents.searchers.ss_search",
        return_value=fake_papers,
    )
    ctx = _ctx_with_topic()
    out = await SemanticScholarSearcher().run(ctx)
    assert out["candidates_ss"] == fake_papers


@pytest.mark.asyncio
async def test_searchers_have_no_deps():
    assert ArxivSearcher().deps == []
    assert SemanticScholarSearcher().deps == []


@pytest.mark.asyncio
async def test_searchers_skip_when_source_excludes_them(mocker):
    """If cfg.source == 'arxiv', SS searcher returns empty without calling the API."""
    fake_search = mocker.patch("paper_distiller.agents.searchers.ss_search")
    ctx = _ctx_with_topic()
    ctx.cfg.source = "arxiv"
    out = await SemanticScholarSearcher().run(ctx)
    fake_search.assert_not_called()
    assert out == {"candidates_ss": []}
