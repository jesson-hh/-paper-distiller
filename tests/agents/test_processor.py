"""Tests for PaperProcessor fanout agent — one sub-instance per ranked paper."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.dag import DAG
from paper_distiller.agents.orchestrator import Orchestrator
from paper_distiller.agents.processor import PaperProcessor
from paper_distiller.sources.arxiv import Paper
from paper_distiller.distill.article import ArticleResult


class _StubRanker:
    """Stub upstream agent to satisfy PaperProcessor.deps in tests."""
    name = "candidate-ranker"
    deps: list[str] = []

    async def run(self, ctx):
        return {}


def _paper(pid):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid,
        title=f"P{pid}", authors=[], abstract="...",
        pdf_url=f"https://x/{pid}.pdf", published="2025-01-01",
        categories=[],
    )


def _ctx_with_ranked(papers, **cfg_overrides):
    cfg = SimpleNamespace(
        pdf_timeout_sec=60, verbose=False, source="both",
        **cfg_overrides,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"ranked": papers},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_processor_fans_out_one_subagent_per_paper(mocker):
    papers = [_paper("X1"), _paper("X2"), _paper("X3")]
    mocker.patch("paper_distiller.agents.processor.fetch_with_fallback", return_value="x" * 600)
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda paper, full_text, wiki_index, llm: ArticleResult(
            slug=f"a-{paper.arxiv_id}", title=f"T-{paper.arxiv_id}",
            body="b", tags=[], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        ),
    )
    mocker.patch("paper_distiller.agents.processor.load_index", return_value=MagicMock(slugs=lambda: set()))

    ctx = _ctx_with_ranked(papers)
    orch = Orchestrator(DAG([_StubRanker(), PaperProcessor()]), ctx)
    await orch.run()
    assert len(ctx.shared["articles"]) == 3
    assert {a.slug for a in ctx.shared["articles"]} == {"a-X1", "a-X2", "a-X3"}


@pytest.mark.asyncio
async def test_processor_handles_distill_failure_gracefully(mocker):
    """Per-paper distill failure does NOT abort the whole fanout — just drops that paper."""
    from paper_distiller.llm.openai_compatible import LLMError
    papers = [_paper("X1"), _paper("X2")]
    mocker.patch("paper_distiller.agents.processor.fetch_with_fallback", return_value="x" * 600)
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=[
            ArticleResult(slug="a-X1", title="T1", body="b", tags=[], refs=[], depth="full-pdf"),
            LLMError("LLM borked"),
        ],
    )
    mocker.patch("paper_distiller.agents.processor.load_index", return_value=MagicMock(slugs=lambda: set()))

    ctx = _ctx_with_ranked(papers)
    orch = Orchestrator(DAG([_StubRanker(), PaperProcessor()]), ctx)
    await orch.run()
    assert len(ctx.shared["articles"]) == 1
    assert ctx.shared["articles"][0].slug == "a-X1"


@pytest.mark.asyncio
async def test_processor_no_ranked_papers_is_noop():
    ctx = _ctx_with_ranked([])
    orch = Orchestrator(DAG([_StubRanker(), PaperProcessor()]), ctx)
    await orch.run()
    assert ctx.shared.get("articles", []) == []


@pytest.mark.asyncio
async def test_processor_deps():
    assert PaperProcessor().deps == ["candidate-ranker"]
