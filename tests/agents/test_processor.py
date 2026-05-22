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
        side_effect=lambda paper, full_text, wiki_index, llm, prior_theorems=None: ArticleResult(
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


@pytest.mark.asyncio
async def test_processor_empty_ranked_preserves_prior_articles():
    """Regression: in QA mode, a round with zero ranked papers must
    NOT clobber `ctx.shared['articles']` accumulated from prior rounds."""
    prior = [ArticleResult(
        slug="prior", title="Prior", body="b",
        tags=[], refs=[], depth="full-pdf",
    )]
    ctx = _ctx_with_ranked([])
    ctx.shared["articles"] = list(prior)  # simulate prior rounds
    orch = Orchestrator(DAG([_StubRanker(), PaperProcessor()]), ctx)
    await orch.run()
    assert ctx.shared["articles"] == prior


# ---------------------------------------------------------------------------
# Task 4.2 wiring tests — _DistillOne calls maybe_build_graph
# ---------------------------------------------------------------------------

def _make_fake_article(arxiv_id="X1"):
    """Return a minimal ArticleResult with empty proof_sidecar (no theorems)."""
    return ArticleResult(
        slug=f"a-{arxiv_id}", title=f"T-{arxiv_id}",
        body="b", tags=[], refs=[f"arxiv:{arxiv_id}"],
        depth="full-pdf",
    )


@pytest.mark.asyncio
async def test_distill_one_calls_maybe_build_graph(mocker):
    """_DistillOne.run calls maybe_build_graph once with the fetched full_text
    and the paper's arxiv_id when a proof_store is present."""
    from paper_distiller.agents.processor import _DistillOne
    from paper_distiller.agents.base import Context

    fake_full_text = "x" * 600
    paper = _paper("WIRE1")
    fake_article = _make_fake_article("WIRE1")

    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value=fake_full_text,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        return_value=fake_article,
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )

    graph_calls = []

    def _graph_stub(proof_store, arxiv_id, full_text, *, paper_slug=None, llm=None):
        graph_calls.append({
            "proof_store": proof_store,
            "arxiv_id": arxiv_id,
            "full_text": full_text,
            "paper_slug": paper_slug,
            "llm": llm,
        })
        return None  # gated off; return value doesn't matter here

    mocker.patch(
        "paper_distiller.agents.processor.maybe_build_graph",
        side_effect=_graph_stub,
    )

    fake_proof_store = MagicMock()
    cfg = SimpleNamespace(pdf_timeout_sec=60, verbose=False, source="both")
    fake_llm = MagicMock()
    ctx = Context(
        cfg=cfg, llm=fake_llm, vault=MagicMock(),
        shared={"articles": []},
        on_status=lambda *a, **kw: None,
    )

    import tempfile
    tmpdir = tempfile.mkdtemp()
    dist = _DistillOne(paper, 0, 1, tmpdir, MagicMock(), fake_proof_store)
    await dist.run(ctx)

    assert len(graph_calls) == 1, f"Expected 1 call, got {len(graph_calls)}"
    call = graph_calls[0]
    assert call["proof_store"] is fake_proof_store
    assert call["arxiv_id"] == "WIRE1"
    assert call["full_text"] == fake_full_text
    assert call["llm"] is fake_llm


@pytest.mark.asyncio
async def test_distill_one_graph_failure_does_not_abort(mocker):
    """If maybe_build_graph raises, _DistillOne.run still completes and
    appends the article (best-effort graph build)."""
    from paper_distiller.agents.processor import _DistillOne

    fake_full_text = "y" * 600
    paper = _paper("WIRE2")
    fake_article = _make_fake_article("WIRE2")

    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value=fake_full_text,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        return_value=fake_article,
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    mocker.patch(
        "paper_distiller.agents.processor.maybe_build_graph",
        side_effect=RuntimeError("graph build exploded"),
    )

    cfg = SimpleNamespace(pdf_timeout_sec=60, verbose=False, source="both")
    ctx = Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"articles": []},
        on_status=lambda *a, **kw: None,
    )

    import tempfile
    tmpdir = tempfile.mkdtemp()
    dist = _DistillOne(paper, 0, 1, tmpdir, MagicMock(), MagicMock())
    result = await dist.run(ctx)

    # run() must complete; article must be in shared
    assert result == {}
    assert len(ctx.shared["articles"]) == 1
    assert ctx.shared["articles"][0].slug == "a-WIRE2"


@pytest.mark.asyncio
async def test_distill_one_no_proof_store_skips_graph(mocker):
    """When proof_store is None, maybe_build_graph is NOT called."""
    from paper_distiller.agents.processor import _DistillOne

    paper = _paper("WIRE3")
    fake_article = _make_fake_article("WIRE3")

    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="z" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        return_value=fake_article,
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )

    graph_calls = []

    def _graph_stub(*args, **kwargs):
        graph_calls.append((args, kwargs))
        return None

    mocker.patch(
        "paper_distiller.agents.processor.maybe_build_graph",
        side_effect=_graph_stub,
    )

    cfg = SimpleNamespace(pdf_timeout_sec=60, verbose=False, source="both")
    ctx = Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"articles": []},
        on_status=lambda *a, **kw: None,
    )

    import tempfile
    tmpdir = tempfile.mkdtemp()
    # proof_store=None
    dist = _DistillOne(paper, 0, 1, tmpdir, MagicMock(), None)
    await dist.run(ctx)

    assert graph_calls == [], (
        f"maybe_build_graph should NOT be called when proof_store=None; got {graph_calls}"
    )
