"""Tests for CandidateMerger + CandidateRanker agents."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.curation import CandidateMerger, CandidateRanker
from paper_distiller.sources.arxiv import Paper


def _paper(pid, doi=None):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid, doi=doi,
        title=f"P{pid}", authors=[], abstract="...",
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx(**shared):
    """Note: cfg.topic defaults to None; tests that exercise ranker set it as needed."""
    return Context(
        cfg=SimpleNamespace(top_n=2, qa_per_round=None, topic=None),
        llm=MagicMock(), vault=MagicMock(),
        shared=dict(shared),
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_merger_combines_two_sources_and_dedups():
    a = [_paper("X1"), _paper("X2")]
    b = [_paper("X2"), _paper("X3")]  # X2 duplicates
    ctx = _ctx(candidates_arxiv=a, candidates_ss=b)
    out = await CandidateMerger().run(ctx)
    ids = [p.arxiv_id for p in out["candidates"]]
    # X2 dedup'd; arxiv wins on tie so X2 stays from `a`
    assert ids == ["X1", "X2", "X3"]


@pytest.mark.asyncio
async def test_merger_handles_empty_sources():
    ctx = _ctx(candidates_arxiv=[], candidates_ss=[])
    out = await CandidateMerger().run(ctx)
    assert out["candidates"] == []


@pytest.mark.asyncio
async def test_merger_deps():
    assert set(CandidateMerger().deps) == {
        "arxiv-searcher", "ss-searcher", "openalex-searcher",
    }


@pytest.mark.asyncio
async def test_merger_combines_three_sources():
    """When all 3 source lists are populated, merger should combine them."""
    a = [_paper("X1")]  # arxiv
    b = [_paper("Y1")]  # ss
    c = [_paper("Z1")]  # openalex
    ctx = _ctx(candidates_arxiv=a, candidates_ss=b, candidates_openalex=c)
    out = await CandidateMerger().run(ctx)
    ids = sorted(p.arxiv_id for p in out["candidates"])
    assert "X1" in ids
    assert "Y1" in ids
    assert "Z1" in ids


@pytest.mark.asyncio
async def test_merger_bypass_mode_uses_candidates_direct():
    """When shared['candidates_direct'] is set, merger short-circuits."""
    a = [_paper("X1"), _paper("X2")]
    direct = [_paper("D1"), _paper("D2"), _paper("D3")]
    # candidates_arxiv is set BUT candidates_direct should take precedence
    ctx = _ctx(candidates_arxiv=a, candidates_ss=[], candidates_direct=direct)
    out = await CandidateMerger().run(ctx)
    ids = [p.arxiv_id for p in out["candidates"]]
    assert ids == ["D1", "D2", "D3"]


@pytest.mark.asyncio
async def test_ranker_uses_top_n(mocker):
    candidates = [_paper(f"X{i}") for i in range(5)]
    fake_rank = mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    ctx = _ctx(candidates=candidates)
    ctx.cfg.top_n = 3
    ctx.cfg.qa_per_round = None  # single-pass uses top_n
    out = await CandidateRanker().run(ctx)
    assert len(out["ranked"]) == 3


@pytest.mark.asyncio
async def test_ranker_uses_qa_per_round_when_set(mocker):
    """In QA mode, qa_per_round overrides top_n."""
    candidates = [_paper(f"X{i}") for i in range(5)]
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    ctx = _ctx(candidates=candidates)
    ctx.cfg.top_n = 999
    ctx.cfg.qa_per_round = 2
    out = await CandidateRanker().run(ctx)
    assert len(out["ranked"]) == 2


@pytest.mark.asyncio
async def test_ranker_deps():
    assert CandidateRanker().deps == ["candidate-merger"]


@pytest.mark.asyncio
async def test_ranker_single_pass_uses_top_n_when_qa_per_round_is_none(mocker):
    """Regression: with qa_per_round=None (single-pass default), top_n must win."""
    candidates = [_paper(f"X{i}") for i in range(10)]
    captured = {}
    def _capture(candidates, topic, top_n, llm):
        captured["top_n"] = top_n
        return candidates[:top_n]
    mocker.patch("paper_distiller.agents.curation.rank", side_effect=_capture)
    ctx = _ctx(candidates=candidates)
    ctx.cfg.top_n = 7
    # qa_per_round already None from _ctx default
    await CandidateRanker().run(ctx)
    assert captured["top_n"] == 7


@pytest.mark.asyncio
async def test_ranker_returns_empty_for_no_candidates():
    """Short-circuit: no candidates → no LLM call, return empty."""
    ctx = _ctx()  # no candidates in shared
    out = await CandidateRanker().run(ctx)
    assert out == {"ranked": []}
