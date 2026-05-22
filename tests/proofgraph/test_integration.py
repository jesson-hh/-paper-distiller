"""Integration tests for maybe_build_graph — PD_GRAPH_DEPTH gating."""
from __future__ import annotations

import pytest

from paper_distiller.proofgraph.pipeline import CoverageReport


def _fake_report(**kwargs):
    defaults = dict(
        segments_total=1, segments_processed=1, proof_blocks=0,
        nodes_by_kind={}, rejected_quotes=0, gaps=0, obligations=[],
    )
    defaults.update(kwargs)
    return CoverageReport(**defaults)


class _FakeProofStore:
    pass


class _FakeLLM:
    pass


def _noop_link_paper(*args, **kwargs):
    """Stub for link_paper that does nothing (no network, no store)."""
    from paper_distiller.proofgraph.linker import LinkReport
    return LinkReport()


# ---------------------------------------------------------------------------
# Tests for maybe_build_graph gating
# ---------------------------------------------------------------------------

def test_maybe_build_graph_returns_none_when_env_unset(monkeypatch):
    """PD_GRAPH_DEPTH unset → returns None, build_graph_for_paper NOT called."""
    monkeypatch.delenv("PD_GRAPH_DEPTH", raising=False)

    calls = []

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_report()

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _stub
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    result = maybe_build_graph(
        _FakeProofStore(), "1234.5678", "some full text",
        paper_slug="my-paper", llm=_FakeLLM(),
    )
    assert result is None
    assert calls == []


def test_maybe_build_graph_step_depth_calls_build(monkeypatch):
    """PD_GRAPH_DEPTH=step → build_graph_for_paper called once with depth='step'."""
    monkeypatch.setenv("PD_GRAPH_DEPTH", "step")

    calls = []
    expected_report = _fake_report(nodes_by_kind={"proof_step": 3})

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return expected_report

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _stub
    )
    monkeypatch.setattr(
        "paper_distiller.proofgraph.linker.link_paper", _noop_link_paper
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    store = _FakeProofStore()
    llm = _FakeLLM()
    result = maybe_build_graph(
        store, "1234.5678", "some full text",
        paper_slug="my-paper", llm=llm,
    )
    assert result is expected_report
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] is store
    assert args[1] == "1234.5678"
    assert args[2] == "some full text"
    assert kwargs["depth"] == "step"
    assert kwargs["paper_slug"] == "my-paper"
    assert kwargs["llm"] is llm


def test_maybe_build_graph_theorem_depth_calls_build(monkeypatch):
    """PD_GRAPH_DEPTH=theorem → build_graph_for_paper called once with depth='theorem'."""
    monkeypatch.setenv("PD_GRAPH_DEPTH", "theorem")

    calls = []

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_report()

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _stub
    )
    monkeypatch.setattr(
        "paper_distiller.proofgraph.linker.link_paper", _noop_link_paper
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    result = maybe_build_graph(
        _FakeProofStore(), "9999.0001", "text here",
        llm=_FakeLLM(),
    )
    assert isinstance(result, CoverageReport)
    assert len(calls) == 1
    assert calls[0][1]["depth"] == "theorem"


def test_maybe_build_graph_garbage_depth_returns_none(monkeypatch):
    """PD_GRAPH_DEPTH=garbage → treated as off; returns None, stub not called."""
    monkeypatch.setenv("PD_GRAPH_DEPTH", "garbage")

    calls = []

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_report()

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _stub
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    result = maybe_build_graph(
        _FakeProofStore(), "1234.5678", "text",
        llm=_FakeLLM(),
    )
    assert result is None
    assert calls == []


def test_maybe_build_graph_none_store_returns_none(monkeypatch):
    """proof_store=None → returns None, build_graph_for_paper not called."""
    monkeypatch.setenv("PD_GRAPH_DEPTH", "step")

    calls = []

    def _stub(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_report()

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _stub
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    result = maybe_build_graph(
        None, "1234.5678", "text",
        llm=_FakeLLM(),
    )
    assert result is None
    assert calls == []


def test_maybe_build_graph_swallows_exception(monkeypatch):
    """If build_graph_for_paper raises, maybe_build_graph returns None (best-effort)."""
    monkeypatch.setenv("PD_GRAPH_DEPTH", "step")

    def _exploding_stub(*args, **kwargs):
        raise RuntimeError("graph build exploded")

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _exploding_stub
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    result = maybe_build_graph(
        _FakeProofStore(), "1234.5678", "text",
        llm=_FakeLLM(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# New tests: link_paper wiring
# ---------------------------------------------------------------------------

def test_maybe_build_graph_calls_link_paper_once(monkeypatch):
    """After a successful build, link_paper is called once with store + paper_arxiv_id + llm."""
    monkeypatch.setenv("PD_GRAPH_DEPTH", "step")

    expected_report = _fake_report(nodes_by_kind={"proof_step": 2})

    def _build_stub(*args, **kwargs):
        return expected_report

    link_calls = []

    def _link_stub(store, paper_arxiv_id, llm, **kwargs):
        link_calls.append((store, paper_arxiv_id, llm))
        from paper_distiller.proofgraph.linker import LinkReport
        return LinkReport()

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _build_stub
    )
    monkeypatch.setattr(
        "paper_distiller.proofgraph.linker.link_paper", _link_stub
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    store = _FakeProofStore()
    llm = _FakeLLM()
    result = maybe_build_graph(store, "9876.5432", "full text here", llm=llm)

    assert result is expected_report
    assert len(link_calls) == 1
    called_store, called_id, called_llm = link_calls[0]
    assert called_store is store
    assert called_id == "9876.5432"
    assert called_llm is llm


def test_maybe_build_graph_link_paper_failure_still_returns_report(monkeypatch):
    """If link_paper raises, maybe_build_graph still returns the CoverageReport (best-effort)."""
    monkeypatch.setenv("PD_GRAPH_DEPTH", "step")

    expected_report = _fake_report(nodes_by_kind={"theorem": 1})

    def _build_stub(*args, **kwargs):
        return expected_report

    def _exploding_link(*args, **kwargs):
        raise RuntimeError("linker kaboom")

    monkeypatch.setattr(
        "paper_distiller.proofgraph.pipeline.build_graph_for_paper", _build_stub
    )
    monkeypatch.setattr(
        "paper_distiller.proofgraph.linker.link_paper", _exploding_link
    )

    from paper_distiller.proofgraph.pipeline import maybe_build_graph
    result = maybe_build_graph(_FakeProofStore(), "1111.2222", "text", llm=_FakeLLM())

    assert result is expected_report
