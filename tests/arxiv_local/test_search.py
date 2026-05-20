"""Tests for arxiv_local.search — FTS5 local query."""

from __future__ import annotations

import pytest


@pytest.fixture
def populated_store(tmp_path):
    from paper_distiller.arxiv_local.store import Store, PaperRow
    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([
        PaperRow(arxiv_id="2401.0", title="Diffusion Models for Image Generation",
                 authors=["Alice"], abstract="We propose latent diffusion models...",
                 categories=["cs.CV"], primary_category="cs.CV",
                 published="2024-01-15", updated=None, doi=None, comment=None,
                 journal_ref=None, source="bootstrap"),
        PaperRow(arxiv_id="2401.1", title="Transformer Architectures",
                 authors=["Bob"], abstract="Attention mechanism scaling...",
                 categories=["cs.LG"], primary_category="cs.LG",
                 published="2024-02-01", updated=None, doi=None, comment=None,
                 journal_ref=None, source="bootstrap"),
        PaperRow(arxiv_id="2401.2", title="Score-Based Generative Modeling",
                 authors=["Carol"], abstract="Diffusion processes for generation",
                 categories=["stat.ML", "cs.LG"], primary_category="stat.ML",
                 published="2024-03-01", updated=None, doi=None, comment=None,
                 journal_ref=None, source="bootstrap"),
    ])
    yield store
    store.close()


def test_search_returns_matching_papers(populated_store):
    from paper_distiller.arxiv_local.search import search

    results = search(populated_store, "diffusion", n=10)
    titles = [p.title for p in results]
    assert any("Diffusion" in t for t in titles)


def test_search_returns_paper_dataclass(populated_store):
    """Search must return objects compatible with sources.arxiv.Paper."""
    from paper_distiller.arxiv_local.search import search
    from paper_distiller.sources.arxiv import Paper

    results = search(populated_store, "diffusion", n=10)
    assert all(isinstance(p, Paper) for p in results)
    p = results[0]
    assert p.source == "arxiv"
    assert p.pdf_url
    assert p.arxiv_id


def test_search_sort_by_date(populated_store):
    from paper_distiller.arxiv_local.search import search

    results = search(populated_store, "diffusion OR transformer OR score",
                     n=10, sort="date")
    dates = [p.published for p in results]
    assert dates == sorted(dates, reverse=True)


def test_search_with_primary_category_filter(populated_store):
    from paper_distiller.arxiv_local.search import search

    results = search(
        populated_store, "diffusion", n=10, primary_category="stat.ML"
    )
    assert len(results) == 1
    assert results[0].arxiv_id == "2401.2"


def test_search_empty_query_returns_empty(populated_store):
    from paper_distiller.arxiv_local.search import search

    assert search(populated_store, "", n=10) == []
    assert search(populated_store, "   ", n=10) == []


def test_search_n_limits_result_count(populated_store):
    from paper_distiller.arxiv_local.search import search

    results = search(populated_store, "diffusion OR transformer OR score", n=2)
    assert len(results) <= 2


def test_search_no_match_returns_empty(populated_store):
    from paper_distiller.arxiv_local.search import search
    assert search(populated_store, "quantumchromodynamics", n=10) == []


def test_search_since_filter(populated_store):
    from paper_distiller.arxiv_local.search import search

    results = search(
        populated_store, "diffusion OR transformer OR score",
        n=10, since="2024-02-15",
    )
    arxiv_ids = {p.arxiv_id for p in results}
    assert arxiv_ids == {"2401.2"}


# ---------------------------------------------------------------------------
# Author search (v1.6.2)
# ---------------------------------------------------------------------------

def test_search_by_author_matches_in_authors_json(tmp_path):
    from paper_distiller.arxiv_local.search import search_by_author
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([
        PaperRow(arxiv_id="2401.0", title="Generative Models", authors=["Yuling Jiao", "Other"],
                 abstract="x", categories=["stat.ML"], primary_category="stat.ML",
                 published="2024-01-01", updated=None, doi=None, comment=None,
                 journal_ref=None, source="bootstrap"),
        PaperRow(arxiv_id="2401.1", title="Unrelated Paper", authors=["Jane Doe"],
                 abstract="y", categories=["cs.LG"], primary_category="cs.LG",
                 published="2024-02-01", updated=None, doi=None, comment=None,
                 journal_ref=None, source="bootstrap"),
    ])
    results = search_by_author(store, "Yuling Jiao", n=10)
    assert len(results) == 1
    assert results[0].arxiv_id == "2401.0"
    store.close()


def test_search_by_author_case_insensitive(tmp_path):
    from paper_distiller.arxiv_local.search import search_by_author
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([PaperRow(
        arxiv_id="x", title="t", authors=["Yuling Jiao"],
        abstract="x", categories=["x"], primary_category="x",
        published="2024-01-01", updated=None, doi=None, comment=None,
        journal_ref=None, source="bootstrap",
    )])
    assert len(search_by_author(store, "yuling jiao")) == 1
    assert len(search_by_author(store, "JIAO")) == 1
    store.close()


def test_looks_like_author_name():
    from paper_distiller.arxiv_local.search import _looks_like_author_name

    assert _looks_like_author_name("Yuling Jiao")
    assert _looks_like_author_name("Geoffrey Hinton")
    assert _looks_like_author_name("Yann LeCun")
    assert _looks_like_author_name("Jian-Wei Pan")  # hyphen ok
    assert not _looks_like_author_name("diffusion models")  # lowercase
    # "Diffusion Models" (both capitalized) WOULD match the heuristic — known
    # false positive. Acceptable since author fallback only fires when FTS
    # already returned < n results; cost is bounded.
    assert not _looks_like_author_name("a")  # single token
    assert not _looks_like_author_name("attention is all you need")  # too many tokens
    assert not _looks_like_author_name("")  # empty


def test_search_falls_back_to_author_when_fts_empty(tmp_path):
    """Search('Yuling Jiao') should find papers via authors fallback even
    if title/abstract don't mention 'Yuling Jiao'."""
    from paper_distiller.arxiv_local.search import search
    from paper_distiller.arxiv_local.store import Store, PaperRow

    store = Store(tmp_path / "arxiv.db")
    store.upsert_many([PaperRow(
        arxiv_id="2401.0", title="A Statistical Method", authors=["Yuling Jiao"],
        abstract="We propose a method.", categories=["stat.ML"],
        primary_category="stat.ML",
        published="2024-01-01", updated=None, doi=None, comment=None,
        journal_ref=None, source="bootstrap",
    )])
    results = search(store, "Yuling Jiao", n=10)
    assert len(results) == 1
    assert results[0].arxiv_id == "2401.0"
    store.close()
