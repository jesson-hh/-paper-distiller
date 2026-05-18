import json
from unittest.mock import MagicMock

from paper_distiller.sources.arxiv import ArxivPaper
from paper_distiller.vault.crosslink import WikiIndex
from paper_distiller.distill.article import distill, _scrub_invented_links


def _paper():
    return ArxivPaper(
        source="arxiv", paper_id="2501.00001",
        arxiv_id="2501.00001", title="Test Paper",
        authors=["A", "B"], abstract="abstract text",
        pdf_url="", published="2025-01-01", categories=["math.AT"],
    )


def _index_with(slugs):
    return WikiIndex(entries=[
        {"category": "articles", "slug": s, "title": s, "tags": []}
        for s in slugs
    ])


def test_distill_returns_article_result():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "测试论文",
        "body": "# 测试\n\n## 一句话\n这是测试。",
        "tags": ["test"],
        "refs": ["arxiv:2501.00001"],
    })
    result = distill(_paper(), "x" * 600, _index_with([]), llm)  # 600 chars > 500 threshold
    assert result.slug == "ce-shi-lun-wen" or result.slug.startswith("entry-")  # CJK fallback
    assert result.title == "测试论文"
    assert "测试" in result.body
    assert result.tags == ["test"]
    assert result.refs == ["arxiv:2501.00001"]
    assert result.depth == "full-pdf"


def test_distill_marks_abstract_only_when_no_full_text():
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "T", "body": "# T\nbody", "tags": [], "refs": [],
    })
    result = distill(_paper(), full_text="", wiki_index=_index_with([]), llm=llm)
    assert result.depth == "abstract-only"


def test_distill_falls_back_to_abstract_for_short_extract():
    """If full_text is shorter than 500 chars (e.g. PyMuPDF returned garbage
    from a scanned/corrupt PDF), depth_mode should be 'abstract-only' so the
    article body gets the ⚠️ callout and the LLM uses the abstract instead."""
    llm = MagicMock()
    llm.complete.return_value = json.dumps({
        "title": "T", "body": "# T\nbody", "tags": [], "refs": [],
    })
    result = distill(
        _paper(),
        full_text="50 chars of garbage from a corrupt PDF",  # 38 chars, well below 500
        wiki_index=_index_with([]),
        llm=llm,
    )
    assert result.depth == "abstract-only"


def test_scrub_invented_links_strips_unknown_slugs():
    valid_slugs = {"foo", "bar"}
    body = "see [[foo]] and [[unknown-slug|Alias]] and [[bar|B]]"
    cleaned = _scrub_invented_links(body, valid_slugs)
    assert "[[foo]]" in cleaned
    assert "[[bar|B]]" in cleaned
    assert "[[unknown-slug|Alias]]" not in cleaned
    assert "Alias" in cleaned  # display text preserved
