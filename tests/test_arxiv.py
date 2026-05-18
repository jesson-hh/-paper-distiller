"""Tests for the arxiv source. Uses pytest-mock to stub the arxiv lib."""
from pathlib import Path
from unittest.mock import MagicMock

from paper_distiller.sources.arxiv import ArxivPaper, search, download_pdf


def _fake_result(arxiv_id, title, abstract):
    r = MagicMock()
    r.entry_id = f"http://arxiv.org/abs/{arxiv_id}v1"
    r.title = title
    r.summary = abstract
    r.authors = [MagicMock(name=f"A{i}") for i in range(2)]
    for i, a in enumerate(r.authors):
        a.name = f"Author{i}"
    r.published.isoformat.return_value = "2024-01-01T00:00:00+00:00"
    r.pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    r.categories = ["math.AT"]
    return r


def test_search_returns_arxiv_papers(mocker):
    mock_client = mocker.patch("paper_distiller.sources.arxiv.arxiv.Client")
    mock_client.return_value.results.return_value = iter([
        _fake_result("2501.00001", "Paper One", "Abstract one."),
        _fake_result("2501.00002", "Paper Two", "Abstract two."),
    ])
    papers = search("test query", max_results=5)
    assert len(papers) == 2
    assert isinstance(papers[0], ArxivPaper)
    assert papers[0].arxiv_id == "2501.00001"
    assert papers[0].title == "Paper One"


def test_download_pdf_writes_file(mocker, tmp_path):
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = [b"PDFCONTENT"]
    mock_response.raise_for_status = MagicMock()
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    mocker.patch("paper_distiller.sources.arxiv.httpx.stream",
                 return_value=mock_response)

    paper = ArxivPaper(
        source="arxiv",
        paper_id="2501.00001",
        arxiv_id="2501.00001",
        title="t", authors=[],
        abstract="a", pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
        published="2024-01-01", categories=[],
    )
    dest = download_pdf(paper, tmp_path)
    assert dest.exists()
    assert dest.suffix == ".pdf"
    assert dest.read_bytes() == b"PDFCONTENT"
