"""arxiv.org search + PDF download."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import arxiv
import httpx


@dataclass
class Paper:
    """A research paper, sourced from arxiv or Semantic Scholar.

    source: which API produced this record ("arxiv" or "semanticscholar")
    paper_id: canonical id within source (arxiv_id for arxiv, paperId for SS)
    arxiv_id / doi / ss_paper_id: cross-source identity (any may be None;
        at least one is always set)
    venue / open_access_pdf_url: SS-only enrichment (None when source="arxiv")
    """
    source: str
    paper_id: str
    title: str
    authors: list
    abstract: str
    published: str
    pdf_url: str
    categories: list = field(default_factory=list)

    arxiv_id: str | None = None
    doi: str | None = None
    ss_paper_id: str | None = None

    venue: str | None = None
    open_access_pdf_url: str | None = None


# Backward-compat alias — v0.2 imports of ArxivPaper continue to work.
ArxivPaper = Paper


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    return _client


def search(query: str, max_results: int = 30) -> list[ArxivPaper]:
    """Search arxiv.org. Returns up to max_results papers ranked by relevance."""
    s = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = _get_client()
    papers = []
    for result in client.results(s):
        arxiv_id = result.entry_id.rsplit("/", 1)[-1].split("v")[0]
        papers.append(Paper(
            source="arxiv",
            paper_id=arxiv_id,
            title=result.title.strip(),
            authors=[a.name for a in result.authors[:10]],
            abstract=result.summary.strip(),
            pdf_url=result.pdf_url,
            published=result.published.isoformat()[:10],
            categories=list(result.categories),
            arxiv_id=arxiv_id,
        ))
    return papers


def download_pdf(paper: ArxivPaper, dest_dir: Path, timeout: float = 60.0) -> Path:
    """Download paper PDF to dest_dir / <arxiv_id>.pdf. Returns the path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{paper.arxiv_id}.pdf"
    with httpx.stream("GET", paper.pdf_url, timeout=timeout, follow_redirects=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return dest
