"""Local FTS5 search over the arxiv mirror."""

from __future__ import annotations

import json

from ..sources.arxiv import Paper
from .store import Store


_PDF_URL_FMT = "https://arxiv.org/pdf/{arxiv_id}"


def _normalize_fts_query(q: str) -> str:
    """Escape FTS5 syntax in a user query, preserving simple OR/AND/NOT usage.

    Strategy: split on whitespace; uppercase boolean tokens pass through;
    other tokens are wrapped in double quotes so any embedded FTS operators
    are treated as literals.
    """
    tokens = q.strip().split()
    out = []
    for t in tokens:
        if t.upper() in ("AND", "OR", "NOT"):
            out.append(t.upper())
        else:
            clean = t.replace('"', "")
            out.append(f'"{clean}"')
    return " ".join(out)


def _row_to_paper(row) -> Paper:
    return Paper(
        source="arxiv",
        paper_id=row["arxiv_id"],
        title=row["title"],
        authors=json.loads(row["authors"]),
        abstract=row["abstract"],
        pdf_url=_PDF_URL_FMT.format(arxiv_id=row["arxiv_id"]),
        published=row["published"],
        categories=json.loads(row["categories"]),
        arxiv_id=row["arxiv_id"],
        doi=row["doi"] or None,
    )


def search_by_author(
    store: Store,
    name: str,
    n: int = 30,
    sort: str = "date",
    since: str | None = None,
) -> list:
    """Find papers whose `authors` JSON contains `name` (case-insensitive).

    Uses SQL LIKE on the authors TEXT column. FTS5 only indexes title +
    abstract; this is the explicit author-search fallback.
    """
    if not name.strip():
        return []
    # Lowercase needle, lowercase haystack (collated by NOCASE in LIKE).
    # The JSON stored is `["First Last", ...]` so any substring match works.
    needle = f'%{name.strip()}%'
    order = "p.published DESC" if sort == "date" else "p.published DESC"
    parts = [
        "SELECT p.* FROM papers p",
        "WHERE p.authors LIKE ? COLLATE NOCASE",
    ]
    params: list = [needle]
    if since:
        parts.append("AND p.published >= ?")
        params.append(since)
    parts.append(f"ORDER BY {order}")
    parts.append("LIMIT ?")
    params.append(n)
    sql = " ".join(parts)
    rows = store._conn.execute(sql, params).fetchall()
    return [_row_to_paper(r) for r in rows]


def _looks_like_author_name(q: str) -> bool:
    """Heuristic: 2-4 capitalized words, no quotes/operators, ASCII letters.

    'Yuling Jiao'           → True
    'Geoffrey Hinton'        → True
    'diffusion models'       → False (lowercase)
    'attention is all you need' → False (5+ words)
    """
    s = q.strip()
    parts = s.split()
    if not (2 <= len(parts) <= 4):
        return False
    for p in parts:
        if not p:
            return False
        # First letter uppercase, no special chars
        if not p[0].isupper() or not all(c.isalpha() or c in "-." for c in p):
            return False
    return True


def search(
    store: Store,
    query: str,
    n: int = 30,
    sort: str = "relevance",
    primary_category: str | None = None,
    since: str | None = None,
) -> list:
    """Local FTS5 search over title + abstract. Falls back to author LIKE
    search when the query looks like a person's name and FTS gives no hits.
    Returns sources.arxiv.Paper objects.
    """
    if not query.strip():
        return []

    fts_query = _normalize_fts_query(query)

    sql_parts = [
        "SELECT p.* FROM papers p",
        "JOIN papers_fts ON papers_fts.rowid = p.rowid",
        "WHERE papers_fts MATCH ?",
    ]
    params: list = [fts_query]

    if primary_category:
        sql_parts.append("AND p.primary_category = ?")
        params.append(primary_category)

    if since:
        sql_parts.append("AND p.published >= ?")
        params.append(since)

    if sort == "date":
        sql_parts.append("ORDER BY p.published DESC")
    else:
        sql_parts.append("ORDER BY bm25(papers_fts)")

    sql_parts.append("LIMIT ?")
    params.append(n)

    sql = " ".join(sql_parts)
    rows = store._conn.execute(sql, params).fetchall()
    fts_results = [_row_to_paper(r) for r in rows]

    # Author fallback: when FTS gives few hits AND the query looks like a
    # person's name, also try authors LIKE — merge + dedupe.
    if len(fts_results) < n and _looks_like_author_name(query):
        author_results = search_by_author(
            store, query, n=n - len(fts_results),
            sort=sort if sort == "date" else "date",
            since=since,
        )
        seen_ids = {p.arxiv_id for p in fts_results}
        for p in author_results:
            if p.arxiv_id not in seen_ids:
                fts_results.append(p)
                seen_ids.add(p.arxiv_id)
        fts_results = fts_results[:n]

    return fts_results
