"""OpenCLIOpenAlexSearcher — wraps OpenCLI's openalex adapter via subprocess.

OpenAlex has 100 req/s rate limit (vs Semantic Scholar's much stricter limits),
and covers ~250M scholarly works. Going through OpenCLI keeps the door open
for future adapters (Zhihu/Bilibili discussion context, ACM/IEEE venue logins).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys

from ..sources.arxiv import Paper
from .base import Context


async def _opencli_call(args: list, timeout: float = 60.0) -> list:
    """Run opencli subprocess, return parsed JSON or [] on failure.

    Graceful degradation: timeouts, non-zero exits, and JSON parse errors
    all log to stderr and return []. The agent caller treats empty results
    as "no candidates" without aborting the DAG.

    Resolves the `opencli` binary via shutil.which so Windows .cmd shims
    (installed by `npm install -g`) are picked up — create_subprocess_exec
    does NOT honor PATHEXT, only the bare executable name.
    """
    binary = shutil.which("opencli")
    if binary is None:
        print("  opencli not found on PATH", file=sys.stderr)
        return []
    cmd = [binary, *args, "-f", "json"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError, NotImplementedError) as e:
        print(f"  opencli unavailable ({type(e).__name__}): {e}", file=sys.stderr)
        return []
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        print(f"  opencli timeout: {' '.join(cmd)}", file=sys.stderr)
        return []
    if proc.returncode != 0:
        # Compress OpenCLI's structured error dump into a single readable line.
        err_text = err.decode("utf-8", "replace")
        out_text = out.decode("utf-8", "replace")
        blob = err_text + " " + out_text
        msg = "openalex unreachable"  # default summary
        for needle, summary in (
            ("fetch failed", "openalex unreachable (network)"),
            ("ENOTFOUND", "openalex DNS lookup failed"),
            ("ETIMEDOUT", "openalex request timed out"),
            ("rate limit", "openalex rate-limited"),
            ("429", "openalex rate-limited (429)"),
            ("getaddrinfo", "openalex DNS lookup failed"),
        ):
            if needle in blob:
                msg = summary
                break
        print(
            f"  [openalex degraded] {msg}; continuing with arxiv + SS.",
            file=sys.stderr,
        )
        return []
    text = out.decode("utf-8", "replace").strip()
    # opencli may emit warning lines before the JSON payload; skip to first '[' or '{'
    for i, ch in enumerate(text):
        if ch in "[{":
            text = text[i:]
            break
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(
            f"  opencli JSON parse failed: {e}; output prefix: {text[:200]}",
            file=sys.stderr,
        )
        return []


def _to_paper(work: dict) -> Paper:
    """Convert an OpenCLI openalex work dict into a Paper dataclass.

    Handles both search-result shape (with firstAuthor) and detail-result
    shape (with full authors string). Extracts arxiv_id from arxiv-style DOIs
    (10.48550/arxiv.<id>).
    """
    doi = work.get("doi", "") or ""
    arxiv_id = None
    # OpenAlex assigns DOI like "10.48550/arxiv.2107.03502" for arxiv preprints
    if doi.startswith("10.48550/arxiv."):
        arxiv_id = doi[len("10.48550/arxiv."):]
    authors_str = work.get("authors", "") or work.get("firstAuthor", "") or ""
    authors = [a.strip() for a in authors_str.split(",") if a.strip()]
    published = work.get("date") or str(work.get("year", "")) or ""
    abstract = (work.get("abstract") or "").replace("\\n", " ").strip()
    return Paper(
        source="openalex",
        paper_id=work.get("id", "") or "",
        arxiv_id=arxiv_id,
        doi=doi or None,
        title=work.get("title", "") or "",
        authors=authors,
        abstract=abstract,
        pdf_url=work.get("openAccessUrl", "") or "",
        published=published,
        categories=[],
    )


class OpenCLIOpenAlexSearcher:
    """Search OpenAlex via the OpenCLI Node CLI tool.

    Two-phase: a `search` call yields IDs + summaries, then per-ID `work` calls
    enrich with abstract + open-access PDF URL. Detail fetches run with a
    concurrency cap of 5 to stay polite under OpenAlex's 100 req/s limit.
    """

    name = "openalex-searcher"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        if ctx.cfg.source not in ("openalex", "all"):
            return {"candidates_openalex": []}
        query = ctx.shared.get("next_query") or ctx.cfg.topic or ctx.cfg.author or ""
        if not query.strip():
            return {"candidates_openalex": []}

        # Step 1: search. Stay reasonable on limit even though OpenAlex allows 200.
        limit = min(getattr(ctx.cfg, "pool", 30), 25)
        search_results = await _opencli_call(
            ["openalex", "search", query, "--limit", str(limit)],
        )
        if not search_results:
            # Mark openalex as degraded so tool_search can distinguish empty
            # results from "all sources rate-limited" thrashing.
            ctx.shared.setdefault("degraded_sources", []).append("openalex")
            return {"candidates_openalex": []}

        # Step 2: fetch detail for each (parallel, cap 5 concurrent to stay polite).
        sem = asyncio.Semaphore(5)

        async def _fetch_one(item: dict) -> Paper:
            async with sem:
                detail = await _opencli_call(["openalex", "work", item["id"]])
            if not detail:
                # Fall back to the search-result row (no abstract, but still a candidate).
                return _to_paper(item)
            return _to_paper(detail[0])

        papers = await asyncio.gather(*[_fetch_one(item) for item in search_results])
        return {"candidates_openalex": list(papers)}
