"""Search-source agents — wrap existing sources/{arxiv,semantic_scholar}.py.

Both run as no-deps agents and can execute in parallel (each level-0 in the DAG).

Graceful degradation: HTTP failures (429, network errors) at one source
are caught and converted to an empty candidates list with a stderr warning,
so the other source can still feed the pipeline. Only unexpected exceptions
propagate as AgentFailed.
"""

from __future__ import annotations

import asyncio
import sys

from ..sources.arxiv import search as arxiv_search
from ..sources.semantic_scholar import search as ss_search
from .base import Context
from .rate_limit import ARXIV_LIMITER, SS_LIMITER


def _is_transient_search_error(exc: Exception) -> bool:
    """Return True for HTTP / network / rate-limit errors we should degrade on."""
    s = str(exc).lower()
    return (
        "429" in s
        or "client error" in s
        or "server error" in s
        or "http" in s and "error" in s
        or "timeout" in s
        or "connection" in s and ("refused" in s or "reset" in s or "aborted" in s)
    )


def _mark_degraded(ctx: Context, source: str) -> None:
    """Append to ctx.shared['degraded_sources'] so tool_search can distinguish
    'genuinely empty' from 'rate-limited'."""
    ctx.shared.setdefault("degraded_sources", []).append(source)


class ArxivSearcher:
    name = "arxiv-searcher"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        if ctx.cfg.source not in ("arxiv", "both", "all"):
            return {"candidates_arxiv": []}
        if not await ARXIV_LIMITER.acquire():
            cd = ARXIV_LIMITER.seconds_until_ready()
            print(
                f"  arxiv cooling down ({cd:.0f}s remaining); skipping",
                file=sys.stderr,
            )
            _mark_degraded(ctx, "arxiv")
            return {"candidates_arxiv": []}
        query = ctx.shared.get("next_query") or ctx.cfg.topic or ctx.cfg.author or ""
        sort = ctx.shared.get("arxiv_sort") or "relevance"
        try:
            papers = await asyncio.to_thread(
                arxiv_search,
                query=query,
                max_results=ctx.cfg.pool,
                sort=sort,
            )
        except Exception as e:
            if _is_transient_search_error(e):
                ARXIV_LIMITER.mark_429()
                print(f"  arxiv search degraded ({type(e).__name__}): {str(e)[:120]}",
                      file=sys.stderr)
                _mark_degraded(ctx, "arxiv")
                return {"candidates_arxiv": []}
            raise
        return {"candidates_arxiv": papers}


class SemanticScholarSearcher:
    name = "ss-searcher"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        if ctx.cfg.source not in ("ss", "both", "all"):
            return {"candidates_ss": []}
        if not await SS_LIMITER.acquire():
            cd = SS_LIMITER.seconds_until_ready()
            print(
                f"  SS cooling down ({cd:.0f}s remaining); skipping",
                file=sys.stderr,
            )
            _mark_degraded(ctx, "ss")
            return {"candidates_ss": []}
        query = ctx.shared.get("next_query") or ctx.cfg.topic or ctx.cfg.author or ""
        try:
            papers = await asyncio.to_thread(
                ss_search,
                query=query,
                max_results=ctx.cfg.pool,
                api_key=ctx.cfg.ss_api_key,
            )
        except Exception as e:
            if _is_transient_search_error(e):
                SS_LIMITER.mark_429()
                print(f"  SS search degraded ({type(e).__name__}): {str(e)[:120]}",
                      file=sys.stderr)
                _mark_degraded(ctx, "ss")
                return {"candidates_ss": []}
            raise
        return {"candidates_ss": papers}
