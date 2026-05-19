"""Search-source agents — wrap existing sources/{arxiv,semantic_scholar}.py.

Both run as no-deps agents and can execute in parallel (each level-0 in the DAG).
"""

from __future__ import annotations

import asyncio

from ..sources.arxiv import search as arxiv_search
from ..sources.semantic_scholar import search as ss_search
from .base import Context


class ArxivSearcher:
    name = "arxiv-searcher"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        if ctx.cfg.source not in ("arxiv", "both"):
            return {"candidates_arxiv": []}
        query = ctx.shared.get("next_query") or ctx.cfg.topic or ctx.cfg.author or ""
        papers = await asyncio.to_thread(
            arxiv_search,
            query=query,
            max_results=ctx.cfg.pool,
        )
        return {"candidates_arxiv": papers}


class SemanticScholarSearcher:
    name = "ss-searcher"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        if ctx.cfg.source not in ("ss", "both"):
            return {"candidates_ss": []}
        query = ctx.shared.get("next_query") or ctx.cfg.topic or ctx.cfg.author or ""
        papers = await asyncio.to_thread(
            ss_search,
            query=query,
            max_results=ctx.cfg.pool,
            api_key=ctx.cfg.ss_api_key,
        )
        return {"candidates_ss": papers}
