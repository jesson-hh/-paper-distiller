"""CandidateMerger + CandidateRanker — combine + LLM-rank candidate Papers."""

from __future__ import annotations

import asyncio

from ..distill.filter import rank
from ..pipeline import merge_candidates
from .base import Context


class CandidateMerger:
    name = "candidate-merger"
    deps = ["arxiv-searcher", "ss-searcher", "openalex-searcher"]

    async def run(self, ctx: Context) -> dict:
        # Bypass mode (research-mode Phase 2): candidates injected directly,
        # no searcher chain needed. Caller must override deps=[] on the
        # instance to satisfy DAG validation.
        if "candidates_direct" in ctx.shared:
            return {"candidates": list(ctx.shared["candidates_direct"])}
        a = ctx.shared.get("candidates_arxiv", [])
        b = ctx.shared.get("candidates_ss", [])
        c = ctx.shared.get("candidates_openalex", [])
        # Two-pass merge: arxiv wins on tie over ss, then (arxiv+ss) wins over openalex.
        merged = merge_candidates(a, b)
        if c:
            merged = merge_candidates(merged, c)
        return {"candidates": merged}


class CandidateRanker:
    name = "candidate-ranker"
    deps = ["candidate-merger"]

    async def run(self, ctx: Context) -> dict:
        candidates = ctx.shared.get("candidates", [])
        if not candidates:
            return {"ranked": []}
        # Prefer qa_per_round (set in QA mode) over top_n (single-pass)
        top_n = ctx.cfg.qa_per_round if ctx.cfg.qa_per_round is not None else ctx.cfg.top_n
        topic = ctx.shared.get("next_query") or ctx.cfg.topic or ""
        ranked = await asyncio.to_thread(
            rank, candidates, topic, top_n, ctx.llm,
        )
        return {"ranked": ranked}
