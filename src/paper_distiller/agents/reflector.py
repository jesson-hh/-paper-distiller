"""ProgressReflector — wraps qa.reflection.reflect into an agent."""

from __future__ import annotations

import asyncio

from ..qa.reflection import reflect
from .base import Context


def _article_summary_line(article) -> str:
    title = (article.title or "").replace("\n", " ").strip()[:120]
    return f"[[{article.slug}]] {title}"


class ProgressReflector:
    name = "progress-reflector"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        state = ctx.shared["qa_state"]
        articles_summary = [
            _article_summary_line(a) for a in state.articles_distilled
        ]
        prior_queries = [r.query for r in state.history if r.query]
        reflection_dict = await asyncio.to_thread(
            reflect,
            question=state.question,
            articles_summary=articles_summary,
            prior_queries=prior_queries,
            round_num=state.rounds_completed + 1,
            max_rounds=ctx.cfg.qa_max_rounds,
            llm=ctx.llm,
        )
        return {"reflection": reflection_dict}
