"""PaperProcessor — fanout agent: one sub-agent per paper. Each does
fetch + extract + distill independently, in parallel.

Per-paper LLM failures are logged + dropped — they don't abort the run.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..distill.article import distill as distill_article
from ..llm.openai_compatible import LLMError
from ..pipeline import fetch_with_fallback
from ..vault.crosslink import load_index
from .base import Agent, Context

# Module-level lock — fine because each run() invocation creates a fresh DAG.
# Multiple _DistillOne instances inside one fanout invocation share this lock
# to serialize their ctx.shared["articles"].append() calls.
_articles_lock = asyncio.Lock()


class _DistillOne:
    def __init__(self, paper, idx, total, tmpdir, wiki_index):
        self.name = f"paper-processor[{idx + 1}/{total}]"
        self.deps: list[str] = []
        self._paper = paper
        self._tmpdir = tmpdir
        self._wiki_index = wiki_index

    async def run(self, ctx: Context) -> dict:
        try:
            full_text = await asyncio.to_thread(
                fetch_with_fallback, self._paper, ctx.cfg, self._tmpdir,
            )
            article = await asyncio.to_thread(
                distill_article, self._paper, full_text, self._wiki_index, ctx.llm,
            )
        except LLMError:
            if ctx.cfg.verbose:
                print(f"  distill failed for {self._paper.arxiv_id}")
            return {}
        async with _articles_lock:
            current = ctx.shared.get("articles", [])
            current.append(article)
            ctx.shared["articles"] = current
        return {}


class PaperProcessor:
    """Fanout agent — produces N _DistillOne sub-agents at runtime.

    Sub-agents MUST have deps=[] — they run as a synthetic single-level
    fanout, not via topological sort.
    """
    name = "paper-processor"
    deps = ["candidate-ranker"]

    def expand(self, ctx: Context) -> list[Agent]:
        papers = ctx.shared.get("ranked", [])
        if not papers:
            # initialize empty list so downstream agents always see the key
            ctx.shared["articles"] = []
            return []
        ctx.shared.setdefault("articles", [])
        tmpdir = Path(tempfile.mkdtemp(prefix="paper-distiller-"))
        wiki_index = load_index(ctx.vault)
        return [
            _DistillOne(p, i, len(papers), tmpdir, wiki_index)
            for i, p in enumerate(papers)
        ]
