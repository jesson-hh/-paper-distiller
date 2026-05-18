"""Distill one paper into an ArticleResult.

Three steps:
  1. Render article.md prompt with paper + full_text + wiki_index
  2. Call LLM with response_format=json
  3. Parse, scrub invented [[wikilinks]], return ArticleResult
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..llm.openai_compatible import LLMClient, LLMError
from ..sources.arxiv import ArxivPaper
from ..vault.crosslink import WikiIndex
from ..vault.store import slugify

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "article.md"
_LINK_RE = re.compile(r"\[\[([^\]\|]+)(?:\|([^\]]+))?\]\]")


@dataclass
class ArticleResult:
    slug: str
    title: str
    body: str
    tags: list
    refs: list
    depth: str  # "full-pdf" or "abstract-only"

    def to_save_kwargs(self) -> dict:
        return {
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "refs": self.refs,
            "slug": self.slug,
        }


def _scrub_invented_links(body: str, valid_slugs: set) -> str:
    """Replace [[slug]] / [[slug|Display]] with bare slug or display text when slug is not in valid_slugs."""
    def repl(m):
        slug = m.group(1).strip()
        display = m.group(2)
        if slug in valid_slugs:
            return m.group(0)
        return display if display else slug
    return _LINK_RE.sub(repl, body)


def distill(
    paper: ArxivPaper,
    full_text: str,
    wiki_index: WikiIndex,
    llm: LLMClient,
) -> ArticleResult:
    """LLM-distill one paper into an article entry."""
    depth_mode = "full-pdf" if full_text and len(full_text) > 500 else "abstract-only"
    body_input = full_text if depth_mode == "full-pdf" else paper.abstract

    index_lines = wiki_index.to_prompt_lines() or ["(vault is empty — no crosslinks yet)"]
    wiki_index_block = "\n".join(index_lines[:200])  # cap at 200 lines (~10K tokens)

    prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
        paper_title=paper.title,
        paper_authors=", ".join(paper.authors),
        paper_arxiv_id=paper.arxiv_id,
        paper_published=paper.published,
        paper_abstract=paper.abstract,
        depth_mode=depth_mode,
        full_text=body_input[:120_000],  # cap to ~30K tokens worth
        wiki_index_block=wiki_index_block,
    )
    raw = llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.5,
        response_format="json",
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"article distillation returned non-JSON: {raw[:200]}") from e

    title = parsed.get("title", paper.title).strip()
    body = parsed.get("body", "").strip()
    tags = parsed.get("tags", []) or []
    refs = parsed.get("refs", []) or []
    if f"arxiv:{paper.arxiv_id}" not in refs:
        refs.insert(0, f"arxiv:{paper.arxiv_id}")
    if not body:
        raise LLMError("article distillation returned empty body")

    body = _scrub_invented_links(body, wiki_index.slugs())
    slug = slugify(title)

    return ArticleResult(
        slug=slug,
        title=title,
        body=body,
        tags=tags,
        refs=refs,
        depth=depth_mode,
    )
