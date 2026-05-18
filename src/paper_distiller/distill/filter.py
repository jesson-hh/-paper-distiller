"""LLM-based ranking of arxiv candidates against the user's topic."""

from __future__ import annotations

import json
from pathlib import Path

from ..sources.arxiv import ArxivPaper
from ..llm.openai_compatible import LLMClient, LLMError

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "filter.md"


def rank(
    candidates: list[ArxivPaper],
    topic: str,
    top_n: int,
    llm: LLMClient,
) -> list[ArxivPaper]:
    """Ask the LLM to pick top_n best candidates. Returns ordered list.

    Hallucinated arxiv_ids (not present in candidates) are silently dropped.
    """
    if not candidates:
        return []
    candidates_json = json.dumps(
        [{"arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract[:500]}
         for p in candidates],
        ensure_ascii=False,
    )
    prompt = _PROMPT_FILE.read_text(encoding="utf-8").format(
        top_n=top_n,
        topic=topic,
        candidates_json=candidates_json,
    )
    raw = llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format="json",
    )
    try:
        parsed = json.loads(raw)
        selected = parsed.get("selected", [])
    except json.JSONDecodeError as e:
        raise LLMError(f"filter returned non-JSON: {raw[:200]}") from e

    candidates_by_id = {p.arxiv_id: p for p in candidates}
    out = []
    for entry in selected:
        aid = entry.get("arxiv_id", "")
        if aid in candidates_by_id:
            out.append(candidates_by_id[aid])
    return out
