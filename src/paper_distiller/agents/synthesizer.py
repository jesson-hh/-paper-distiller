"""AnswerSynthesizer — wraps qa.answer.synthesize, appends audit trail,
writes the final answer survey to <vault>/surveys/qa-<slug>-<date>.md.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime

from ..qa.answer import synthesize
from ..vault.store import slugify
from .base import Context


def _audit_trail_markdown(state) -> str:
    rows = ["| 轮 | Query | 新增 | LLM 判断 | Confidence |",
            "|---|---|---|---|---|"]
    for r in state.history:
        what_missing = (r.what_is_missing or r.what_we_know or "").replace("\n", " ")[:50]
        rows.append(
            f"| {r.round} | {r.query[:40]} | {r.new_articles} | "
            f"{what_missing} | {r.confidence} |"
        )
    table = "\n".join(rows)
    footer = (
        f"\n\n**Stop reason**: {state.stop_reason}\n"
        f"**Rounds**: {state.rounds_completed}\n"
        f"**Articles distilled**: {len(state.articles_distilled)}\n"
        f"**Total cost**: ¥{state.cost_cny:.2f} ({state.tokens_in_total} in / "
        f"{state.tokens_out_total} out tokens)\n"
        f"**Session ID**: {state.session_id}\n"
    )
    return table + footer


def _build_full_body(answer: dict, state) -> str:
    parts = [answer["body"]]
    cited = answer.get("cited_slugs") or []
    if cited:
        cited_rows = ["", "## 引用的 articles", "", "| Slug | 标题 |", "|---|---|"]
        slug_to_article = {a.slug: a for a in state.articles_distilled}
        for slug in cited:
            article = slug_to_article.get(slug)
            if article is not None:
                title = (article.title or "").replace("\n", " ")[:80]
                cited_rows.append(f"| [[{slug}]] | {title} |")
        parts.append("\n".join(cited_rows))
    parts.append("\n## 研究过程 (audit trail)\n")
    parts.append(_audit_trail_markdown(state))
    return "\n".join(parts)


class AnswerSynthesizer:
    name = "answer-synthesizer"
    deps = ["vault-writer"]

    async def run(self, ctx: Context) -> dict:
        state = ctx.shared["qa_state"]
        if not state.articles_distilled:
            return {"answer_survey_slug": None}
        answer = await asyncio.to_thread(
            synthesize, state.question, state.articles_distilled, ctx.llm,
        )
        body = _build_full_body(answer, state)
        slug_base = slugify(state.question)[:30] or "untitled"
        slug = f"qa-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
        try:
            saved = await asyncio.to_thread(
                ctx.vault.save_entry,
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        except ValueError:
            slug = f"{slug}-{secrets.token_hex(2)}"
            saved = await asyncio.to_thread(
                ctx.vault.save_entry,
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        return {"answer_survey_slug": saved["slug"]}
