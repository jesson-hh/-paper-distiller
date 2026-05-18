"""Question-driven multi-round research loop.

Composes existing L2 primitives (gather_candidates / rank /
fetch_with_fallback / distill_article) into a state-machine loop. The
loop terminates when any of seven conditions fire (see stop_reason in
SessionState).
"""

from __future__ import annotations

import secrets
import tempfile
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..distill.article import distill as distill_article
from ..distill.filter import rank
from ..llm.openai_compatible import LLMClient, LLMError
from ..pipeline import gather_candidates, fetch_with_fallback
from ..vault.crosslink import load_index
from ..vault.store import VaultStore, slugify
from .answer import synthesize, AnswerError
from .reflection import reflect, ReflectionError
from .state import SessionState, RoundRecord, write_state, read_state


# qwen-plus pricing in CNY per 1M tokens (rough; only used for the cost budget
# circuit breaker, not for billing)
_PRICE_IN_CNY_PER_M = 2.1
_PRICE_OUT_CNY_PER_M = 12.7


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M") + "-" + secrets.token_hex(3)[:5]


def _article_summary_line(article) -> str:
    """One-line summary of an article for the reflection prompt."""
    title = (article.title or "").replace("\n", " ").strip()[:120]
    return f"[[{article.slug}]] {title}"


def _record_round(
    state: SessionState,
    *,
    round_num: int,
    reflection: dict,
    candidates_count: int,
    distilled: list,
) -> None:
    state.history.append(RoundRecord(
        round=round_num,
        query=reflection.get("next_query", ""),
        rationale=reflection.get("next_query_rationale", ""),
        candidates_found=candidates_count,
        new_articles=len(distilled),
        article_slugs=[a.slug for a in distilled],
        what_we_know=reflection.get("what_we_know", ""),
        what_is_missing=reflection.get("what_is_missing", ""),
        confidence=int(reflection.get("confidence", 0)),
        timestamp=datetime.now().isoformat(timespec="seconds"),
    ))


def _update_cost(state: SessionState, llm: LLMClient) -> None:
    state.tokens_in_total = llm.total_tokens_in
    state.tokens_out_total = llm.total_tokens_out
    state.cost_cny = (
        llm.total_tokens_in * _PRICE_IN_CNY_PER_M / 1_000_000
        + llm.total_tokens_out * _PRICE_OUT_CNY_PER_M / 1_000_000
    )


def _audit_trail_markdown(history: list, stop_reason: str, state: SessionState) -> str:
    """Render the audit table for the survey footer."""
    rows = ["| 轮 | Query | 新增 | LLM 判断 | Confidence |",
            "|---|---|---|---|---|"]
    for r in history:
        what_missing = (r.what_is_missing or r.what_we_know or "").replace("\n", " ")[:50]
        rows.append(
            f"| {r.round} | {r.query[:40]} | {r.new_articles} | "
            f"{what_missing} | {r.confidence} |"
        )
    table = "\n".join(rows)
    footer = (
        f"\n\n**Stop reason**: {stop_reason}\n"
        f"**Rounds**: {state.rounds_completed}\n"
        f"**Articles distilled**: {len(state.articles_distilled)}\n"
        f"**Total cost**: ¥{state.cost_cny:.2f} ({state.tokens_in_total} in / "
        f"{state.tokens_out_total} out tokens)\n"
        f"**Session ID**: {state.session_id}\n"
    )
    return table + footer


def _build_survey_body(answer: dict, state: SessionState) -> str:
    """Assemble the final survey body: LLM answer + cited articles table + audit trail."""
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
    parts.append(_audit_trail_markdown(state.history, state.stop_reason, state))
    return "\n".join(parts)


def _interactive_continue(reflection: dict, round_num: int, max_rounds: int) -> bool:
    """Print reflection JSON and prompt Y/n/q. Returns True (continue) or False (stop)."""
    print(f"\n--- Round {round_num} / {max_rounds} reflection ---")
    print(f"  confidence: {reflection.get('confidence')}")
    print(f"  what_we_know: {reflection.get('what_we_know')}")
    print(f"  what_is_missing: {reflection.get('what_is_missing')}")
    print(f"  next_query: {reflection.get('next_query')}")
    print(f"  rationale: {reflection.get('next_query_rationale')}")
    reply = input(f"Continue to round {round_num + 1}? [Y/n/q] ").strip().lower()
    return reply in ("", "y", "yes")


def run(cfg: Config) -> dict:
    """Execute the QA loop. Returns a summary dict."""
    if cfg.qa_resume_session_id:
        existing = read_state(cfg.vault_path, cfg.qa_resume_session_id)
        if existing is None:
            raise ValueError(f"resume session not found: {cfg.qa_resume_session_id}")
        if existing.is_done:
            raise ValueError(
                f"session {cfg.qa_resume_session_id} already done "
                f"(stop_reason={existing.stop_reason!r}); cannot resume"
            )
        state = existing
    else:
        state = SessionState(
            session_id=_new_session_id(),
            question=cfg.qa_question,
            config_snapshot={
                "max_rounds": cfg.qa_max_rounds,
                "max_articles": cfg.qa_max_articles,
                "max_cost_cny": cfg.qa_max_cost_cny,
                "confidence_threshold": cfg.qa_confidence_threshold,
                "per_round": cfg.qa_per_round,
                "source": cfg.source,
            },
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

    if cfg.dry_run:
        print(f"[DRY-RUN] Would run QA loop for question: {cfg.qa_question!r}")
        return {
            "session_id": state.session_id,
            "stop_reason": "dry_run",
            "rounds_completed": 0,
            "articles_distilled_count": 0,
            "survey_slug": None,
            "cost_cny": 0.0,
            "tokens_in_total": 0,
            "tokens_out_total": 0,
        }

    store = VaultStore(cfg.vault_path)
    wiki_index = load_index(store)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)

    prior_queries: list = [r.query for r in state.history if r.query]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        try:
            while True:
                # 1. reflection
                articles_summary = [
                    _article_summary_line(a) for a in state.articles_distilled
                ]
                round_num = state.rounds_completed + 1
                try:
                    reflection = reflect(
                        question=state.question,
                        articles_summary=articles_summary,
                        prior_queries=prior_queries,
                        round_num=round_num,
                        max_rounds=cfg.qa_max_rounds,
                        llm=llm,
                    )
                except ReflectionError as e:
                    state.stop_reason = f"error: reflection failed: {e}"
                    break
                state.last_reflection = reflection
                _update_cost(state, llm)

                # 2. termination checks
                if state.rounds_completed >= cfg.qa_max_rounds:
                    state.stop_reason = "max_rounds"
                    break
                if reflection.get("is_done") and \
                        int(reflection.get("confidence", 0)) >= cfg.qa_confidence_threshold:
                    state.stop_reason = "llm_done"
                    break
                if reflection.get("suggest_stop"):
                    state.stop_reason = "llm_brake"
                    break

                # 3. interactive checkpoint
                if cfg.qa_interactive:
                    if not _interactive_continue(reflection, round_num, cfg.qa_max_rounds):
                        state.stop_reason = "user_quit"
                        break

                # 4. search
                next_query = reflection.get("next_query") or ""
                if not next_query:
                    state.stop_reason = "no_candidates"
                    break
                try:
                    cfg.topic = next_query
                    candidates = gather_candidates(cfg)
                except Exception as e:
                    state.stop_reason = f"error: search failed: {e}"
                    break

                # 5. dedup
                new_candidates = []
                for p in candidates:
                    pid = p.arxiv_id or p.doi
                    if pid and pid in state.articles_seen_ids:
                        continue
                    new_candidates.append(p)

                if not new_candidates:
                    state.stop_reason = "no_candidates"
                    break

                # 6. rank
                try:
                    top = rank(new_candidates, state.question,
                                cfg.qa_per_round, llm)
                except Exception as e:
                    state.stop_reason = f"error: ranker failed: {e}"
                    break

                # 7. distill loop
                distilled_this_round = []
                for paper in top:
                    full_text = fetch_with_fallback(paper, cfg, tmpdir_path)
                    try:
                        article = distill_article(paper, full_text, wiki_index, llm)
                    except LLMError as e:
                        if cfg.verbose:
                            print(f"  distill failed for {paper.arxiv_id}: {e}")
                        continue
                    store.save_entry(category="articles", **article.to_save_kwargs())
                    state.articles_distilled.append(article)
                    pid = paper.arxiv_id or paper.doi or paper.paper_id
                    if pid:
                        state.articles_seen_ids.add(pid)
                    distilled_this_round.append(article)

                # 8. record round + persist
                _record_round(
                    state, round_num=round_num, reflection=reflection,
                    candidates_count=len(candidates), distilled=distilled_this_round,
                )
                if next_query:
                    prior_queries.append(next_query)
                state.rounds_completed += 1
                _update_cost(state, llm)
                write_state(cfg.vault_path, state)

                # 9. article + cost budgets
                if len(state.articles_distilled) >= cfg.qa_max_articles:
                    state.stop_reason = "max_articles"
                    break
                if state.cost_cny >= cfg.qa_max_cost_cny:
                    state.stop_reason = "max_cost"
                    break
        except KeyboardInterrupt:
            state.stop_reason = "user_quit"
            write_state(cfg.vault_path, state)
            print(f"\nSession paused. Resume with: --resume {state.session_id}")

    # Final synthesis (skip if no articles)
    survey_slug = None
    if state.articles_distilled:
        try:
            answer = synthesize(state.question, state.articles_distilled, llm)
        except AnswerError as e:
            if cfg.verbose:
                print(f"answer synthesis failed: {e}; writing skeleton survey only")
            answer = {
                "title": f"QA: {state.question[:60]}",
                "body": f"> 答案合成失败 ({e}). 已蒸馏 {len(state.articles_distilled)} 篇相关文章。",
                "tags": ["qa", "synthesis-failed"],
                "cited_slugs": [a.slug for a in state.articles_distilled],
            }
        body = _build_survey_body(answer, state)
        slug_base = slugify(state.question)[:30] or "untitled"
        slug = f"qa-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
        try:
            saved = store.save_entry(
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        except ValueError:
            slug = f"{slug}-{secrets.token_hex(2)}"
            saved = store.save_entry(
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        survey_slug = saved["slug"]

    state.is_done = True
    _update_cost(state, llm)
    write_state(cfg.vault_path, state)

    return {
        "session_id": state.session_id,
        "stop_reason": state.stop_reason,
        "rounds_completed": state.rounds_completed,
        "articles_distilled_count": len(state.articles_distilled),
        "survey_slug": survey_slug,
        "cost_cny": round(state.cost_cny, 2),
        "tokens_in_total": state.tokens_in_total,
        "tokens_out_total": state.tokens_out_total,
    }
