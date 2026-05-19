"""QA-loop driver. Orchestrates multiple Orchestrator.run() invocations per
QA session: reflection -> distillation -> repeat -> synthesis.

Stop reasons (7 total): max_rounds, llm_done, llm_brake, no_candidates,
max_articles, max_cost, user_quit, plus 'error: ...' for transient failures.

State persisted to <vault>/.paper_distiller/qa-sessions/<sid>/state.json
after every round.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime

from rich.console import Console
from rich.live import Live

from ..agents.base import Context
from ..agents.curation import CandidateMerger, CandidateRanker
from ..agents.dag import DAG
from ..agents.dedup import CandidateDedup
from ..agents.opencli_openalex import OpenCLIOpenAlexSearcher
from ..agents.orchestrator import AgentFailed, Orchestrator
from ..agents.processor import PaperProcessor
from ..agents.reflector import ProgressReflector
from ..agents.renderer import ConsoleRenderer
from ..agents.searchers import ArxivSearcher, SemanticScholarSearcher
from ..agents.synthesizer import AnswerSynthesizer
from ..agents.writer import VaultWriter
from ..config import Config
from ..llm.openai_compatible import LLMClient
from ..qa.state import RoundRecord, SessionState, read_state, write_state
from ..vault.store import VaultStore


# qwen-plus pricing in CNY per 1M tokens (rough; only used for the cost budget
# circuit breaker, not for billing)
_PRICE_IN_CNY_PER_M = 2.1
_PRICE_OUT_CNY_PER_M = 12.7


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M") + "-" + secrets.token_hex(3)[:5]


def _build_reflection_dag() -> DAG:
    return DAG([ProgressReflector()])


def _build_distillation_dag() -> DAG:
    return DAG([
        ArxivSearcher(),
        SemanticScholarSearcher(),
        OpenCLIOpenAlexSearcher(),
        CandidateMerger(),
        CandidateDedup(),
        CandidateRanker(),
        PaperProcessor(),
        VaultWriter(),
    ])


def _build_synthesis_dag() -> DAG:
    # AnswerSynthesizer normally depends on "vault-writer" because it runs
    # at the end of a distillation DAG. In the standalone synthesis DAG we
    # override the instance's deps so the DAG validates with just one node.
    syn = AnswerSynthesizer()
    syn.deps = []
    return DAG([syn])


def _update_cost(state: SessionState, llm) -> None:
    state.tokens_in_total = llm.total_tokens_in
    state.tokens_out_total = llm.total_tokens_out
    state.cost_cny = (
        llm.total_tokens_in * _PRICE_IN_CNY_PER_M / 1_000_000
        + llm.total_tokens_out * _PRICE_OUT_CNY_PER_M / 1_000_000
    )


async def _arun_qa_loop(cfg: Config) -> SessionState:
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

    vault = VaultStore(cfg.vault_path)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    renderer = ConsoleRenderer(title=f"QA: {state.question[:50]}")

    def _build_ctx() -> Context:
        return Context(
            cfg=cfg, llm=llm, vault=vault,
            shared={"qa_state": state},
            on_status=renderer.on_status,
        )

    console = Console()
    with Live(renderer.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher():
            while True:
                live.update(renderer.build_table())
                await asyncio.sleep(0.1)
        refresher_task = asyncio.create_task(_refresher())
        try:
            while True:
                # 1. Reflection
                ctx = _build_ctx()
                try:
                    await Orchestrator(_build_reflection_dag(), ctx).run()
                except AgentFailed as e:
                    state.stop_reason = f"error: reflection failed: {e.__cause__}"
                    break
                reflection = ctx.shared["reflection"]
                state.last_reflection = reflection
                _update_cost(state, llm)

                # 2. Termination checks
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
                next_query = reflection.get("next_query") or ""
                if not next_query:
                    state.stop_reason = "no_candidates"
                    break

                # 3. Distillation round
                ctx = _build_ctx()
                ctx.shared["next_query"] = next_query
                try:
                    await Orchestrator(_build_distillation_dag(), ctx).run()
                except AgentFailed as e:
                    state.stop_reason = f"error: distillation failed: {e.__cause__}"
                    break

                # 4. Process this round's results
                articles_this_round = ctx.shared.get("articles", [])
                prior_slugs = {a.slug for a in state.articles_distilled}
                new_articles = [a for a in articles_this_round if a.slug not in prior_slugs]

                # Only stop with no_candidates if dedup wiped the input pool —
                # NOT on transient per-paper distill failures (v0.5 fault tolerance).
                candidates_after_dedup = ctx.shared.get("candidates", [])
                if not candidates_after_dedup:
                    state.stop_reason = "no_candidates"
                    break

                state.articles_distilled.extend(new_articles)
                # Update seen_ids from successfully-distilled articles' refs
                # (matches v0.5: only mark a paper "seen" if distill succeeded).
                for article in new_articles:
                    for ref in article.refs:
                        if ref.startswith("arxiv:"):
                            state.articles_seen_ids.add(ref[6:])
                            break
                        if ref.startswith("doi:"):
                            state.articles_seen_ids.add(ref[4:])
                            break

                # 5. Record round in history
                state.history.append(RoundRecord(
                    round=state.rounds_completed + 1,
                    query=next_query,
                    rationale=reflection.get("next_query_rationale", ""),
                    candidates_found=len(ctx.shared.get("candidates_arxiv", []))
                                    + len(ctx.shared.get("candidates_ss", []))
                                    + len(ctx.shared.get("candidates_openalex", [])),
                    new_articles=len(new_articles),
                    article_slugs=[a.slug for a in new_articles],
                    what_we_know=reflection.get("what_we_know", ""),
                    what_is_missing=reflection.get("what_is_missing", ""),
                    confidence=int(reflection.get("confidence", 0)),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                ))
                state.rounds_completed += 1
                _update_cost(state, llm)
                write_state(cfg.vault_path, state)

                # 6. Budget checks (post-round)
                if len(state.articles_distilled) >= cfg.qa_max_articles:
                    state.stop_reason = "max_articles"
                    break
                if state.cost_cny >= cfg.qa_max_cost_cny:
                    state.stop_reason = "max_cost"
                    break
        except KeyboardInterrupt:
            state.stop_reason = "user_quit"
            write_state(cfg.vault_path, state)
        finally:
            refresher_task.cancel()
            try:
                await refresher_task
            except asyncio.CancelledError:
                pass
            live.update(renderer.build_table())

    # 7. Synthesis (if articles exist)
    if state.articles_distilled:
        ctx = _build_ctx()
        try:
            await Orchestrator(_build_synthesis_dag(), ctx).run()
        except AgentFailed as e:
            print(f"Synthesis failed: {e.__cause__}")

    # 8. Final state save + is_done semantics
    # Transient stops (user_quit, error: *) leave the session resumable.
    # Terminal stops (budgets, llm_done, llm_brake, no_candidates) mark it done.
    non_terminal = (
        state.stop_reason == "user_quit"
        or state.stop_reason.startswith("error:")
    )
    state.is_done = not non_terminal
    _update_cost(state, llm)
    write_state(cfg.vault_path, state)
    return state


def run_qa_loop(cfg: Config) -> dict:
    """Sync entry point that returns a summary dict."""
    state = asyncio.run(_arun_qa_loop(cfg))
    return {
        "session_id": state.session_id,
        "stop_reason": state.stop_reason,
        "rounds_completed": state.rounds_completed,
        "articles_distilled_count": len(state.articles_distilled),
        "cost_cny": round(state.cost_cny, 2),
        "tokens_in_total": state.tokens_in_total,
        "tokens_out_total": state.tokens_out_total,
    }
