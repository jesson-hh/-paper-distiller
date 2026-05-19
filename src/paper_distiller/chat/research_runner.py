"""Deep Research driver — 5-phase rolling cycle.

Phases:
  1. SEED       — Reflect + distill the first round (QA-style)
  2. EXPAND     — Citation graph expansion using CitationExplorer + bypass merger
  3. STRUCTURE  — TheoremExtractor adds structured frontmatter
  4. SYNTHESIZE — ThemeClusterer + per-cluster SurveyComposer
  5. GAP CHECK  — GapDetector LLM judges continue or stop

Stops on: max_papers, max_cost, max_duration, all_themes_synthesized,
user_quit, error:*
"""

from __future__ import annotations

import asyncio
import secrets
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live

from ..agents.base import Context
from ..agents.citation_explorer import CitationExplorer
from ..agents.curation import CandidateMerger, CandidateRanker
from ..agents.dag import DAG
from ..agents.dedup import CandidateDedup
from ..agents.gap_detector import GapDetector
from ..agents.opencli_openalex import OpenCLIOpenAlexSearcher
from ..agents.orchestrator import AgentFailed, Orchestrator
from ..agents.processor import PaperProcessor
from ..agents.reflector import ProgressReflector
from ..agents.renderer import ConsoleRenderer
from ..agents.searchers import ArxivSearcher, SemanticScholarSearcher
from ..agents.theme_clusterer import ThemeClusterer
from ..agents.theorem_extractor import TheoremExtractor
from ..agents.writer import VaultWriter
from ..config import Config
from ..distill.survey import compose as compose_survey
from ..llm.openai_compatible import LLMClient
from ..qa.research_state import (
    ResearchState, read_research_state, write_research_state,
)
from ..qa.state import SessionState
from ..vault.store import VaultStore, slugify


_PRICE_IN_CNY_PER_M = 2.1
_PRICE_OUT_CNY_PER_M = 12.7


def _new_session_id() -> str:
    return "rs-" + datetime.now().strftime("%Y%m%d-%H%M") + "-" + secrets.token_hex(3)[:5]


def _update_cost(state: ResearchState, llm) -> None:
    state.total_tokens_in = llm.total_tokens_in
    state.total_tokens_out = llm.total_tokens_out
    state.total_cost_cny = (
        llm.total_tokens_in * _PRICE_IN_CNY_PER_M / 1_000_000
        + llm.total_tokens_out * _PRICE_OUT_CNY_PER_M / 1_000_000
    )


def _qa_state_proxy(research_state: ResearchState, articles: list) -> SessionState:
    """Build a SessionState that v1.0 QA agents can read in research mode.

    CandidateDedup does `ids & seen` so seen MUST be a set.
    ProgressReflector reads articles_distilled (ArticleResult-like) + history.
    """
    return SessionState(
        session_id=research_state.session_id + "-qa",
        question=research_state.question,
        config_snapshot={},
        started_at=research_state.started_at,
        rounds_completed=research_state.iterations_completed,
        articles_distilled=list(articles),
        articles_seen_ids=set(research_state.papers_seen_ids),
        history=[],
    )


def _track_seen_id(state: ResearchState, article) -> None:
    for r in article.refs:
        if r.startswith("arxiv:"):
            state.papers_seen_ids.append(r[6:])
            return
        if r.startswith("doi:"):
            state.papers_seen_ids.append(r[4:])
            return


async def _phase_seed(cfg, llm, vault, state, renderer, all_articles, next_query):
    """Phase 1: QA-style reflection + distillation round."""
    qa_state = _qa_state_proxy(state, all_articles)
    ctx = Context(
        cfg=cfg, llm=llm, vault=vault,
        shared={"qa_state": qa_state},
        on_status=renderer.on_status,
    )
    # Reflection (may set next_query); failures here are tolerable — fall back.
    try:
        await Orchestrator(DAG([ProgressReflector()]), ctx).run()
        refl = ctx.shared.get("reflection", {})
        query = next_query or refl.get("next_query") or state.question
    except AgentFailed:
        query = next_query or state.question
    ctx.shared["next_query"] = query
    # Distill DAG (full v1.0 chain)
    dag = DAG([
        ArxivSearcher(), SemanticScholarSearcher(), OpenCLIOpenAlexSearcher(),
        CandidateMerger(), CandidateDedup(), CandidateRanker(),
        PaperProcessor(), VaultWriter(),
    ])
    await Orchestrator(dag, ctx).run()
    seed_papers = ctx.shared.get("ranked", [])
    new_articles = ctx.shared.get("articles", [])
    return seed_papers, new_articles


async def _phase_expand(cfg, llm, vault, state, renderer, all_articles, seed_pseudo):
    """Phase 2: Citation expansion using bypass mode."""
    if not seed_pseudo:
        return []
    qa_state = _qa_state_proxy(state, all_articles)
    ctx = Context(
        cfg=cfg, llm=llm, vault=vault,
        shared={"qa_state": qa_state, "seed_papers": seed_pseudo},
        on_status=renderer.on_status,
    )
    # Step 1: CitationExplorer
    await Orchestrator(DAG([CitationExplorer()]), ctx).run()
    candidates = ctx.shared.get("citation_expansion_candidates", [])
    if not candidates:
        return []
    # Step 2: Bypass-mode distill — feed candidates directly via candidates_direct
    ctx.shared["candidates_direct"] = candidates
    merger = CandidateMerger()
    merger.deps = []  # bypass deps validation; no searchers in this DAG
    bypass_dag = DAG([
        merger, CandidateDedup(), CandidateRanker(),
        PaperProcessor(), VaultWriter(),
    ])
    await Orchestrator(bypass_dag, ctx).run()
    return ctx.shared.get("articles", [])


async def _phase_structure(cfg, llm, vault, state, renderer, all_articles):
    """Phase 3: Run TheoremExtractor on all distilled articles."""
    if not all_articles:
        return {}
    ctx = Context(
        cfg=cfg, llm=llm, vault=vault,
        shared={"all_articles": all_articles},
        on_status=renderer.on_status,
    )
    await Orchestrator(DAG([TheoremExtractor()]), ctx).run()
    return ctx.shared.get("structured_extractions", {})


async def _phase_synthesize(cfg, llm, vault, state, renderer, all_articles):
    """Phase 4: Cluster into themes, write one synthesis per cluster."""
    if not all_articles:
        return [], []
    ctx = Context(
        cfg=cfg, llm=llm, vault=vault,
        shared={"all_articles": all_articles},
        on_status=renderer.on_status,
    )
    await Orchestrator(DAG([ThemeClusterer()]), ctx).run()
    themes = ctx.shared.get("themes", [])
    synthesis_slugs = []
    slug_to_article = {a.slug: a for a in all_articles}
    for i, theme in enumerate(themes):
        theme_articles = [
            slug_to_article[s]
            for s in theme.get("slugs", [])
            if s in slug_to_article
        ]
        if not theme_articles:
            continue
        try:
            survey = await asyncio.to_thread(
                compose_survey, theme_articles, theme["name"], None, llm,
            )
        except Exception as e:
            if cfg.verbose:
                print(f"  synthesis failed for theme {theme.get('name')}: {e}")
            continue
        slug_base = slugify(theme.get("name", f"theme-{i}"))[:30] or f"theme-{i}"
        slug = f"synthesis-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
        save_kwargs = dict(
            category="surveys",
            title=getattr(survey, "title", f"Synthesis: {theme.get('name')}"),
            body=getattr(survey, "body", ""),
            tags=getattr(survey, "tags", None) or ["synthesis"],
            refs=[f"theme:{theme.get('name')}"],
            slug=slug,
        )
        try:
            saved = await asyncio.to_thread(vault.save_entry, **save_kwargs)
        except ValueError:
            save_kwargs["slug"] = f"{slug}-{secrets.token_hex(2)}"
            saved = await asyncio.to_thread(vault.save_entry, **save_kwargs)
        synthesis_slugs.append(saved["slug"])
    return themes, synthesis_slugs


async def _phase_gap(cfg, llm, vault, state, renderer):
    """Phase 5: GapDetector decides continue or stop."""
    ctx = Context(
        cfg=cfg, llm=llm, vault=vault,
        shared={"research_state": state},
        on_status=renderer.on_status,
    )
    await Orchestrator(DAG([GapDetector()]), ctx).run()
    return ctx.shared.get(
        "gap_analysis", {"should_continue": False, "next_query": ""}
    )


def _write_final_report(cfg, vault, state) -> str:
    body_parts = [
        f"# Research Report: {state.question}\n",
        f"> 总文章: {len(state.papers_distilled)} 篇",
        f"> 主题数: {len(state.themes)}",
        f"> 合成文档数: {len(state.synthesis_slugs)}",
        f"> 迭代轮数: {state.iterations_completed}",
        f"> 总成本: ¥{state.total_cost_cny:.2f}",
        f"> Stop reason: {state.stop_reason}\n",
        "## 主题综合",
    ]
    if state.synthesis_slugs:
        for s in state.synthesis_slugs:
            body_parts.append(f"- [[{s}]]")
    else:
        body_parts.append("(无)")
    body_parts.append("\n## 蒸馏到的所有 articles\n")
    for slug in state.papers_distilled:
        body_parts.append(f"- [[{slug}]]")
    slug_base = slugify(state.question)[:30] or "research"
    slug = f"research-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
    save_kwargs = dict(
        category="surveys",
        title=f"Research: {state.question[:60]}",
        body="\n".join(body_parts),
        tags=["research"],
        refs=[f"research-session:{state.session_id}"],
        slug=slug,
    )
    try:
        saved = vault.save_entry(**save_kwargs)
    except ValueError:
        save_kwargs["slug"] = f"{slug}-{secrets.token_hex(2)}"
        saved = vault.save_entry(**save_kwargs)
    return saved["slug"]


async def _arun_research_loop(cfg: Config) -> ResearchState:
    # Init or resume
    if cfg.research_resume_session_id:
        existing = read_research_state(cfg.vault_path, cfg.research_resume_session_id)
        if existing is None:
            raise ValueError(
                f"resume session not found: {cfg.research_resume_session_id}"
            )
        if existing.is_done:
            raise ValueError(
                f"session {cfg.research_resume_session_id} already done; cannot resume"
            )
        state = existing
    else:
        state = ResearchState(
            session_id=_new_session_id(),
            question=cfg.qa_question,
            config_snapshot={
                "max_papers": cfg.research_max_papers,
                "max_cost_cny": cfg.research_max_cost_cny,
                "max_duration_sec": cfg.research_max_duration_sec,
            },
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

    vault = VaultStore(cfg.vault_path)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    renderer = ConsoleRenderer(title=f"Research: {state.question[:50]}")

    start_t = time.monotonic()
    console = Console()
    all_articles: list = []
    next_query = state.question  # initial query
    last_seed_papers: list = []  # raw Paper objects from latest seed phase

    with Live(renderer.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher():
            while True:
                live.update(renderer.build_table())
                await asyncio.sleep(0.1)
        refresher_task = asyncio.create_task(_refresher())

        try:
            while not state.is_done:
                # Budget checks BEFORE each phase
                elapsed = time.monotonic() - start_t
                if state.total_cost_cny >= cfg.research_max_cost_cny:
                    state.stop_reason = "max_cost"
                    state.is_done = True
                    break
                if elapsed >= cfg.research_max_duration_sec:
                    state.stop_reason = "max_duration"
                    state.is_done = True
                    break

                # Phase 1: SEED
                if state.phase == "seed":
                    try:
                        seed_papers, new_articles = await _phase_seed(
                            cfg, llm, vault, state, renderer,
                            all_articles, next_query,
                        )
                    except AgentFailed as e:
                        state.stop_reason = f"error: seed phase failed: {e.__cause__}"
                        break
                    last_seed_papers = list(seed_papers)
                    all_articles.extend(new_articles)
                    state.papers_distilled.extend([a.slug for a in new_articles])
                    for a in new_articles:
                        _track_seen_id(state, a)
                    if len(state.papers_distilled) >= cfg.research_max_papers:
                        state.stop_reason = "max_papers"
                        state.is_done = True
                        # still go through structure + synthesize before exiting
                    state.phase = "expand"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # Phase 2: EXPAND (skip fetching if max_papers already hit, but still advance phase)
                if state.phase == "expand":
                    if not state.is_done and last_seed_papers:
                        try:
                            expanded = await _phase_expand(
                                cfg, llm, vault, state, renderer,
                                all_articles, last_seed_papers,
                            )
                        except AgentFailed as e:
                            state.stop_reason = f"error: expand phase failed: {e.__cause__}"
                            break
                        all_articles.extend(expanded)
                        state.papers_distilled.extend([a.slug for a in expanded])
                        for a in expanded:
                            _track_seen_id(state, a)
                        if len(state.papers_distilled) >= cfg.research_max_papers:
                            state.stop_reason = "max_papers"
                            state.is_done = True
                    state.phase = "structure"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # Phase 3: STRUCTURE (runs even if is_done — we want extractions on what we have)
                if state.phase == "structure":
                    try:
                        extractions = await _phase_structure(
                            cfg, llm, vault, state, renderer, all_articles,
                        )
                    except AgentFailed as e:
                        state.stop_reason = f"error: structure phase failed: {e.__cause__}"
                        break
                    state.structured_extractions = extractions
                    state.phase = "synthesize"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # Phase 4: SYNTHESIZE (runs even if is_done — we want the synthesis docs)
                if state.phase == "synthesize":
                    try:
                        themes, synthesis_slugs = await _phase_synthesize(
                            cfg, llm, vault, state, renderer, all_articles,
                        )
                    except AgentFailed as e:
                        state.stop_reason = f"error: synthesize phase failed: {e.__cause__}"
                        break
                    state.themes = themes
                    state.synthesis_slugs.extend(synthesis_slugs)
                    state.phase = "gap"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)

                # Phase 5: GAP CHECK (skip if we already know we're done from earlier phases)
                state.iterations_completed += 1
                if state.is_done:
                    break  # max_papers / max_cost / max_duration hit earlier
                try:
                    gap = await _phase_gap(cfg, llm, vault, state, renderer)
                except AgentFailed as e:
                    state.stop_reason = f"error: gap phase failed: {e.__cause__}"
                    break
                if not gap.get("should_continue"):
                    state.stop_reason = "all_themes_synthesized"
                    state.is_done = True
                else:
                    next_query = gap.get("next_query") or state.question
                    state.phase = "seed"
                    _update_cost(state, llm)
                    write_research_state(cfg.vault_path, state)
        except KeyboardInterrupt:
            state.stop_reason = "user_quit"
            write_research_state(cfg.vault_path, state)
        finally:
            refresher_task.cancel()
            try:
                await refresher_task
            except asyncio.CancelledError:
                pass
            live.update(renderer.build_table())

    # Final report
    if state.papers_distilled:
        try:
            state.final_report_slug = _write_final_report(cfg, vault, state)
        except Exception as e:
            if cfg.verbose:
                print(f"final report failed: {e}")

    # Final state
    non_terminal = (
        state.stop_reason == "user_quit"
        or state.stop_reason.startswith("error:")
    )
    state.is_done = not non_terminal
    _update_cost(state, llm)
    write_research_state(cfg.vault_path, state)
    return state


def run_research_loop(cfg: Config) -> dict:
    """Sync entry. Returns summary dict."""
    state = asyncio.run(_arun_research_loop(cfg))
    return {
        "session_id": state.session_id,
        "stop_reason": state.stop_reason,
        "papers_distilled_count": len(state.papers_distilled),
        "themes_count": len(state.themes),
        "synthesis_count": len(state.synthesis_slugs),
        "final_report_slug": state.final_report_slug,
        "total_cost_cny": round(state.total_cost_cny, 2),
        "total_tokens_in": state.total_tokens_in,
        "total_tokens_out": state.total_tokens_out,
        "iterations_completed": state.iterations_completed,
    }
