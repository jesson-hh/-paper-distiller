"""paper-distiller-chat entry point.

v1.4 (default): natural-language conversational agent. Invoking
`paper-distiller-chat --vault X` (no subcommand) launches the AgentLoop —
the LLM decides which tools to call based on user input.

Legacy access: `paper-distiller-chat legacy-repl --vault X` opens the
pre-v1.4 slash-command REPL for users who prefer explicit control.

The single-shot subcommands (distill / browse / ask / resume / research)
remain for scripts and CI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from rich.console import Console
from rich.live import Live

from ..agents.base import Context
from ..agents.curation import CandidateMerger, CandidateRanker
from ..agents.dag import DAG
from ..agents.opencli_openalex import OpenCLIOpenAlexSearcher
from ..agents.orchestrator import Orchestrator, AgentFailed
from ..agents.processor import PaperProcessor
from ..agents.renderer import ConsoleRenderer
from ..agents.searchers import ArxivSearcher, SemanticScholarSearcher
from ..agents.writer import SurveyComposer, VaultWriter
from ..config import load_config, load_config_qa, load_config_research
from ..llm.openai_compatible import LLMClient
from ..vault.store import VaultStore
from ._durations import parse_duration as _parse_duration_impl
from .agent_loop import AgentLoop
from .qa_runner import run_qa_loop
from .repl.loop import REPL
from .research_runner import run_research_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-distiller-chat",
        description="Chat-first paper distillation.",
    )
    p.add_argument("--vault", help="Vault path (used when launching REPL without subcommand)")
    sub = p.add_subparsers(dest="subcommand", required=False)

    distill = sub.add_parser("distill", help="Single-pass: search a topic, distill N papers")
    distill.add_argument("--vault", required=True)
    distill.add_argument("--topic", help="Search topic")
    distill.add_argument("--author", help="Search author (alternative to --topic)")
    distill.add_argument("--n", type=int, default=3, help="Articles to distill (default 3)")
    distill.add_argument("--pool", type=int, default=30, help="Search pool size (default 30)")
    distill.add_argument(
        "--source",
        choices=["arxiv", "ss", "openalex", "both", "all"], default="all",
    )
    distill.add_argument("--dry-run", action="store_true")
    distill.add_argument("--verbose", "-v", action="store_true")
    distill.add_argument("--model", help="Override PD_MODEL env var")
    distill.add_argument("--provider", help="Override PD_PROVIDER_NAME label")

    browse = sub.add_parser(
        "browse",
        help="Search + show abstracts, pick which to distill (cheap interactive review)",
    )
    browse.add_argument("--vault", required=True)
    browse.add_argument("--topic", help="Search topic")
    browse.add_argument("--author", help="Search by author")
    browse.add_argument(
        "--n", type=int, default=10, help="How many candidates to show (default 10)"
    )
    browse.add_argument("--pool", type=int, default=30)
    browse.add_argument(
        "--source",
        choices=["arxiv", "ss", "openalex", "both", "all"], default="all",
    )
    browse.add_argument("--dry-run", action="store_true")
    browse.add_argument("--verbose", "-v", action="store_true")
    browse.add_argument("--model")
    browse.add_argument("--provider")

    ask = sub.add_parser("ask", help="QA loop: ask a research question, multiple rounds")
    ask.add_argument("--vault", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--max-rounds", type=int, default=5)
    ask.add_argument("--max-articles", type=int, default=15)
    ask.add_argument("--max-cost-cny", type=float, default=20.0)
    ask.add_argument("--confidence-threshold", type=int, default=8)
    ask.add_argument("--per-round", type=int, default=2)
    ask.add_argument(
        "--source",
        choices=["arxiv", "ss", "openalex", "both", "all"], default="all",
    )
    ask.add_argument("--interactive", action="store_true")
    ask.add_argument("--dry-run", action="store_true")
    ask.add_argument("--verbose", "-v", action="store_true")
    ask.add_argument("--model")
    ask.add_argument("--provider")

    resume = sub.add_parser("resume", help="Resume a paused/errored QA session")
    resume.add_argument("--vault", required=True)
    resume.add_argument("--session-id", required=True)
    resume.add_argument("--verbose", "-v", action="store_true")
    resume.add_argument("--model")
    resume.add_argument("--provider")

    research = sub.add_parser("research", help="Deep research: 4h autonomous loop on a question")
    research.add_argument("--vault", required=True)
    research.add_argument("--question", required=True)
    research.add_argument("--max-papers", type=int, default=30)
    research.add_argument("--max-cost-cny", type=float, default=30.0)
    research.add_argument("--duration", default="4h",
                          help="Time budget, e.g. '2h', '30m', '1h30m', '3600s'")
    research.add_argument(
        "--source",
        choices=["arxiv", "ss", "openalex", "both", "all"], default="all",
    )
    research.add_argument("--resume", help="Resume session-id")
    research.add_argument("--dry-run", action="store_true")
    research.add_argument("--verbose", "-v", action="store_true")
    research.add_argument("--model")
    research.add_argument("--provider")

    legacy = sub.add_parser(
        "legacy-repl",
        help="Pre-v1.4 slash-command REPL (intent router + slot filling). "
             "Default mode is the conversational agent.",
    )
    legacy.add_argument("--vault", required=True)
    legacy.add_argument("--model")
    legacy.add_argument("--provider")
    return p


# Kept as a module-level name for backward compat with any tests / scripts
# that import it from this file. Real implementation lives in ._durations.
_parse_duration = _parse_duration_impl


def _build_single_pass_dag() -> DAG:
    return DAG([
        ArxivSearcher(),
        SemanticScholarSearcher(),
        OpenCLIOpenAlexSearcher(),
        CandidateMerger(),
        CandidateRanker(),
        PaperProcessor(),
        VaultWriter(),
        SurveyComposer(),
    ])


async def _run_distill(args) -> int:
    try:
        cfg = load_config(
            vault_path=args.vault,
            topic=args.topic,
            author=args.author,
            n=args.n,                       # correct kwarg name (not top_n)
            pool=args.pool,
            source=args.source,
            dry_run=args.dry_run,
            verbose=args.verbose,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if cfg.dry_run:
        print(f"[DRY-RUN] Would distill {cfg.top_n} papers on {cfg.topic!r}")
        return 0

    vault = VaultStore(cfg.vault_path)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    renderer = ConsoleRenderer(title=f"distill · {cfg.topic or cfg.author}")
    ctx = Context(cfg=cfg, llm=llm, vault=vault, shared={}, on_status=renderer.on_status)

    console = Console()
    dag = _build_single_pass_dag()
    orch = Orchestrator(dag, ctx)

    with Live(renderer.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher():
            while True:
                live.update(renderer.build_table())
                await asyncio.sleep(0.1)
        refresher_task = asyncio.create_task(_refresher())
        try:
            await orch.run()
        except AgentFailed as e:
            print(f"\nAgent {e.agent_name!r} failed: {e.__cause__}", file=sys.stderr)
            return 3
        finally:
            refresher_task.cancel()
            try:
                await refresher_task
            except asyncio.CancelledError:
                pass
            live.update(renderer.build_table())

    articles = ctx.shared.get("articles", [])
    survey_slug = ctx.shared.get("survey_slug")
    print()
    print(f"  Articles distilled: {len(articles)}")
    print(f"  Survey slug:        {survey_slug or '(none)'}")
    print(f"  Tokens in/out:      {llm.total_tokens_in} / {llm.total_tokens_out}")

    # Render each distilled article body in the terminal (markdown → rich).
    if articles:
        from rich.markdown import Markdown
        from rich.rule import Rule
        for article in articles:
            console.print()
            console.print(Rule(f"[bold]{article.title}[/bold]  ([cyan]{article.slug}[/cyan])"))
            console.print(Markdown(article.body))
        console.print()
    return 0


def _render_browse_list(ranked, console) -> None:
    """Print N candidates with abstracts for user review."""
    from rich.rule import Rule
    console.print(Rule(f"[bold]{len(ranked)} candidates[/bold]"))
    for i, paper in enumerate(ranked, start=1):
        year = (paper.published or "")[:4] or "?"
        ident = paper.arxiv_id or paper.doi or paper.paper_id or "?"
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += f", et al. ({len(paper.authors) - 3} more)"
        abstract_preview = (paper.abstract or "(no abstract)")[:250].replace("\n", " ")
        if paper.abstract and len(paper.abstract) > 250:
            abstract_preview += "..."
        console.print(f"\n[bold][{i}][/bold] [cyan]{ident}[/cyan] ([dim]{year}[/dim])")
        console.print(f"    [bold]{paper.title}[/bold]")
        console.print(f"    [dim]─ {authors_str} ─[/dim]")
        console.print(f"    [dim]Abstract:[/dim] {abstract_preview}")
    console.print()


def _parse_picks(s: str, n: int) -> list[int] | None:
    """Parse user pick input. Returns sorted unique 1-based indices, or None on error/cancel.

    - "q" / "quit" / "exit" / "" → []  (explicit cancel)
    - "all" → range(1, n+1)
    - "1,3,5" → [1, 3, 5]
    - "1-3" → [1, 2, 3]
    - "1,3-5,7" → [1, 3, 4, 5, 7]
    - reversed range, out-of-range, non-numeric → None
    """
    s = s.strip().lower()
    if not s or s in ("q", "quit", "exit"):
        return []  # explicit cancel
    if s == "all":
        return list(range(1, n + 1))
    picks: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                lo, hi = (int(x.strip()) for x in part.split("-", 1))
            except ValueError:
                return None
            if lo > hi:
                return None
            for k in range(lo, hi + 1):
                picks.add(k)
        else:
            try:
                picks.add(int(part))
            except ValueError:
                return None
    if not picks or any(p < 1 or p > n for p in picks):
        return None
    return sorted(picks)


def _prompt_picks(n: int) -> list[int]:
    """Prompt user with retry. Returns empty list on cancel or 3 failed attempts."""
    for _attempt in range(3):
        try:
            raw = input(
                "Pick papers to distill (e.g. '1,3,5' or '1-5' or 'all'; 'q' to quit): "
            )
        except (KeyboardInterrupt, EOFError):
            return []
        picks = _parse_picks(raw, n)
        if picks is not None:
            return picks
        print(f"  invalid input. expected like '1,3-5' (range 1-{n}). Try again.")
    print("  (3 invalid attempts; exiting)")
    return []


async def _run_browse(args) -> int:
    try:
        cfg = load_config(
            vault_path=args.vault,
            topic=args.topic,
            author=args.author,
            n=args.n,
            pool=args.pool,
            source=args.source,
            dry_run=args.dry_run,
            verbose=args.verbose,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if cfg.dry_run:
        print(f"[DRY-RUN] Would browse {cfg.top_n} candidates for {cfg.topic!r}")
        return 0

    vault = VaultStore(cfg.vault_path)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    renderer = ConsoleRenderer(title=f"browse · {cfg.topic or cfg.author}")
    ctx = Context(cfg=cfg, llm=llm, vault=vault, shared={}, on_status=renderer.on_status)

    console = Console()

    # Phase 1: search + rank (no distill)
    browse_dag = DAG([
        ArxivSearcher(),
        SemanticScholarSearcher(),
        OpenCLIOpenAlexSearcher(),
        CandidateMerger(),
        CandidateRanker(),
    ])
    orch = Orchestrator(browse_dag, ctx)
    with Live(renderer.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher():
            while True:
                live.update(renderer.build_table())
                await asyncio.sleep(0.1)
        refresher_task = asyncio.create_task(_refresher())
        try:
            await orch.run()
        except AgentFailed as e:
            print(f"\nAgent {e.agent_name!r} failed: {e.__cause__}", file=sys.stderr)
            return 3
        finally:
            refresher_task.cancel()
            try:
                await refresher_task
            except asyncio.CancelledError:
                pass
            live.update(renderer.build_table())

    ranked = ctx.shared.get("ranked", [])
    if not ranked:
        print("\nNo candidates found.")
        return 0

    # Phase 2: render + pick
    print()
    _render_browse_list(ranked, console)
    picks = _prompt_picks(len(ranked))
    if not picks:
        print("  (no papers picked; exiting)")
        print(f"  Tokens used (search+rank): {llm.total_tokens_in} / {llm.total_tokens_out}")
        return 0

    # Phase 3: distill picked papers only
    selected = [ranked[i - 1] for i in picks]
    print(f"\n  Distilling {len(selected)} picked papers...")
    ctx.shared["ranked"] = selected      # replace ranked with user-curated list
    ctx.shared.pop("articles", None)     # clear any leftover state

    processor = PaperProcessor()
    processor.deps = []                  # bypass candidate-ranker dep
    distill_dag = DAG([
        processor,
        VaultWriter(),
        SurveyComposer(),
    ])
    # Reset renderer for clean second phase
    renderer2 = ConsoleRenderer(title=f"distill · {len(selected)} picked")
    ctx.on_status = renderer2.on_status
    orch2 = Orchestrator(distill_dag, ctx)
    with Live(renderer2.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher2():
            while True:
                live.update(renderer2.build_table())
                await asyncio.sleep(0.1)
        rt = asyncio.create_task(_refresher2())
        try:
            await orch2.run()
        except AgentFailed as e:
            print(f"\nAgent {e.agent_name!r} failed: {e.__cause__}", file=sys.stderr)
            return 3
        finally:
            rt.cancel()
            try:
                await rt
            except asyncio.CancelledError:
                pass
            live.update(renderer2.build_table())

    articles = ctx.shared.get("articles", [])
    survey_slug = ctx.shared.get("survey_slug")
    print()
    print(f"  Articles distilled: {len(articles)}")
    print(f"  Survey slug:        {survey_slug or '(none)'}")
    print(f"  Tokens in/out:      {llm.total_tokens_in} / {llm.total_tokens_out}")

    if articles:
        from rich.markdown import Markdown
        from rich.rule import Rule
        for article in articles:
            console.print()
            console.print(Rule(f"[bold]{article.title}[/bold]  ([cyan]{article.slug}[/cyan])"))
            console.print(Markdown(article.body))
        console.print()
    return 0


def _run_ask(args) -> int:
    try:
        cfg = load_config_qa(
            vault_path=args.vault,
            question=args.question,
            max_rounds=args.max_rounds,
            max_articles=args.max_articles,
            max_cost_cny=args.max_cost_cny,
            confidence_threshold=args.confidence_threshold,
            per_round=args.per_round,
            source=args.source,
            interactive=args.interactive,
            resume_session_id=None,
            verbose=args.verbose,
            dry_run=args.dry_run,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if cfg.dry_run:
        print(f"[DRY-RUN] Would run QA loop for {cfg.qa_question!r}")
        return 0
    try:
        summary = run_qa_loop(cfg)
    except Exception as e:
        print(f"\nError during QA loop: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 3
    print()
    print(f"  Session:      {summary['session_id']}")
    print(f"  Stop reason:  {summary['stop_reason']}")
    print(f"  Rounds:       {summary['rounds_completed']}")
    print(f"  Articles:     {summary['articles_distilled_count']}")
    print(f"  Cost:         CNY {summary['cost_cny']:.2f}")
    print(f"  Tokens:       {summary['tokens_in_total']} / {summary['tokens_out_total']}")
    return 0


def _run_resume(args) -> int:
    from ..qa.state import read_state
    from pathlib import Path
    existing = read_state(Path(args.vault), args.session_id)
    if existing is None:
        print(f"Error: session {args.session_id!r} not found in {args.vault}", file=sys.stderr)
        return 2
    try:
        cfg = load_config_qa(
            vault_path=args.vault,
            question=existing.question,
            max_rounds=existing.config_snapshot.get("max_rounds", 5),
            max_articles=existing.config_snapshot.get("max_articles", 15),
            max_cost_cny=existing.config_snapshot.get("max_cost_cny", 20.0),
            confidence_threshold=existing.config_snapshot.get("confidence_threshold", 8),
            per_round=existing.config_snapshot.get("per_round", 2),
            source=existing.config_snapshot.get("source", "both"),
            interactive=False,
            resume_session_id=args.session_id,
            verbose=args.verbose,
            dry_run=False,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    try:
        summary = run_qa_loop(cfg)
    except Exception as e:
        print(f"\nError during resume: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    print()
    print(f"  Session:      {summary['session_id']} (resumed)")
    print(f"  Stop reason:  {summary['stop_reason']}")
    print(f"  Rounds:       {summary['rounds_completed']}")
    print(f"  Articles:     {summary['articles_distilled_count']}")
    print(f"  Cost:         CNY {summary['cost_cny']:.2f}")
    return 0


def _run_research(args) -> int:
    try:
        duration_sec = _parse_duration(args.duration)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    try:
        cfg = load_config_research(
            vault_path=args.vault, question=args.question,
            max_papers=args.max_papers,
            max_cost_cny=args.max_cost_cny,
            max_duration_sec=duration_sec,
            source=args.source,
            resume_session_id=args.resume,
            verbose=args.verbose, dry_run=args.dry_run,
            model_override=args.model, provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if cfg.dry_run:
        print(f"[DRY-RUN] Would run deep research on {cfg.qa_question!r}")
        return 0
    try:
        summary = run_research_loop(cfg)
    except Exception as e:
        print(f"\nError during research: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 3
    print()
    print(f"  Session:        {summary['session_id']}")
    print(f"  Stop reason:    {summary['stop_reason']}")
    print(f"  Papers:         {summary['papers_distilled_count']}")
    print(f"  Themes:         {summary['themes_count']}")
    print(f"  Syntheses:      {summary['synthesis_count']}")
    print(f"  Final report:   {summary['final_report_slug'] or '(none)'}")
    print(f"  Iterations:     {summary['iterations_completed']}")
    print(f"  Cost:           CNY {summary['total_cost_cny']:.2f}")
    print(f"  Tokens:         {summary['total_tokens_in']} / {summary['total_tokens_out']}")
    return 0


def _run_agent(args) -> int:
    """Launch the v1.4 conversational AgentLoop."""
    api_key = os.getenv("PD_API_KEY")
    base_url = os.getenv("PD_BASE_URL")
    model = getattr(args, "model", None) or os.getenv("PD_MODEL")
    if not api_key or not base_url or not model:
        print(
            "Error: PD_API_KEY / PD_BASE_URL / PD_MODEL must all be set "
            "(via env or .env). See .env.example.",
            file=sys.stderr,
        )
        return 2
    llm = LLMClient(api_key=api_key, base_url=base_url, model=model)

    console = Console()

    def _announce(name: str, arguments: dict) -> None:
        arg_preview = ", ".join(
            f"{k}={_short(v)}" for k, v in list(arguments.items())[:4]
        )
        console.print(f"[dim]→ tool: [cyan]{name}[/cyan]({arg_preview})[/dim]")

    loop = AgentLoop(
        llm=llm,
        vault_path=args.vault,
        console=console,
        on_tool_call=_announce,
    )
    return loop.run()


def _short(v) -> str:
    """One-line repr for tool-call argument preview."""
    s = repr(v)
    return s if len(s) <= 60 else s[:57] + "..."


def _run_legacy_repl(args) -> int:
    """Launch the pre-v1.4 slash-command REPL."""
    repl = REPL(vault_path=args.vault)
    return repl.run()


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "distill":
        return asyncio.run(_run_distill(args))
    if args.subcommand == "browse":
        return asyncio.run(_run_browse(args))
    if args.subcommand == "ask":
        return _run_ask(args)  # run_qa_loop wraps asyncio.run internally
    if args.subcommand == "resume":
        return _run_resume(args)
    if args.subcommand == "research":
        return _run_research(args)
    if args.subcommand == "legacy-repl":
        return _run_legacy_repl(args)
    # No subcommand: launch v1.4 conversational agent (requires --vault)
    if not getattr(args, "vault", None):
        print(
            "Error: --vault is required when launching the conversational agent.\n"
            "  paper-distiller-chat --vault /path/to/vault",
            file=sys.stderr,
        )
        return 2
    return _run_agent(args)


if __name__ == "__main__":
    sys.exit(main())
