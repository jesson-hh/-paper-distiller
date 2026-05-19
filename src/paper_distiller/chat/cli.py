"""paper-distiller-chat entry point.

In Plan 1, supports only the one-shot `distill` subcommand. Plan 2 adds
`ask` + `resume`; Plan 3 adds the interactive REPL.
"""

from __future__ import annotations

import argparse
import asyncio
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
    return p


def _parse_duration(s: str) -> int:
    """Parse '4h' / '30m' / '1h30m' / '3600s' → seconds."""
    import re
    m = re.match(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", s.strip())
    if not m or not any(m.groups()):
        raise ValueError(f"invalid duration: {s!r}")
    h, mn, sc = (int(g or 0) for g in m.groups())
    total = h * 3600 + mn * 60 + sc
    if total < 60:
        raise ValueError(f"duration too short: {total}s (min 60s)")
    return total


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


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "distill":
        return asyncio.run(_run_distill(args))
    if args.subcommand == "ask":
        return _run_ask(args)  # run_qa_loop wraps asyncio.run internally
    if args.subcommand == "resume":
        return _run_resume(args)
    if args.subcommand == "research":
        return _run_research(args)
    # No subcommand: launch REPL (requires --vault)
    if not getattr(args, "vault", None):
        print("Error: --vault is required when launching REPL", file=sys.stderr)
        return 2
    repl = REPL(vault_path=args.vault)
    return repl.run()


if __name__ == "__main__":
    sys.exit(main())
