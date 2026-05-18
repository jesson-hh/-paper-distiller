"""paper-distiller command-line entry point."""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import run as pipeline_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper-distiller",
        description="Distill arxiv papers into an Obsidian-compatible markdown wiki.",
    )
    p.add_argument("--vault", required=True, help="Path to your Obsidian vault.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--topic", help="Research topic to search arxiv for.")
    group.add_argument("--author", help="Author name to search arxiv for.")
    p.add_argument("--n", type=int, default=5, help="Top-N after filtering (default 5).")
    p.add_argument("--pool", type=int, default=30, help="Candidate pool size (default 30).")
    p.add_argument("--force", action="store_true", help="Overwrite existing article slugs.")
    p.add_argument("--dry-run", action="store_true", help="Plan only; no LLM, no writes.")
    p.add_argument("--verbose", "-v", action="store_true", help="Detailed logging.")
    p.add_argument("--model", help="Override PD_MODEL env var.")
    p.add_argument("--provider", help="Override PD_PROVIDER_NAME label.")
    return p


def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_config(
            vault_path=args.vault,
            topic=args.topic,
            author=args.author,
            n=args.n,
            pool=args.pool,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        summary = pipeline_run(cfg)
    except Exception as e:
        print(f"\nError during pipeline run: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        else:
            print("(run with --verbose for full traceback)", file=sys.stderr)
        return 3

    print()
    print(f"  Distilled:     {summary['distilled']}")
    print(f"  Skipped dedup: {summary['skipped_dedup']}")
    print(f"  Skipped error: {summary['skipped_failed']}")
    if summary["article_slugs"]:
        print(f"  Article slugs: {', '.join(summary['article_slugs'])}")
    if summary.get("survey_slug"):
        print(f"  Survey:        {summary['survey_slug']}")
    print(f"  Duration:      {summary['duration_sec']}s")
    print(f"  Tokens in/out: {summary['tokens_in_total']} / {summary['tokens_out_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
