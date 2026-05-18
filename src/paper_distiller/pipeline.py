"""End-to-end orchestrator.

Wire-up only. Business logic lives in the subsystems.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import Config
from .distill.article import distill as distill_article
from .distill.filter import rank
from .distill.survey import compose as compose_survey
from .extract.pymupdf_extractor import extract_text
from .llm.openai_compatible import LLMClient, LLMError
from .sources.arxiv import search as arxiv_search, download_pdf
from .vault.crosslink import load_index
from .vault.store import VaultStore, slugify


def _query_for(cfg: Config) -> str:
    if cfg.topic:
        return cfg.topic
    return f"author:{cfg.author}"


def _emit_summary(run_record: dict, vault_path: Path) -> None:
    log_dir = vault_path / ".paper_distiller"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "runs.jsonl").open("a", encoding="utf-8").write(
        json.dumps(run_record, ensure_ascii=False) + "\n"
    )


def run(cfg: Config) -> dict:
    """Execute one L2 pipeline run. Returns the run summary dict."""
    start = time.time()
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "topic": cfg.topic or f"author:{cfg.author}",
        "n_requested": cfg.top_n,
        "candidates_found": 0,
        "after_filter": 0,
        "distilled": 0,
        "skipped_dedup": 0,
        "skipped_failed": 0,
        "depth_breakdown": {"full-pdf": 0, "abstract-only": 0},
        "article_slugs": [],
        "survey_slug": None,
        "duration_sec": 0,
        "tokens_in_total": 0,
        "tokens_out_total": 0,
    }

    if cfg.dry_run:
        print(f"[DRY-RUN] Would search arxiv for '{_query_for(cfg)}' "
              f"(pool={cfg.pool}, top_n={cfg.top_n}), distill PDFs, "
              f"and write to {cfg.vault_path}.")
        return summary

    store = VaultStore(cfg.vault_path)
    wiki_index = load_index(store)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)

    # 3. arxiv search
    candidates = arxiv_search(_query_for(cfg), max_results=cfg.pool)
    summary["candidates_found"] = len(candidates)
    if not candidates:
        summary["duration_sec"] = round(time.time() - start, 1)
        _emit_summary(summary, cfg.vault_path)
        return summary

    # 4. filter
    top = rank(candidates, cfg.topic or cfg.author, cfg.top_n, llm)
    summary["after_filter"] = len(top)

    # 5. per-paper loop
    articles = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for paper in top:
            # arxiv-id-based dedup: search vault for any article whose `refs` contains
            # this arxiv id. Catches duplicates where slug pattern differs from our
            # conventions (e.g. user hand-wrote the entry with a longer descriptive slug).
            if not cfg.force:
                existing = store.find_by_arxiv_id(paper.arxiv_id)
                if existing is not None:
                    summary["skipped_dedup"] += 1
                    if cfg.verbose:
                        print(f"  skipping arxiv:{paper.arxiv_id} — already in "
                              f"articles/{existing.slug}.md")
                    continue

            # Slug-based fallback: legacy entries without refs metadata, or future
            # callers that bypass our save_entry conventions.
            arxiv_slug = f"paper-{paper.arxiv_id}"
            title_slug = slugify(paper.title)
            if not cfg.force and (
                store.slug_exists("articles", arxiv_slug)
                or store.slug_exists("articles", title_slug)
            ):
                summary["skipped_dedup"] += 1
                continue

            full_text = ""
            try:
                pdf_path = download_pdf(paper, Path(tmpdir), timeout=cfg.pdf_timeout_sec)
                full_text = extract_text(pdf_path)
            except Exception as e:
                if cfg.verbose:
                    print(f"  PDF fetch/parse failed for {paper.arxiv_id}: {e}; using abstract.")

            try:
                article = distill_article(paper, full_text, wiki_index, llm)
            except LLMError as e:
                summary["skipped_failed"] += 1
                if cfg.verbose:
                    print(f"  LLM distill failed for {paper.arxiv_id}: {e}")
                continue

            saved = store.save_entry(
                category="articles",
                **article.to_save_kwargs(),
            )
            articles.append(article)
            summary["distilled"] += 1
            summary["article_slugs"].append(saved["slug"])
            summary["depth_breakdown"][article.depth] += 1

    # 6. survey
    if len(articles) >= cfg.min_papers_for_survey:
        try:
            survey = compose_survey(articles, cfg.topic or cfg.author, wiki_index, llm)
            store.save_entry(category="surveys", **survey.to_save_kwargs())
            summary["survey_slug"] = survey.slug
        except LLMError as e:
            if cfg.verbose:
                print(f"  Survey composition failed: {e}")

    summary["duration_sec"] = round(time.time() - start, 1)
    try:
        summary["tokens_in_total"] = int(llm.total_tokens_in)
        summary["tokens_out_total"] = int(llm.total_tokens_out)
    except (TypeError, ValueError):
        # llm.total_tokens_* may be non-numeric in tests with a mocked LLMClient.
        summary["tokens_in_total"] = 0
        summary["tokens_out_total"] = 0

    _emit_summary(summary, cfg.vault_path)
    return summary
