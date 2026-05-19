"""Configuration loading from env vars + CLI args."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Auto-load .env from current working dir if present
load_dotenv()


@dataclass
class Config:
    vault_path: Path
    topic: str | None
    author: str | None
    top_n: int
    pool: int
    force: bool
    dry_run: bool
    verbose: bool

    api_key: str
    base_url: str
    model: str
    provider_name: str

    pdf_timeout_sec: int
    min_papers_for_survey: int

    source: str = "both"
    ss_api_key: str | None = None

    # QA loop (v0.5) — only used by paper-distiller-qa entry
    qa_max_rounds: int = 5
    qa_max_articles: int = 15
    qa_max_cost_cny: float = 20.0
    qa_confidence_threshold: int = 8
    qa_per_round: int | None = None
    qa_interactive: bool = False
    qa_resume_session_id: str | None = None
    qa_question: str | None = None


def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Required env var {name} not set. See .env.example.")
    return v


def load_config(
    vault_path: Path | str,
    topic: str | None = None,
    author: str | None = None,
    n: int = 5,
    pool: int = 30,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    source: str = "both",
    model_override: str | None = None,
    provider_override: str | None = None,
) -> Config:
    """Build a Config from CLI args + env vars."""
    if not topic and not author:
        raise ValueError("Either topic or author must be provided.")
    if source not in ("arxiv", "ss", "both"):
        raise ValueError(f"source must be one of arxiv/ss/both (got {source!r})")

    return Config(
        vault_path=Path(vault_path),
        topic=topic,
        author=author,
        top_n=n,
        pool=pool,
        force=force,
        dry_run=dry_run,
        verbose=verbose,
        source=source,
        ss_api_key=os.getenv("PD_SS_API_KEY") or None,
        api_key=_require("PD_API_KEY"),
        base_url=_require("PD_BASE_URL"),
        model=model_override or _require("PD_MODEL"),
        provider_name=provider_override or os.getenv("PD_PROVIDER_NAME", "unspecified"),
        pdf_timeout_sec=int(os.getenv("PD_PDF_TIMEOUT", "60")),
        min_papers_for_survey=int(os.getenv("PD_MIN_SURVEY", "2")),
    )


def load_config_qa(
    vault_path: Path | str,
    question: str,
    max_rounds: int = 5,
    max_articles: int = 15,
    max_cost_cny: float = 20.0,
    confidence_threshold: int = 8,
    per_round: int = 2,
    source: str = "both",
    interactive: bool = False,
    resume_session_id: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    model_override: str | None = None,
    provider_override: str | None = None,
) -> Config:
    """Build a Config for paper-distiller-qa. Mirrors load_config() but with
    the QA-loop specific fields populated."""
    if not question or not question.strip():
        raise ValueError("question is required and must be non-empty")
    if source not in ("arxiv", "ss", "both"):
        raise ValueError(f"source must be one of arxiv/ss/both (got {source!r})")
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1 (got {max_rounds})")
    if max_articles < 1:
        raise ValueError(f"max_articles must be >= 1 (got {max_articles})")
    if max_cost_cny <= 0:
        raise ValueError(f"max_cost_cny must be > 0 (got {max_cost_cny})")
    if not (0 <= confidence_threshold <= 10):
        raise ValueError(f"confidence_threshold must be in [0, 10] (got {confidence_threshold})")
    if per_round < 1:
        raise ValueError(f"per_round must be >= 1 (got {per_round})")

    return Config(
        vault_path=Path(vault_path),
        topic=None,
        author=None,
        top_n=per_round,
        pool=30,
        force=False,
        dry_run=dry_run,
        verbose=verbose,
        api_key=_require("PD_API_KEY"),
        base_url=_require("PD_BASE_URL"),
        model=model_override or _require("PD_MODEL"),
        provider_name=provider_override or os.getenv("PD_PROVIDER_NAME", "unspecified"),
        pdf_timeout_sec=int(os.getenv("PD_PDF_TIMEOUT", "60")),
        min_papers_for_survey=int(os.getenv("PD_MIN_SURVEY", "2")),
        source=source,
        ss_api_key=os.getenv("PD_SS_API_KEY") or None,
        qa_max_rounds=max_rounds,
        qa_max_articles=max_articles,
        qa_max_cost_cny=max_cost_cny,
        qa_confidence_threshold=confidence_threshold,
        qa_per_round=per_round,
        qa_interactive=interactive,
        qa_resume_session_id=resume_session_id,
        qa_question=question,
    )
