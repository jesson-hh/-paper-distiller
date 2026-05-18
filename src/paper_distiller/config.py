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
    model_override: str | None = None,
    provider_override: str | None = None,
) -> Config:
    """Build a Config from CLI args + env vars."""
    if not topic and not author:
        raise ValueError("Either topic or author must be provided.")

    return Config(
        vault_path=Path(vault_path),
        topic=topic,
        author=author,
        top_n=n,
        pool=pool,
        force=force,
        dry_run=dry_run,
        verbose=verbose,
        api_key=_require("PD_API_KEY"),
        base_url=_require("PD_BASE_URL"),
        model=model_override or _require("PD_MODEL"),
        provider_name=provider_override or os.getenv("PD_PROVIDER_NAME", "unspecified"),
        pdf_timeout_sec=int(os.getenv("PD_PDF_TIMEOUT", "60")),
        min_papers_for_survey=int(os.getenv("PD_MIN_SURVEY", "2")),
    )
