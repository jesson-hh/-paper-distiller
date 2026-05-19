"""End-to-end integration test for paper-distiller-chat distill — all subsystems mocked."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i):
    return Paper(
        source="arxiv", paper_id=f"2501.0000{i}", arxiv_id=f"2501.0000{i}",
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://x/{i}.pdf", published="2025-01-01", categories=[],
    )


def test_distill_e2e_writes_articles_to_vault(mocker, tmp_path, monkeypatch):
    """`paper-distiller-chat distill --vault tmp --topic X --n 2`
    should write 2 articles + 1 survey to the vault."""
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # Mock source searches
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(1), _paper(2), _paper(3)],
    )
    mocker.patch(
        "paper_distiller.agents.searchers.ss_search",
        return_value=[],
    )
    # Mock rank
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    # Mock fetch + distill
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda paper, full_text, wiki_index, llm: ArticleResult(
            slug=f"a-{paper.arxiv_id}", title=f"T-{paper.arxiv_id}",
            body=f"body {paper.arxiv_id}", tags=["t"], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        ),
    )
    # Mock survey composer (SurveyResult dataclass — mock returns SimpleNamespace)
    from types import SimpleNamespace
    mocker.patch(
        "paper_distiller.agents.writer.compose_survey",
        return_value=SimpleNamespace(
            slug="session-survey-1", title="S", body="...", tags=["s"], related_articles=[],
        ),
    )
    # Mock processor's load_index call
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set(), to_prompt_lines=lambda: []),
    )
    # Mock writer's load_index call (used by SurveyComposer for compose's wiki_index arg)
    mocker.patch(
        "paper_distiller.agents.writer.load_index",
        return_value=MagicMock(),
    )

    vault = tmp_path / "vault"
    vault.mkdir()

    from paper_distiller.chat.cli import main
    rc = main(["distill", "--vault", str(vault), "--topic", "diffusion", "--n", "2"])
    assert rc == 0

    # Vault should have 2 articles + 1 survey
    articles_dir = vault / "articles"
    surveys_dir = vault / "surveys"
    assert articles_dir.exists()
    assert surveys_dir.exists()
    assert len(list(articles_dir.glob("*.md"))) == 2
    assert len(list(surveys_dir.glob("*.md"))) == 1
