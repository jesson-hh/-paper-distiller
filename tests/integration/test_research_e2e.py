"""End-to-end integration test for paper-distiller-chat research."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i):
    return Paper(
        source="arxiv", paper_id=f"2501.000{i}", arxiv_id=f"2501.000{i}",
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://x/{i}.pdf", published="2025-01-01", categories=[],
    )


def test_research_e2e(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # Phase 1 reflection (says done so we move on quickly)
    mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": True, "confidence": 9,
            "what_we_know": "...", "what_is_missing": "",
            "next_query": "diffusion", "next_query_rationale": "",
            "suggest_stop": False,
        },
    )
    # Sources
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(1), _paper(2)],
    )
    mocker.patch("paper_distiller.agents.searchers.ss_search", return_value=[])
    # Ranker
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda c, t, n, llm: c[:n],
    )
    # Fetch + distill
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda p, ft, wi, llm: ArticleResult(
            slug=f"a-{p.arxiv_id}", title=f"T-{p.arxiv_id}",
            body=f"# T-{p.arxiv_id}\n\nBody.", tags=["t"],
            refs=[f"arxiv:{p.arxiv_id}"], depth="full-pdf",
        ),
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    # Citation explorer returns nothing -> expand phase is no-op
    mocker.patch(
        "paper_distiller.agents.citation_explorer.ss_paper_refs",
        return_value=[],
    )
    # SurveyComposer (compose) - used by synthesize phase
    mocker.patch(
        "paper_distiller.chat.research_runner.compose_survey",
        return_value=SimpleNamespace(
            slug="synth", title="Synthesis title",
            body="synthesis body", tags=["synthesis"],
            related_articles=[],
        ),
    )
    # LLMClient at the research_runner import site - handles cluster + extract + gap LLM calls
    fake_llm_class = mocker.patch("paper_distiller.chat.research_runner.LLMClient")
    llm_instance = fake_llm_class.return_value
    llm_instance.total_tokens_in = 1000
    llm_instance.total_tokens_out = 500

    def _llm_complete(messages, **kw):
        content = messages[0]["content"]
        if "聚类" in content or "themes" in content.lower():
            return json.dumps({
                "themes": [
                    {"name": "Theory", "description": "...",
                     "slugs": ["a-2501.0001", "a-2501.0002"]},
                ],
            })
        if "theorems" in content.lower() or "假设" in content:
            return json.dumps({
                "theorems": ["T1"], "assumptions": ["A1"],
                "convergence_rates": [], "key_lemmas": [],
            })
        if "覆盖度" in content or "gap" in content.lower():
            return json.dumps({
                "should_continue": False,
                "missing_aspects": [],
                "next_query": "",
                "rationale": "Coverage sufficient.",
            })
        return "{}"
    llm_instance.complete.side_effect = _llm_complete

    vault = tmp_path / "vault"
    vault.mkdir()

    from paper_distiller.chat.cli import main
    rc = main([
        "research", "--vault", str(vault), "--question", "why diffusion?",
        "--max-papers", "2", "--max-cost-cny", "5", "--duration", "300s",
    ])
    assert rc == 0

    # Vault must have articles and at least one survey
    articles = list((vault / "articles").glob("*.md"))
    assert len(articles) >= 1, f"no articles written; vault={list(vault.rglob('*.md'))}"

    surveys = list((vault / "surveys").glob("*.md"))
    assert len(surveys) >= 1, "no survey/synthesis/research-report written"

    # state.json present
    state_files = list((vault / ".paper_distiller" / "research-sessions").glob("*/state.json"))
    assert len(state_files) == 1, f"expected exactly 1 research session, got {len(state_files)}"
    data = json.loads(state_files[0].read_text(encoding="utf-8"))
    assert data["is_done"] is True
    # Stop reason is either max_papers or all_themes_synthesized
    assert data["stop_reason"] in (
        "max_papers", "all_themes_synthesized", "max_cost", "max_duration",
    )
