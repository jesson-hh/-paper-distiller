"""Tests for VaultWriter + SurveyComposer agents."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.writer import VaultWriter, SurveyComposer
from paper_distiller.distill.article import ArticleResult


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _ctx(articles, **cfg_overrides):
    cfg = SimpleNamespace(
        min_papers_for_survey=2,
        topic="t", verbose=False,
        **cfg_overrides,
    )
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"articles": articles},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_vault_writer_calls_save_entry_per_article():
    arts = [_article("a"), _article("b")]
    ctx = _ctx(arts)
    ctx.vault.save_entry = MagicMock(side_effect=lambda **kw: {"slug": kw["slug"]})
    out = await VaultWriter().run(ctx)
    assert ctx.vault.save_entry.call_count == 2
    assert set(out["saved_slugs"]) == {"a", "b"}


@pytest.mark.asyncio
async def test_vault_writer_empty_articles_is_noop():
    ctx = _ctx([])
    ctx.vault.save_entry = MagicMock()
    out = await VaultWriter().run(ctx)
    ctx.vault.save_entry.assert_not_called()
    assert out["saved_slugs"] == []


@pytest.mark.asyncio
async def test_vault_writer_deps():
    assert VaultWriter().deps == ["paper-processor"]


@pytest.mark.asyncio
async def test_survey_composer_skipped_when_below_min(mocker):
    """fewer than min_papers_for_survey -> skip."""
    fake_compose = mocker.patch("paper_distiller.agents.writer.compose_survey")
    ctx = _ctx([_article("a")])  # 1 article, min=2
    out = await SurveyComposer().run(ctx)
    fake_compose.assert_not_called()
    assert out["survey_slug"] is None


@pytest.mark.asyncio
async def test_survey_composer_runs_when_above_min(mocker):
    # SurveyResult is a dataclass — mock returns SimpleNamespace with same fields
    fake_compose = mocker.patch(
        "paper_distiller.agents.writer.compose_survey",
        return_value=SimpleNamespace(
            slug="s-1", title="S", body="...", tags=["t"], related_articles=[],
        ),
    )
    # load_index is called inside SurveyComposer to fetch wiki_index for compose()
    mocker.patch(
        "paper_distiller.agents.writer.load_index",
        return_value=MagicMock(),
    )
    arts = [_article("a"), _article("b"), _article("c")]
    ctx = _ctx(arts)
    ctx.vault.save_entry = MagicMock(side_effect=lambda **kw: {"slug": kw["slug"]})
    out = await SurveyComposer().run(ctx)
    fake_compose.assert_called_once()
    assert out["survey_slug"] == "s-1"


@pytest.mark.asyncio
async def test_survey_composer_deps():
    assert SurveyComposer().deps == ["vault-writer"]
