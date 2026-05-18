import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paper_distiller.config import Config
from paper_distiller.sources.arxiv import ArxivPaper
from paper_distiller.distill.article import ArticleResult
from paper_distiller.distill.survey import SurveyResult


def _config(tmp_path):
    return Config(
        vault_path=tmp_path / "vault",
        topic="diffusion",
        author=None,
        top_n=2, pool=10, force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
    )


def _paper(i):
    return ArxivPaper(
        arxiv_id=f"2501.0000{i}", title=f"Paper {i}", authors=["A"],
        abstract=f"abstract {i}", pdf_url=f"https://arxiv.org/pdf/2501.0000{i}.pdf",
        published="2025-01-01", categories=["math.AT"],
    )


def test_pipeline_dry_run_makes_no_external_calls(tmp_path, mocker):
    from paper_distiller.pipeline import run
    cfg = _config(tmp_path); cfg.dry_run = True
    cfg.vault_path.mkdir()

    mock_search = mocker.patch("paper_distiller.pipeline.arxiv_search")
    mock_llm_cls = mocker.patch("paper_distiller.pipeline.LLMClient")
    run(cfg)
    mock_search.assert_not_called()
    mock_llm_cls.assert_not_called()


def test_pipeline_happy_path(tmp_path, mocker):
    from paper_distiller.pipeline import run
    cfg = _config(tmp_path)
    cfg.vault_path.mkdir()

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1), _paper(2), _paper(3)])
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1), _paper(2)])
    mocker.patch("paper_distiller.pipeline.download_pdf",
                 side_effect=lambda p, d, **k: Path(d) / f"{p.arxiv_id}.pdf")
    mocker.patch("paper_distiller.pipeline.extract_text",
                 return_value="x" * 1000)  # > 500 -> full-pdf depth

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(
            slug=f"paper-{paper.arxiv_id}",
            title=f"Title {paper.arxiv_id}",
            body=f"# {paper.title}\n\nbody",
            tags=["t"], refs=[f"arxiv:{paper.arxiv_id}"],
            depth="full-pdf",
        )
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey",
                 return_value=SurveyResult(
                     slug="diffusion-survey-20260502",
                     title="Diffusion Survey",
                     body="# Survey\n\n[[paper-2501.00001]] [[paper-2501.00002]]",
                     tags=["survey"], related_articles=["paper-2501.00001", "paper-2501.00002"],
                 ))
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    arts = sorted((cfg.vault_path / "articles").glob("*.md"))
    surveys = sorted((cfg.vault_path / "surveys").glob("*.md"))
    assert len(arts) == 2
    assert len(surveys) == 1
    runs_log = cfg.vault_path / ".paper_distiller" / "runs.jsonl"
    assert runs_log.exists()
    line = json.loads(runs_log.read_text().strip().split("\n")[-1])
    assert line["distilled"] == 2
    assert line["survey_slug"] == "diffusion-survey-20260502"


def test_pipeline_dedup_skips_existing(tmp_path, mocker):
    from paper_distiller.pipeline import run
    from paper_distiller.vault.store import VaultStore
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    # Pre-populate one article
    store = VaultStore(cfg.vault_path)
    store.save_entry(title="Title 2501.00001", category="articles",
                     body="x", slug="paper-2501.00001")

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1), _paper(2)])
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1), _paper(2)])
    mocker.patch("paper_distiller.pipeline.download_pdf",
                 side_effect=lambda p, d, **k: Path(d) / f"{p.arxiv_id}.pdf")
    mocker.patch("paper_distiller.pipeline.extract_text", return_value="x" * 1000)

    def fake_distill(paper, full_text, wiki_index, llm):
        return ArticleResult(slug=f"paper-{paper.arxiv_id}",
                             title=f"Title {paper.arxiv_id}",
                             body="b", tags=[], refs=[], depth="full-pdf")
    mocker.patch("paper_distiller.pipeline.distill_article", side_effect=fake_distill)
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    line = json.loads((cfg.vault_path / ".paper_distiller" / "runs.jsonl").read_text().strip().split("\n")[-1])
    assert line["skipped_dedup"] == 1
    assert line["distilled"] == 1


def test_pipeline_arxiv_id_dedup_skips_existing(tmp_path, mocker):
    """If the vault already has an article with refs containing this arxiv id,
    skip — even if the slug pattern doesn't match. Fixes the v0.1 issue where
    cofindiff.md and cofindiff-controllable-financial-diffusion.md could both
    exist for the same arxiv paper."""
    from paper_distiller.pipeline import run
    from paper_distiller.vault.store import VaultStore
    cfg = _config(tmp_path); cfg.vault_path.mkdir()
    store = VaultStore(cfg.vault_path)
    # Pre-populate with a hand-written-style entry: slug doesn't match arxiv pattern,
    # but refs contains the arxiv id of the candidate we'll search for.
    store.save_entry(
        title="CoFinDiff (hand-written)",
        category="articles",
        body="pre-existing hand-written content",
        refs=["arxiv:2501.00001"],
        slug="cofindiff-handwritten",
    )

    mocker.patch("paper_distiller.pipeline.arxiv_search",
                 return_value=[_paper(1)])  # _paper(1) has arxiv_id "2501.00001"
    mocker.patch("paper_distiller.pipeline.rank",
                 return_value=[_paper(1)])
    mock_distill = mocker.patch("paper_distiller.pipeline.distill_article")
    mocker.patch("paper_distiller.pipeline.compose_survey")
    mocker.patch("paper_distiller.pipeline.LLMClient")

    run(cfg)

    log = (cfg.vault_path / ".paper_distiller" / "runs.jsonl").read_text()
    line = json.loads(log.strip().split("\n")[-1])
    assert line["skipped_dedup"] == 1
    assert line["distilled"] == 0
    # Critically: distill_article was never called — the skip happened upstream
    mock_distill.assert_not_called()
