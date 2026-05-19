"""VaultWriter — saves distilled articles to vault.
SurveyComposer — composes optional cross-article survey."""

from __future__ import annotations

import asyncio

from ..distill.survey import compose as compose_survey
from ..vault.crosslink import load_index
from .base import Context


class VaultWriter:
    name = "vault-writer"
    deps = ["paper-processor"]

    async def run(self, ctx: Context) -> dict:
        articles = ctx.shared.get("articles", [])
        saved = []
        for article in articles:
            await asyncio.to_thread(
                ctx.vault.save_entry,
                category="articles",
                **article.to_save_kwargs(),
            )
            saved.append(article.slug)
        return {"saved_slugs": saved}


class SurveyComposer:
    name = "survey-composer"
    deps = ["vault-writer"]

    async def run(self, ctx: Context) -> dict:
        articles = ctx.shared.get("articles", [])
        if len(articles) < ctx.cfg.min_papers_for_survey:
            return {"survey_slug": None}
        wiki_index = await asyncio.to_thread(load_index, ctx.vault)
        survey = await asyncio.to_thread(
            compose_survey, articles, ctx.cfg.topic or "", wiki_index, ctx.llm,
        )
        saved = await asyncio.to_thread(
            ctx.vault.save_entry,
            category="surveys",
            title=survey.title,
            body=survey.body,
            tags=survey.tags or [],
            refs=[f"articles:{a.slug}" for a in articles],
            slug=survey.slug,
        )
        return {"survey_slug": saved["slug"]}
