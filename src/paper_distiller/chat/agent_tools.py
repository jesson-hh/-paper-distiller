"""Agent tool definitions + Python wrappers for the conversational agent loop.

Each tool exposes:
  1. an OpenAI tools-format JSON schema (entry in TOOL_SCHEMAS), and
  2. a synchronous Python wrapper (entry in TOOL_FUNCTIONS) that takes parsed
     kwargs and returns a JSON-serializable dict.

Wrappers catch their own exceptions and return {"error": "<type>: <msg>"} so
the agent loop never crashes on a tool failure. Status events from the DAG
are routed to a ConsoleRenderer; rendering (rich.live.Live) is the agent
loop's responsibility, not the tool's.

Wrappers are synchronous. Each internally calls ``asyncio.run()``, so they
MUST be invoked from a synchronous caller — calling them from inside a
running event loop will raise ``RuntimeError: asyncio.run() cannot be
called from a running event loop``. The agent loop dispatching these
tools must execute them via a sync function or a ``run_in_executor`` hop.

This module deliberately wraps existing functionality (search/distill/QA/
research runners) — it does not duplicate orchestration logic.
"""

from __future__ import annotations

import asyncio
from typing import Callable

from ..agents.base import Context
from ..agents.curation import CandidateMerger, CandidateRanker
from ..agents.dag import DAG
from ..agents.opencli_openalex import OpenCLIOpenAlexSearcher
from ..agents.orchestrator import Orchestrator
from ..agents.processor import PaperProcessor
from ..agents.renderer import ConsoleRenderer
from ..agents.searchers import ArxivSearcher, SemanticScholarSearcher
from ..agents.writer import SurveyComposer, VaultWriter
from ..config import load_config, load_config_qa, load_config_research
from ..llm.openai_compatible import LLMClient
from ..vault.store import VaultStore
from ._durations import parse_duration as _parse_duration
from .qa_runner import run_qa_loop
from .research_runner import run_research_loop


__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_FUNCTIONS",
    "execute_tool",
    "tool_search",
    "tool_distill_by_id",
    "tool_show",
    "tool_ask",
    "tool_research",
]


# ---------------------------------------------------------------------------
# JSON schemas (OpenAI tools format)
# ---------------------------------------------------------------------------

_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search arxiv + Semantic Scholar + OpenAlex in parallel for a "
            "research topic or author. Returns ranked candidates with titles, "
            "authors, and short abstracts — no PDF download, no distillation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Natural-language topic or author name.",
                },
                "n": {
                    "type": "integer",
                    "description": "How many candidates to return (default 10).",
                    "default": 10,
                },
                "source": {
                    "type": "string",
                    "enum": ["arxiv", "ss", "openalex", "all"],
                    "description": "Which source(s) to search (default 'all').",
                    "default": "all",
                },
            },
            "required": ["topic"],
        },
    },
}


_DISTILL_BY_ID_SCHEMA = {
    "type": "function",
    "function": {
        "name": "distill_by_id",
        "description": (
            "Download and distill a list of papers by ID (arxiv id, DOI, or "
            "Semantic Scholar paperId, typically obtained from a prior search "
            "result). Saves articles to the vault and composes an optional "
            "survey. ALWAYS pass `topic` with the same query you used in the "
            "preceding `search` call when possible — passing only IDs without "
            "a topic often returns matched_count: 0 because the underlying "
            "search treats IDs as opaque keywords. If you must call without "
            "topic, expect higher unmatched rates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of paper IDs to distill. Each must match an "
                        "arxiv_id, DOI, or paper_id seen in a prior search."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": (
                        "Search query that retrieved these IDs. STRONGLY "
                        "RECOMMENDED — without it, ID resolution may fail."
                    ),
                },
            },
            "required": ["ids"],
        },
    },
}


_SHOW_SCHEMA = {
    "type": "function",
    "function": {
        "name": "show",
        "description": (
            "Read a saved vault entry by slug and return its markdown body, "
            "tags, refs, and links."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": (
                        "Vault slug, e.g. 'latent-schrodinger-bridge-diffusion'."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": [
                        "articles",
                        "techniques",
                        "directions",
                        "open-problems",
                        "authors",
                        "surveys",
                    ],
                    "description": "Vault category (default 'articles').",
                    "default": "articles",
                },
            },
            "required": ["slug"],
        },
    },
}


_ASK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ask",
        "description": (
            "Ask a research question; runs a multi-round QA loop that "
            "alternates search + distill until the question is answered or a "
            "budget is exhausted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The research question.",
                },
                "max_rounds": {
                    "type": "integer",
                    "description": "Cap on QA rounds (default 3).",
                    "default": 3,
                },
                "per_round": {
                    "type": "integer",
                    "description": "How many papers to distill per round (default 2).",
                    "default": 2,
                },
                "max_cost_cny": {
                    "type": "number",
                    "description": "Cost ceiling in CNY (default 5.0).",
                    "default": 5.0,
                },
                "max_articles": {
                    "type": "integer",
                    "description": "Cap on total articles distilled (default 10).",
                    "default": 10,
                },
            },
            "required": ["question"],
        },
    },
}


_RESEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "research",
        "description": (
            "Long-running autonomous deep-research mode: 5-phase loop "
            "(seed → expand → structure → synthesize → gap-check) that "
            "produces ~30 distilled articles plus theme syntheses and a "
            "final report. Budgeted by time + cost + paper count."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The research question or topic.",
                },
                "duration": {
                    "type": "string",
                    "description": (
                        "Time budget like '30m', '2h', '1h30m' (default '2h')."
                    ),
                    "default": "2h",
                },
                "max_papers": {
                    "type": "integer",
                    "description": "Cap on papers to distill (default 20).",
                    "default": 20,
                },
                "max_cost_cny": {
                    "type": "number",
                    "description": "Cost ceiling in CNY (default 15.0).",
                    "default": 15.0,
                },
            },
            "required": ["question"],
        },
    },
}


TOOL_SCHEMAS: list = [
    _SEARCH_SCHEMA,
    _DISTILL_BY_ID_SCHEMA,
    _SHOW_SCHEMA,
    _ASK_SCHEMA,
    _RESEARCH_SCHEMA,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NoDepsProcessor(PaperProcessor):
    """Phase-B processor for tool_distill_by_id: skips the merger dep
    because search/merge already ran in Phase A and ``ranked`` is
    pre-populated with the user-selected papers."""
    deps: list[str] = []


def _paper_matches_id(paper, target_id: str) -> bool:
    """True if any of paper's IDs matches target_id (case-insensitive)."""
    if not target_id:
        return False
    needle = target_id.strip().lower()
    if not needle:
        return False
    for attr in ("arxiv_id", "doi", "paper_id", "ss_paper_id"):
        v = getattr(paper, attr, None)
        if v and str(v).strip().lower() == needle:
            return True
    return False


def _error(exc: Exception) -> dict:
    return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------

def tool_search(
    topic: str,
    n: int = 10,
    source: str = "all",
    *,
    vault_path: str,
) -> dict:
    """Search arxiv + SS + OpenAlex in parallel; return ranked candidates."""
    try:
        cfg = load_config(
            vault_path=vault_path,
            topic=topic,
            n=n,
            pool=max(n * 3, 30),
            source=source,
        )
        vault = VaultStore(cfg.vault_path)
        llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
        renderer = ConsoleRenderer(title=f"search · {topic}")
        ctx = Context(
            cfg=cfg, llm=llm, vault=vault,
            shared={}, on_status=renderer.on_status,
        )
        dag = DAG([
            ArxivSearcher(),
            SemanticScholarSearcher(),
            OpenCLIOpenAlexSearcher(),
            CandidateMerger(),
            CandidateRanker(),
        ])
        asyncio.run(Orchestrator(dag, ctx).run())
        ranked = ctx.shared.get("ranked", []) or []
        candidates = []
        for p in ranked[:n]:
            pid = getattr(p, "arxiv_id", None) or getattr(p, "doi", None) \
                or getattr(p, "paper_id", None) or ""
            candidates.append({
                "id": pid,
                "title": getattr(p, "title", "") or "",
                "authors": (getattr(p, "authors", None) or [])[:5],
                "year": (getattr(p, "published", "") or "")[:4],
                "abstract": (getattr(p, "abstract", "") or "")[:500],
                "pdf_url": getattr(p, "pdf_url", "") or "",
            })
        return {"candidates": candidates}
    except Exception as e:
        return _error(e)


def tool_distill_by_id(
    ids: list,
    topic: str | None = None,
    *,
    vault_path: str,
) -> dict:
    """Download + distill papers by ID, save to vault."""
    try:
        if not ids:
            return {"error": "ids must be a non-empty list"}
        # NOTE: re-runs the full search to resolve IDs into Paper metadata.
        # Wasteful when IDs come from a prior tool_search call in the same
        # conversation — caching across tool calls is a TODO for a later task.
        #
        # Fallback when caller omits topic: arxiv keyword search won't match
        # numeric IDs as keywords; this often produces matched_count: 0.
        # The schema description tells the LLM to always pass `topic`.
        search_topic = topic or " ".join(ids[:5])

        cfg = load_config(
            vault_path=vault_path,
            topic=search_topic,
            n=len(ids),
            pool=max(len(ids) * 5, 30),
            source="all",
        )
        vault = VaultStore(cfg.vault_path)
        llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
        renderer = ConsoleRenderer(title=f"distill_by_id · {len(ids)} papers")
        ctx = Context(
            cfg=cfg, llm=llm, vault=vault,
            shared={}, on_status=renderer.on_status,
        )

        # Phase A: search + merge + rank (cheap, populates ctx.shared["ranked"]
        # and the per-source candidate lists for matching).
        search_dag = DAG([
            ArxivSearcher(),
            SemanticScholarSearcher(),
            OpenCLIOpenAlexSearcher(),
            CandidateMerger(),
            CandidateRanker(),
        ])
        asyncio.run(Orchestrator(search_dag, ctx).run())

        # Match across the full merged pool — not just the LLM-ranked subset.
        pool = ctx.shared.get("candidates", []) or []
        matched = []
        unmatched = []
        for target in ids:
            hit = next((p for p in pool if _paper_matches_id(p, target)), None)
            if hit is not None:
                matched.append(hit)
            else:
                unmatched.append(target)

        # Replace the ranked list with the user-curated set, dropping
        # leftover state from the search phase.
        ctx.shared["ranked"] = matched
        ctx.shared.pop("articles", None)

        if not matched:
            return {
                "distilled": [],
                "survey_slug": None,
                "matched_count": 0,
                "requested_count": len(ids),
                "unmatched": unmatched,
            }

        # Phase B: distill-only DAG.
        processor = _NoDepsProcessor()
        distill_dag = DAG([processor, VaultWriter(), SurveyComposer()])
        renderer2 = ConsoleRenderer(title=f"distill · {len(matched)} papers")
        ctx.on_status = renderer2.on_status
        asyncio.run(Orchestrator(distill_dag, ctx).run())

        articles = ctx.shared.get("articles", []) or []
        out = {
            "distilled": [
                {"slug": a.slug, "title": a.title, "category": "articles"}
                for a in articles
            ],
            "survey_slug": ctx.shared.get("survey_slug"),
            "matched_count": len(matched),
            "requested_count": len(ids),
        }
        if unmatched:
            out["unmatched"] = unmatched
        return out
    except Exception as e:
        return _error(e)


def tool_show(
    slug: str,
    category: str = "articles",
    *,
    vault_path: str,
) -> dict:
    """Read a saved vault entry by slug."""
    try:
        vault = VaultStore(vault_path)
        entry = vault.read_entry(category, slug)
        if entry is None:
            return {"error": f"entry {category}/{slug} not found"}
        return {
            "slug": entry.slug,
            "title": entry.title,
            "category": entry.category,
            "tags": entry.tags,
            "refs": entry.refs,
            "links": entry.links,
            "created": entry.created,
            "updated": entry.updated,
            "body": entry.body,
        }
    except Exception as e:
        return _error(e)


def tool_ask(
    question: str,
    max_rounds: int = 3,
    per_round: int = 2,
    max_cost_cny: float = 5.0,
    max_articles: int = 10,
    *,
    vault_path: str,
) -> dict:
    """Run a multi-round QA loop and return the summary dict."""
    try:
        cfg = load_config_qa(
            vault_path=vault_path,
            question=question,
            max_rounds=max_rounds,
            max_articles=max_articles,
            max_cost_cny=max_cost_cny,
            confidence_threshold=8,
            per_round=per_round,
            source="all",
            interactive=False,
            resume_session_id=None,
            dry_run=False,
        )
        return run_qa_loop(cfg)
    except Exception as e:
        return _error(e)


def tool_research(
    question: str,
    duration: str = "2h",
    max_papers: int = 20,
    max_cost_cny: float = 15.0,
    *,
    vault_path: str,
) -> dict:
    """Run the autonomous deep-research loop and return the summary dict."""
    try:
        duration_sec = _parse_duration(duration)
        cfg = load_config_research(
            vault_path=vault_path,
            question=question,
            max_papers=max_papers,
            max_cost_cny=max_cost_cny,
            max_duration_sec=duration_sec,
            source="all",
            resume_session_id=None,
            dry_run=False,
        )
        return run_research_loop(cfg)
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Dispatch table + execute_tool
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Callable] = {
    "search": tool_search,
    "distill_by_id": tool_distill_by_id,
    "show": tool_show,
    "ask": tool_ask,
    "research": tool_research,
}


def execute_tool(name: str, arguments: dict, *, vault_path: str) -> dict:
    """Dispatch a tool call by name. Unknown name → {"error": ...}.

    Always passes vault_path into the wrapper. Wrappers themselves catch
    exceptions and return error dicts, so this function does not need its
    own try/except for tool-internal failures — only for dispatch issues
    like a bad arguments shape.
    """
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        kwargs = dict(arguments or {})
        kwargs["vault_path"] = vault_path
        return fn(**kwargs)
    except TypeError as e:
        # Typical cause: missing required arg or unexpected kwarg from LLM.
        return {"error": f"TypeError: {e}"}
