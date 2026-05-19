# paper-distiller v1.0 — Plan 2 (QA Loop Port + ask/resume)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Port the v0.5 QA loop (`paper-distiller-qa`) into the new v1.0 agent framework. After this plan: `paper-distiller-chat ask --question Y` runs a multi-round QA loop end-to-end with live status table; `paper-distiller-chat resume <sid>` picks up a paused/errored session. Old CLIs still alive (Plan 3 deletes them).

**Architecture:** The orchestrator stays "run one DAG once" — it has no native loop concept. A new module `chat/qa_runner.py` drives the multi-round loop, calling `Orchestrator.run()` once per phase (reflection → distillation → optional synthesis). State persists between rounds via existing `qa/state.py::SessionState` + `write_state`. Three new agents: `ProgressReflector`, `CandidateDedup`, `AnswerSynthesizer`.

**Tech Stack:** Same as Plan 1. No new deps.

**Spec:** [docs/superpowers/specs/2026-05-19-paper-distiller-v1.0-chat-design.md](../specs/2026-05-19-paper-distiller-v1.0-chat-design.md) §7.8 / §7.9 / §9.

**Working directory:** `G:\paper-distiller\`

**Test baseline:** 128 (after Plan 1 + housekeeping commit `036a5cc`).

---

## Plan decomposition update

Original spec §12 had 5 phases in 3 plans. Plan 1 turned out larger than expected, so I'm splitting the rest as:

| Plan | Phase | Ships (on main) | tag |
|---|---|---|---|
| **Plan 2 (this doc)** | C | `chat ask` + `chat resume` one-shots | none |
| Plan 3 | D | `chat` interactive REPL + intent-router | none |
| Plan 4 | E | Delete old CLIs + docs + release | **v1.0.0** |

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/paper_distiller/agents/reflector.py` | Create | `ProgressReflector` agent |
| `src/paper_distiller/agents/dedup.py` | Create | `CandidateDedup` agent (filter against `qa_state.articles_seen_ids`) |
| `src/paper_distiller/agents/synthesizer.py` | Create | `AnswerSynthesizer` agent |
| `src/paper_distiller/chat/qa_runner.py` | Create | The QA-loop driver — orchestrates rounds, persists state, decides stop_reason |
| `src/paper_distiller/chat/cli.py` | Modify | Add `ask` + `resume` subparsers; wire them to `qa_runner` |
| `tests/agents/test_reflector.py` | Create | 3 tests |
| `tests/agents/test_dedup.py` | Create | 3 tests |
| `tests/agents/test_synthesizer.py` | Create | 3 tests |
| `tests/chat/test_qa_runner.py` | Create | 6 tests — one per functional stop reason |
| `tests/chat/test_ask_cli.py` | Create | 2 tests — argparse + dispatch |
| `tests/chat/test_resume_cli.py` | Create | 2 tests — load state + continue |
| `tests/integration/test_ask_e2e.py` | Create | 1 e2e test — full QA loop with mocks |

**Test count after Plan 2:** 128 + 20 = **148**.

---

## Agent placement in QA-mode DAGs

Two DAG shapes used by the QA-loop driver:

**Reflection DAG (per round, beginning):**
```
[ProgressReflector]    # writes shared["reflection"]
```

**Distillation DAG (per round, after reflection allows continue):**
```
arxiv-searcher  ss-searcher
       └────┬────┘
       candidate-merger
              │
        candidate-dedup    (NEW — filters against qa_state.articles_seen_ids)
              │
       candidate-ranker
              │
       paper-processor (fanout × N)
              │
        vault-writer
```

Note: NO `survey-composer` in QA distillation rounds — survey composition is replaced by the per-question `answer-synthesizer` at the end.

**Synthesis DAG (once, after loop terminates with articles):**
```
[AnswerSynthesizer]    # composes final cited answer survey
```

Each DAG is built fresh per call; orchestrator is reused.

---

## Task 1: `ProgressReflector` agent

**Files:**
- Create: `src/paper_distiller/agents/reflector.py`
- Create: `tests/agents/test_reflector.py`

The agent wraps `qa.reflection.reflect`. It reads `ctx.shared["qa_state"]` (SessionState instance) and `ctx.cfg.qa_max_rounds`, calls the LLM, writes `ctx.shared["reflection"]: dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_reflector.py`:

```python
"""Tests for ProgressReflector agent — wraps qa.reflection.reflect."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.reflector import ProgressReflector
from paper_distiller.qa.state import SessionState


def _state():
    return SessionState(
        session_id="sid-1", question="why diffusion?",
        config_snapshot={}, started_at="2026-05-19T10:00:00",
    )


def _ctx(state, **cfg_overrides):
    cfg = SimpleNamespace(qa_max_rounds=3, qa_question="why diffusion?", **cfg_overrides)
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"qa_state": state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_reflector_calls_reflect_with_state_inputs(mocker):
    state = _state()
    state.history = []
    state.articles_distilled = []
    fake_reflect = mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": False, "confidence": 5,
            "what_we_know": "...", "what_is_missing": "...",
            "next_query": "diffusion convergence 2024",
            "next_query_rationale": "...", "suggest_stop": False,
        },
    )
    ctx = _ctx(state)
    out = await ProgressReflector().run(ctx)
    fake_reflect.assert_called_once()
    call = fake_reflect.call_args
    # Accept either positional or keyword; check key inputs are present
    args_and_kw = list(call.args) + list(call.kwargs.values())
    assert "why diffusion?" in args_and_kw  # question
    assert 3 in args_and_kw  # max_rounds
    assert out["reflection"]["next_query"] == "diffusion convergence 2024"


@pytest.mark.asyncio
async def test_reflector_passes_prior_queries_from_history(mocker):
    state = _state()
    from paper_distiller.qa.state import RoundRecord
    state.history = [
        RoundRecord(round=1, query="q1", rationale="", candidates_found=3,
                    new_articles=2, article_slugs=[], what_we_know="",
                    what_is_missing="", confidence=4, timestamp="..."),
    ]
    fake_reflect = mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        return_value={
            "is_done": False, "confidence": 6,
            "what_we_know": "", "what_is_missing": "",
            "next_query": "q2", "next_query_rationale": "", "suggest_stop": False,
        },
    )
    ctx = _ctx(state)
    await ProgressReflector().run(ctx)
    # prior_queries should include "q1"
    call = fake_reflect.call_args
    found = False
    for a in list(call.args) + list(call.kwargs.values()):
        if isinstance(a, list) and "q1" in a:
            found = True
    assert found, "prior_queries should include 'q1'"


@pytest.mark.asyncio
async def test_reflector_deps():
    assert ProgressReflector().deps == []
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_reflector.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `src/paper_distiller/agents/reflector.py`**

```python
"""ProgressReflector — wraps qa.reflection.reflect into an agent."""

from __future__ import annotations

import asyncio

from ..qa.reflection import reflect
from .base import Context


def _article_summary_line(article) -> str:
    title = (article.title or "").replace("\n", " ").strip()[:120]
    return f"[[{article.slug}]] {title}"


class ProgressReflector:
    name = "progress-reflector"
    deps: list[str] = []

    async def run(self, ctx: Context) -> dict:
        state = ctx.shared["qa_state"]
        articles_summary = [
            _article_summary_line(a) for a in state.articles_distilled
        ]
        prior_queries = [r.query for r in state.history if r.query]
        reflection_dict = await asyncio.to_thread(
            reflect,
            question=state.question,
            articles_summary=articles_summary,
            prior_queries=prior_queries,
            round_num=state.rounds_completed + 1,
            max_rounds=ctx.cfg.qa_max_rounds,
            llm=ctx.llm,
        )
        return {"reflection": reflection_dict}
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_reflector.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **131 passed** (128 + 3).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/agents/reflector.py tests/agents/test_reflector.py
git commit -m "feat(agents): ProgressReflector — wraps qa.reflection into an agent"
```

---

## Task 2: `CandidateDedup` agent

**Files:**
- Create: `src/paper_distiller/agents/dedup.py`
- Create: `tests/agents/test_dedup.py`

Filters `shared["candidates"]` against `qa_state.articles_seen_ids` (set of paper IDs already distilled in prior rounds). Runs between `candidate-merger` and `candidate-ranker` in the QA distillation DAG. Skipped in single-pass mode (no qa_state).

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_dedup.py`:

```python
"""Tests for CandidateDedup agent — filters candidates against qa_state.articles_seen_ids."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.dedup import CandidateDedup
from paper_distiller.qa.state import SessionState
from paper_distiller.sources.arxiv import Paper


def _paper(pid, doi=None):
    return Paper(
        source="arxiv", paper_id=pid, arxiv_id=pid, doi=doi,
        title=f"P{pid}", authors=[], abstract="...",
        pdf_url="...", published="2025-01-01", categories=[],
    )


def _ctx(candidates, seen_ids=None):
    state = SessionState(
        session_id="sid-1", question="?", config_snapshot={},
        started_at="2026-05-19T10:00:00",
        articles_seen_ids=set(seen_ids or []),
    )
    return Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"candidates": candidates, "qa_state": state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_dedup_passes_through_when_seen_empty():
    cands = [_paper("X1"), _paper("X2")]
    ctx = _ctx(cands, seen_ids=set())
    out = await CandidateDedup().run(ctx)
    assert [p.arxiv_id for p in out["candidates"]] == ["X1", "X2"]


@pytest.mark.asyncio
async def test_dedup_filters_seen_arxiv_ids():
    cands = [_paper("X1"), _paper("X2"), _paper("X3")]
    ctx = _ctx(cands, seen_ids={"X2"})
    out = await CandidateDedup().run(ctx)
    assert [p.arxiv_id for p in out["candidates"]] == ["X1", "X3"]


@pytest.mark.asyncio
async def test_dedup_filters_seen_dois():
    cands = [_paper("X1", doi="10.1/abc"), _paper("X2"), _paper("X3", doi="10.2/def")]
    ctx = _ctx(cands, seen_ids={"10.1/abc"})
    out = await CandidateDedup().run(ctx)
    assert {p.arxiv_id for p in out["candidates"]} == {"X2", "X3"}


@pytest.mark.asyncio
async def test_dedup_noop_when_no_qa_state():
    """In single-pass mode (no qa_state in shared), dedup is a no-op pass-through."""
    cands = [_paper("X1")]
    ctx = Context(
        cfg=SimpleNamespace(), llm=MagicMock(), vault=MagicMock(),
        shared={"candidates": cands},  # no qa_state
        on_status=lambda *a, **kw: None,
    )
    out = await CandidateDedup().run(ctx)
    assert out["candidates"] == cands


@pytest.mark.asyncio
async def test_dedup_deps():
    assert CandidateDedup().deps == ["candidate-merger"]
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_dedup.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `src/paper_distiller/agents/dedup.py`**

```python
"""CandidateDedup — filters shared['candidates'] against qa_state.articles_seen_ids.

QA-mode only. In single-pass (no qa_state), passes candidates through unchanged.
"""

from __future__ import annotations

from .base import Context


class CandidateDedup:
    name = "candidate-dedup"
    deps = ["candidate-merger"]

    async def run(self, ctx: Context) -> dict:
        candidates = ctx.shared.get("candidates", [])
        state = ctx.shared.get("qa_state")
        if state is None:
            return {"candidates": candidates}
        seen = state.articles_seen_ids
        if not seen:
            return {"candidates": candidates}
        filtered = []
        for p in candidates:
            pid = p.arxiv_id or p.doi
            if pid and pid in seen:
                continue
            filtered.append(p)
        return {"candidates": filtered}
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_dedup.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **136 passed** (131 + 5).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/agents/dedup.py tests/agents/test_dedup.py
git commit -m "feat(agents): CandidateDedup — filter candidates against qa_state.articles_seen_ids"
```

---

## Task 3: `AnswerSynthesizer` agent

**Files:**
- Create: `src/paper_distiller/agents/synthesizer.py`
- Create: `tests/agents/test_synthesizer.py`

Wraps `qa.answer.synthesize`. Runs once after the QA loop terminates with `state.articles_distilled` non-empty. Writes the answer survey to `<vault>/surveys/qa-<slug>-<date>.md` and returns `{"answer_survey_slug": ...}`.

The agent also assembles the audit-trail markdown (per-round table) and appends it to the body — same shape as v0.5 `qa/loop.py::_build_survey_body`.

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_synthesizer.py`:

```python
"""Tests for AnswerSynthesizer agent — wraps qa.answer.synthesize + vault write."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paper_distiller.agents.base import Context
from paper_distiller.agents.synthesizer import AnswerSynthesizer
from paper_distiller.distill.article import ArticleResult
from paper_distiller.qa.state import SessionState


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _state(articles):
    s = SessionState(
        session_id="sid-1", question="why diffusion?",
        config_snapshot={}, started_at="2026-05-19T10:00:00",
        articles_distilled=articles,
        stop_reason="llm_done",
        rounds_completed=2,
        cost_cny=1.5,
        tokens_in_total=1000, tokens_out_total=500,
    )
    return s


def _ctx(state, **cfg_overrides):
    cfg = SimpleNamespace(qa_question=state.question, **cfg_overrides)
    return Context(
        cfg=cfg, llm=MagicMock(), vault=MagicMock(),
        shared={"qa_state": state},
        on_status=lambda *a, **kw: None,
    )


@pytest.mark.asyncio
async def test_synthesizer_calls_synthesize_and_saves(mocker):
    arts = [_article("a"), _article("b")]
    state = _state(arts)
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA: 答案", "body": "# 答案\n\n...",
            "tags": ["qa"], "cited_slugs": ["a", "b"],
        },
    )
    ctx = _ctx(state)
    ctx.vault.save_entry = MagicMock(side_effect=lambda **kw: {"slug": kw["slug"]})
    out = await AnswerSynthesizer().run(ctx)
    ctx.vault.save_entry.assert_called_once()
    # category="surveys", slug starts with "qa-"
    call_kw = ctx.vault.save_entry.call_args.kwargs
    assert call_kw["category"] == "surveys"
    assert call_kw["slug"].startswith("qa-")
    assert "answer_survey_slug" in out


@pytest.mark.asyncio
async def test_synthesizer_skipped_when_no_articles():
    state = _state(articles=[])
    ctx = _ctx(state)
    ctx.vault.save_entry = MagicMock()
    out = await AnswerSynthesizer().run(ctx)
    ctx.vault.save_entry.assert_not_called()
    assert out["answer_survey_slug"] is None


@pytest.mark.asyncio
async def test_synthesizer_appends_audit_trail_to_body(mocker):
    """Final body contains the LLM answer + audit trail table."""
    arts = [_article("a")]
    state = _state(arts)
    state.history = []  # empty audit trail still produces a header
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={"title": "QA: x", "body": "# answer", "tags": [], "cited_slugs": ["a"]},
    )
    ctx = _ctx(state)
    saved_body = {}
    def _capture_save(**kw):
        saved_body["body"] = kw["body"]
        return {"slug": kw["slug"]}
    ctx.vault.save_entry = MagicMock(side_effect=_capture_save)
    await AnswerSynthesizer().run(ctx)
    # The saved body must include both the LLM answer and an audit trail section
    assert "# answer" in saved_body["body"]
    assert "audit trail" in saved_body["body"].lower() or "研究过程" in saved_body["body"]


@pytest.mark.asyncio
async def test_synthesizer_deps():
    assert AnswerSynthesizer().deps == ["vault-writer"]
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_synthesizer.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `src/paper_distiller/agents/synthesizer.py`**

```python
"""AnswerSynthesizer — wraps qa.answer.synthesize, appends audit trail,
writes the final answer survey to <vault>/surveys/qa-<slug>-<date>.md.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime

from ..qa.answer import synthesize
from ..vault.store import slugify
from .base import Context


def _audit_trail_markdown(state) -> str:
    rows = ["| 轮 | Query | 新增 | LLM 判断 | Confidence |",
            "|---|---|---|---|---|"]
    for r in state.history:
        what_missing = (r.what_is_missing or r.what_we_know or "").replace("\n", " ")[:50]
        rows.append(
            f"| {r.round} | {r.query[:40]} | {r.new_articles} | "
            f"{what_missing} | {r.confidence} |"
        )
    table = "\n".join(rows)
    footer = (
        f"\n\n**Stop reason**: {state.stop_reason}\n"
        f"**Rounds**: {state.rounds_completed}\n"
        f"**Articles distilled**: {len(state.articles_distilled)}\n"
        f"**Total cost**: ¥{state.cost_cny:.2f} ({state.tokens_in_total} in / "
        f"{state.tokens_out_total} out tokens)\n"
        f"**Session ID**: {state.session_id}\n"
    )
    return table + footer


def _build_full_body(answer: dict, state) -> str:
    parts = [answer["body"]]
    cited = answer.get("cited_slugs") or []
    if cited:
        cited_rows = ["", "## 引用的 articles", "", "| Slug | 标题 |", "|---|---|"]
        slug_to_article = {a.slug: a for a in state.articles_distilled}
        for slug in cited:
            article = slug_to_article.get(slug)
            if article is not None:
                title = (article.title or "").replace("\n", " ")[:80]
                cited_rows.append(f"| [[{slug}]] | {title} |")
        parts.append("\n".join(cited_rows))
    parts.append("\n## 研究过程 (audit trail)\n")
    parts.append(_audit_trail_markdown(state))
    return "\n".join(parts)


class AnswerSynthesizer:
    name = "answer-synthesizer"
    deps = ["vault-writer"]

    async def run(self, ctx: Context) -> dict:
        state = ctx.shared["qa_state"]
        if not state.articles_distilled:
            return {"answer_survey_slug": None}
        answer = await asyncio.to_thread(
            synthesize, state.question, state.articles_distilled, ctx.llm,
        )
        body = _build_full_body(answer, state)
        slug_base = slugify(state.question)[:30] or "untitled"
        slug = f"qa-{slug_base}-{datetime.now().strftime('%Y%m%d')}"
        try:
            saved = await asyncio.to_thread(
                ctx.vault.save_entry,
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        except ValueError:
            slug = f"{slug}-{secrets.token_hex(2)}"
            saved = await asyncio.to_thread(
                ctx.vault.save_entry,
                category="surveys",
                title=answer["title"],
                body=body,
                tags=answer.get("tags") or ["qa"],
                refs=[f"qa-session:{state.session_id}"],
                slug=slug,
            )
        return {"answer_survey_slug": saved["slug"]}
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/agents/test_synthesizer.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **140 passed** (136 + 4).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/agents/synthesizer.py tests/agents/test_synthesizer.py
git commit -m "feat(agents): AnswerSynthesizer — final cited answer survey with audit trail"
```

---

## Task 4: QA-loop driver (`chat/qa_runner.py`)

**Files:**
- Create: `src/paper_distiller/chat/qa_runner.py`
- Create: `tests/chat/test_qa_runner.py`

The QA loop is NOT one DAG run. It's:
1. Build initial state (or read from disk if resume)
2. Loop: reflection DAG → check stop → distillation DAG → update state → persist
3. After loop terminates: synthesis DAG (if articles)
4. Final state persist, return state

Stops: 7 reasons per spec §4. Plan-2 implementation covers the 6 functional ones; `user_quit` (Ctrl+C / interactive) is wired but only fully exercised in Plan 3 (interactive REPL).

- [ ] **Step 1: Write the failing tests**

Create `tests/chat/test_qa_runner.py`:

```python
"""Tests for chat.qa_runner — the QA loop driver. All subsystems mocked."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paper_distiller.distill.article import ArticleResult
from paper_distiller.sources.arxiv import Paper


def _paper(i, arxiv_id=None):
    aid = arxiv_id or f"2501.0000{i}"
    return Paper(
        source="arxiv", paper_id=aid, arxiv_id=aid,
        title=f"P{i}", authors=[], abstract=f"abstract {i}",
        pdf_url=f"https://x/{aid}.pdf", published="2025-01-01", categories=[],
    )


def _article(slug):
    return ArticleResult(
        slug=slug, title=f"T-{slug}", body="b",
        tags=[], refs=[f"arxiv:{slug}"], depth="full-pdf",
    )


def _cfg(tmp_path, max_rounds=5, max_articles=15, max_cost=20.0, threshold=8, per_round=2):
    """Build a Config for QA mode tests."""
    from paper_distiller.config import Config
    return Config(
        vault_path=tmp_path / "vault",
        topic=None, author=None,
        top_n=per_round, pool=10, force=False, dry_run=False, verbose=False,
        api_key="sk-test", base_url="https://x/v1", model="qwen-plus",
        provider_name="test", pdf_timeout_sec=60, min_papers_for_survey=2,
        source="arxiv", ss_api_key=None,
        qa_max_rounds=max_rounds, qa_max_articles=max_articles,
        qa_max_cost_cny=max_cost, qa_confidence_threshold=threshold,
        qa_per_round=per_round, qa_interactive=False,
        qa_resume_session_id=None, qa_question="why diffusion?",
    )


def _common_mocks(mocker, reflection_responses):
    """Mock all the subsystems used by the QA loop's DAGs."""
    mocker.patch("paper_distiller.chat.qa_runner.LLMClient")
    mocker.patch("paper_distiller.chat.qa_runner.VaultStore")
    mocker.patch(
        "paper_distiller.agents.reflector.reflect",
        side_effect=list(reflection_responses),
    )
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        return_value=[_paper(1), _paper(2), _paper(3)],
    )
    mocker.patch(
        "paper_distiller.agents.searchers.ss_search",
        return_value=[],
    )
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda paper, full_text, wiki_index, llm: _article(f"a-{paper.arxiv_id}"),
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA: answer", "body": "# answer\n\n...",
            "tags": ["qa"], "cited_slugs": ["a-2501.00001"],
        },
    )


def test_qa_loop_stops_on_llm_done(tmp_path, mocker):
    """Reflection returns is_done=True with confidence >= threshold → stop."""
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "", "what_is_missing": "",
         "next_query": "q1", "next_query_rationale": "", "suggest_stop": False},
        {"is_done": True, "confidence": 9, "what_we_know": "all clear", "what_is_missing": "",
         "next_query": "", "next_query_rationale": "", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "llm_done"
    assert summary["rounds_completed"] == 1


def test_qa_loop_stops_on_max_rounds(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_rounds=2)
    cfg.vault_path.mkdir()
    not_done = {"is_done": False, "confidence": 4, "what_we_know": "a",
                "what_is_missing": "...", "next_query_rationale": "...", "suggest_stop": False}
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
        {**not_done, "next_query": "q3"},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "max_rounds"
    assert summary["rounds_completed"] == 2


def test_qa_loop_stops_on_no_candidates(tmp_path, mocker):
    """If all candidates were already seen, stop with no_candidates."""
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    not_done = {"is_done": False, "confidence": 4, "what_we_know": "", "what_is_missing": "",
                "next_query_rationale": "", "suggest_stop": False}
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
    ]
    _common_mocks(mocker, reflection_seq)
    # Override arxiv_search: always return the same 2 papers, so round 2 finds nothing new
    same_papers = [_paper(1), _paper(2)]
    mocker.patch("paper_distiller.agents.searchers.arxiv_search", return_value=same_papers)

    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "no_candidates"


def test_qa_loop_stops_on_max_articles(tmp_path, mocker):
    cfg = _cfg(tmp_path, max_articles=2, per_round=2)
    cfg.vault_path.mkdir()
    reflection_seq = [
        {"is_done": False, "confidence": 4, "what_we_know": "a",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "max_articles"
    assert summary["articles_distilled_count"] == 2


def test_qa_loop_stops_on_llm_brake(tmp_path, mocker):
    """reflection.suggest_stop=True → llm_brake stop reason."""
    cfg = _cfg(tmp_path)
    cfg.vault_path.mkdir()
    reflection_seq = [
        {"is_done": False, "confidence": 3, "what_we_know": "", "what_is_missing": "",
         "next_query": "q1", "next_query_rationale": "", "suggest_stop": True},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    assert summary["stop_reason"] == "llm_brake"


def test_qa_loop_persists_state_each_round(tmp_path, mocker):
    """state.json appears under .paper_distiller/qa-sessions/<sid>/ after each round."""
    cfg = _cfg(tmp_path, max_rounds=1)
    cfg.vault_path.mkdir()
    not_done = {"is_done": False, "confidence": 4, "what_we_know": "a",
                "what_is_missing": "...", "next_query_rationale": "...", "suggest_stop": False}
    reflection_seq = [
        {**not_done, "next_query": "q1"},
        {**not_done, "next_query": "q2"},
    ]
    _common_mocks(mocker, reflection_seq)
    from paper_distiller.chat.qa_runner import run_qa_loop
    summary = run_qa_loop(cfg)
    sid = summary["session_id"]
    state_path = cfg.vault_path / ".paper_distiller" / "qa-sessions" / sid / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["stop_reason"] == "max_rounds"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_qa_runner.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Create `src/paper_distiller/chat/qa_runner.py`**

```python
"""QA-loop driver. Orchestrates multiple Orchestrator.run() invocations per
QA session: reflection → distillation → repeat → synthesis.

Stop reasons (7 total): max_rounds, llm_done, llm_brake, no_candidates,
max_articles, max_cost, user_quit, plus 'error: ...' for transient failures.

State persisted to <vault>/.paper_distiller/qa-sessions/<sid>/state.json
after every round.
"""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime

from rich.console import Console
from rich.live import Live

from ..agents.base import Context
from ..agents.curation import CandidateMerger, CandidateRanker
from ..agents.dag import DAG
from ..agents.dedup import CandidateDedup
from ..agents.orchestrator import Orchestrator, AgentFailed
from ..agents.processor import PaperProcessor
from ..agents.reflector import ProgressReflector
from ..agents.renderer import ConsoleRenderer
from ..agents.searchers import ArxivSearcher, SemanticScholarSearcher
from ..agents.synthesizer import AnswerSynthesizer
from ..agents.writer import VaultWriter
from ..config import Config
from ..llm.openai_compatible import LLMClient
from ..qa.state import RoundRecord, SessionState, read_state, write_state
from ..vault.store import VaultStore


_PRICE_IN_CNY_PER_M = 2.1
_PRICE_OUT_CNY_PER_M = 12.7


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M") + "-" + secrets.token_hex(3)[:5]


def _build_reflection_dag() -> DAG:
    return DAG([ProgressReflector()])


def _build_distillation_dag() -> DAG:
    return DAG([
        ArxivSearcher(),
        SemanticScholarSearcher(),
        CandidateMerger(),
        CandidateDedup(),
        CandidateRanker(),
        PaperProcessor(),
        VaultWriter(),
    ])


def _build_synthesis_dag() -> DAG:
    return DAG([AnswerSynthesizer()])


def _update_cost(state: SessionState, llm: LLMClient) -> None:
    state.tokens_in_total = llm.total_tokens_in
    state.tokens_out_total = llm.total_tokens_out
    state.cost_cny = (
        llm.total_tokens_in * _PRICE_IN_CNY_PER_M / 1_000_000
        + llm.total_tokens_out * _PRICE_OUT_CNY_PER_M / 1_000_000
    )


async def _arun_qa_loop(cfg: Config) -> SessionState:
    if cfg.qa_resume_session_id:
        existing = read_state(cfg.vault_path, cfg.qa_resume_session_id)
        if existing is None:
            raise ValueError(f"resume session not found: {cfg.qa_resume_session_id}")
        if existing.is_done:
            raise ValueError(
                f"session {cfg.qa_resume_session_id} already done "
                f"(stop_reason={existing.stop_reason!r}); cannot resume"
            )
        state = existing
    else:
        state = SessionState(
            session_id=_new_session_id(),
            question=cfg.qa_question,
            config_snapshot={
                "max_rounds": cfg.qa_max_rounds,
                "max_articles": cfg.qa_max_articles,
                "max_cost_cny": cfg.qa_max_cost_cny,
                "confidence_threshold": cfg.qa_confidence_threshold,
                "per_round": cfg.qa_per_round,
                "source": cfg.source,
            },
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

    vault = VaultStore(cfg.vault_path)
    llm = LLMClient(cfg.api_key, cfg.base_url, cfg.model)
    renderer = ConsoleRenderer(title=f"QA: {state.question[:50]}")

    def _build_ctx() -> Context:
        return Context(
            cfg=cfg, llm=llm, vault=vault,
            shared={"qa_state": state},
            on_status=renderer.on_status,
        )

    console = Console()
    with Live(renderer.build_table(), refresh_per_second=10, console=console) as live:
        async def _refresher():
            while True:
                live.update(renderer.build_table())
                await asyncio.sleep(0.1)
        refresher_task = asyncio.create_task(_refresher())
        try:
            while True:
                # 1. Reflection
                ctx = _build_ctx()
                try:
                    await Orchestrator(_build_reflection_dag(), ctx).run()
                except AgentFailed as e:
                    state.stop_reason = f"error: reflection failed: {e.__cause__}"
                    break
                reflection = ctx.shared["reflection"]
                state.last_reflection = reflection
                _update_cost(state, llm)

                # 2. Termination checks
                if state.rounds_completed >= cfg.qa_max_rounds:
                    state.stop_reason = "max_rounds"
                    break
                if reflection.get("is_done") and \
                        int(reflection.get("confidence", 0)) >= cfg.qa_confidence_threshold:
                    state.stop_reason = "llm_done"
                    break
                if reflection.get("suggest_stop"):
                    state.stop_reason = "llm_brake"
                    break
                next_query = reflection.get("next_query") or ""
                if not next_query:
                    state.stop_reason = "no_candidates"
                    break

                # 3. Distillation round
                ctx = _build_ctx()
                ctx.shared["next_query"] = next_query
                try:
                    await Orchestrator(_build_distillation_dag(), ctx).run()
                except AgentFailed as e:
                    state.stop_reason = f"error: distillation failed: {e.__cause__}"
                    break

                # 4. Process this round's results
                articles_this_round = ctx.shared.get("articles", [])
                # Filter out articles we already had before this round
                prior_slugs = {a.slug for a in state.articles_distilled}
                new_articles = [a for a in articles_this_round if a.slug not in prior_slugs]
                if not new_articles:
                    # Check: was it because candidates were all seen?
                    candidates_after_dedup = ctx.shared.get("candidates", [])
                    if not candidates_after_dedup:
                        state.stop_reason = "no_candidates"
                        break
                state.articles_distilled.extend(new_articles)
                for a in new_articles:
                    # Best-effort ID: prefer arxiv-id, then DOI, then slug
                    for ref in a.refs:
                        if ref.startswith("arxiv:"):
                            state.articles_seen_ids.add(ref[6:])
                            break
                        if ref.startswith("doi:"):
                            state.articles_seen_ids.add(ref[4:])
                            break

                # 5. Record round in history
                state.history.append(RoundRecord(
                    round=state.rounds_completed + 1,
                    query=next_query,
                    rationale=reflection.get("next_query_rationale", ""),
                    candidates_found=len(ctx.shared.get("candidates_arxiv", []))
                                    + len(ctx.shared.get("candidates_ss", [])),
                    new_articles=len(new_articles),
                    article_slugs=[a.slug for a in new_articles],
                    what_we_know=reflection.get("what_we_know", ""),
                    what_is_missing=reflection.get("what_is_missing", ""),
                    confidence=int(reflection.get("confidence", 0)),
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                ))
                state.rounds_completed += 1
                _update_cost(state, llm)
                write_state(cfg.vault_path, state)

                # 6. Budget checks (post-round)
                if len(state.articles_distilled) >= cfg.qa_max_articles:
                    state.stop_reason = "max_articles"
                    break
                if state.cost_cny >= cfg.qa_max_cost_cny:
                    state.stop_reason = "max_cost"
                    break
        except KeyboardInterrupt:
            state.stop_reason = "user_quit"
            write_state(cfg.vault_path, state)
        finally:
            refresher_task.cancel()
            try:
                await refresher_task
            except asyncio.CancelledError:
                pass
            live.update(renderer.build_table())

    # 7. Synthesis (if articles exist)
    if state.articles_distilled:
        ctx = _build_ctx()
        try:
            await Orchestrator(_build_synthesis_dag(), ctx).run()
        except AgentFailed as e:
            print(f"Synthesis failed: {e.__cause__}")
        else:
            pass  # answer_survey_slug now in ctx.shared

    # 8. Final state save + is_done semantics
    non_terminal = (
        state.stop_reason == "user_quit"
        or state.stop_reason.startswith("error:")
    )
    state.is_done = not non_terminal
    _update_cost(state, llm)
    write_state(cfg.vault_path, state)
    return state


def run_qa_loop(cfg: Config) -> dict:
    """Sync entry point that returns a summary dict."""
    state = asyncio.run(_arun_qa_loop(cfg))
    return {
        "session_id": state.session_id,
        "stop_reason": state.stop_reason,
        "rounds_completed": state.rounds_completed,
        "articles_distilled_count": len(state.articles_distilled),
        "cost_cny": round(state.cost_cny, 2),
        "tokens_in_total": state.tokens_in_total,
        "tokens_out_total": state.tokens_out_total,
    }
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_qa_runner.py -v
```

Expected: 6 passed. If any fail, the most likely culprit is the `_common_mocks` not patching the right import site — adjust mock paths to match where each function is actually imported in the agent modules.

- [ ] **Step 5: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **146 passed** (140 + 6).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/chat/qa_runner.py tests/chat/test_qa_runner.py
git commit -m "feat(chat): QA-loop driver (qa_runner.py)

Orchestrates multi-round QA: reflection -> termination check ->
distillation -> state persist -> repeat -> synthesis. Composes
Plan-1 agents + 3 new Plan-2 agents (reflector/dedup/synthesizer).

6 stop reasons covered: max_rounds / llm_done / llm_brake /
no_candidates / max_articles / max_cost. user_quit + error: paths
also wired (full exercise in Plan 3 REPL).

State persisted per-round to .paper_distiller/qa-sessions/<sid>/state.json."
```

---

## Task 5: `ask` subcommand in chat/cli.py

**Files:**
- Modify: `src/paper_distiller/chat/cli.py` (add `ask` subparser + dispatch)
- Create: `tests/chat/test_ask_cli.py`

The `ask` subcommand takes a `--question` and the same QA budget flags as v0.5's `paper-distiller-qa`. Dispatches to `run_qa_loop(cfg)` from Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/chat/test_ask_cli.py`:

```python
"""Tests for paper-distiller-chat 'ask' subcommand."""
from unittest.mock import MagicMock

import pytest


def test_ask_cli_parses_args(monkeypatch):
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args([
        "ask", "--vault", "/tmp/v", "--question", "why?",
        "--max-rounds", "3", "--per-round", "2",
    ])
    assert args.subcommand == "ask"
    assert args.vault == "/tmp/v"
    assert args.question == "why?"
    assert args.max_rounds == 3
    assert args.per_round == 2


def test_ask_cli_dispatches_to_run_qa_loop(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    fake_run = mocker.patch("paper_distiller.chat.cli.run_qa_loop")
    fake_run.return_value = {
        "session_id": "sid-1", "stop_reason": "llm_done",
        "rounds_completed": 2, "articles_distilled_count": 4,
        "cost_cny": 0.5, "tokens_in_total": 1000, "tokens_out_total": 500,
    }
    from paper_distiller.chat.cli import main
    rc = main([
        "ask", "--vault", str(tmp_path), "--question", "why?",
        "--max-rounds", "3",
    ])
    assert rc == 0
    fake_run.assert_called_once()
    cfg = fake_run.call_args[0][0]
    assert cfg.qa_question == "why?"
    assert cfg.qa_max_rounds == 3
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_ask_cli.py -v
```

Expected: AttributeError or test failure (because `ask` subparser doesn't exist yet).

- [ ] **Step 3: Modify `src/paper_distiller/chat/cli.py`**

Add to top imports:

```python
from ..config import load_config_qa
from .qa_runner import run_qa_loop
```

In `build_parser()`, after the `distill` subparser block, add the `ask` subparser:

```python
    ask = sub.add_parser("ask", help="QA loop: ask a research question, multiple rounds")
    ask.add_argument("--vault", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--max-rounds", type=int, default=5)
    ask.add_argument("--max-articles", type=int, default=15)
    ask.add_argument("--max-cost-cny", type=float, default=20.0)
    ask.add_argument("--confidence-threshold", type=int, default=8)
    ask.add_argument("--per-round", type=int, default=2)
    ask.add_argument("--source", choices=["arxiv", "ss", "both"], default="both")
    ask.add_argument("--interactive", action="store_true")
    ask.add_argument("--dry-run", action="store_true")
    ask.add_argument("--verbose", "-v", action="store_true")
    ask.add_argument("--model")
    ask.add_argument("--provider")
```

After `_run_distill`, add `_run_ask`:

```python
def _run_ask(args) -> int:
    try:
        cfg = load_config_qa(
            vault_path=args.vault,
            question=args.question,
            max_rounds=args.max_rounds,
            max_articles=args.max_articles,
            max_cost_cny=args.max_cost_cny,
            confidence_threshold=args.confidence_threshold,
            per_round=args.per_round,
            source=args.source,
            interactive=args.interactive,
            resume_session_id=None,
            verbose=args.verbose,
            dry_run=args.dry_run,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if cfg.dry_run:
        print(f"[DRY-RUN] Would run QA loop for {cfg.qa_question!r}")
        return 0
    try:
        summary = run_qa_loop(cfg)
    except Exception as e:
        print(f"\nError during QA loop: {type(e).__name__}: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 3
    print()
    print(f"  Session:      {summary['session_id']}")
    print(f"  Stop reason:  {summary['stop_reason']}")
    print(f"  Rounds:       {summary['rounds_completed']}")
    print(f"  Articles:     {summary['articles_distilled_count']}")
    print(f"  Cost:         CNY {summary['cost_cny']:.2f}")
    print(f"  Tokens:       {summary['tokens_in_total']} / {summary['tokens_out_total']}")
    return 0
```

Modify `main` to dispatch `ask`:

Current:
```python
def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "distill":
        return asyncio.run(_run_distill(args))
    return 2
```

Replace with:
```python
def main(argv: list | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "distill":
        return asyncio.run(_run_distill(args))
    if args.subcommand == "ask":
        return _run_ask(args)  # run_qa_loop already wraps asyncio.run internally
    return 2
```

(Note: `run_qa_loop` is sync entry; `_run_ask` is sync. Don't wrap in asyncio.run again.)

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_ask_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full suite + smoke**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
.venv\Scripts\paper-distiller-chat.exe ask --help
```

Expected: 148 passed (146 + 2). Help shows the new ask subcommand.

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/chat/cli.py tests/chat/test_ask_cli.py
git commit -m "feat(chat): paper-distiller-chat ask subcommand (one-shot QA loop)"
```

---

## Task 6: `resume` subcommand in chat/cli.py

**Files:**
- Modify: `src/paper_distiller/chat/cli.py`
- Create: `tests/chat/test_resume_cli.py`

The `resume` subcommand reads a paused/errored session's state.json, sets `cfg.qa_resume_session_id`, and dispatches to `run_qa_loop` which picks up where it left off.

- [ ] **Step 1: Write the failing tests**

Create `tests/chat/test_resume_cli.py`:

```python
"""Tests for paper-distiller-chat 'resume' subcommand."""
import json
from unittest.mock import MagicMock

import pytest


def test_resume_cli_parses_args():
    from paper_distiller.chat.cli import build_parser
    p = build_parser()
    args = p.parse_args(["resume", "--vault", "/tmp/v", "--session-id", "sid-abc"])
    assert args.subcommand == "resume"
    assert args.vault == "/tmp/v"
    assert args.session_id == "sid-abc"


def test_resume_cli_dispatches_with_session_id(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")
    # Need to pre-seed a real state.json so load_config_qa doesn't get a non-existent session
    vault = tmp_path
    sessions_dir = vault / ".paper_distiller" / "qa-sessions" / "sid-abc"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "state.json").write_text(json.dumps({
        "session_id": "sid-abc", "question": "Q?", "config_snapshot": {},
        "started_at": "2026-05-19T10:00:00", "rounds_completed": 1,
        "articles_distilled": [], "articles_seen_ids": [], "history": [],
        "last_reflection": None, "cost_cny": 0.0,
        "tokens_in_total": 0, "tokens_out_total": 0,
        "is_done": False, "stop_reason": "user_quit",
    }), encoding="utf-8")

    fake_run = mocker.patch("paper_distiller.chat.cli.run_qa_loop")
    fake_run.return_value = {
        "session_id": "sid-abc", "stop_reason": "llm_done",
        "rounds_completed": 3, "articles_distilled_count": 5,
        "cost_cny": 1.0, "tokens_in_total": 2000, "tokens_out_total": 800,
    }
    from paper_distiller.chat.cli import main
    rc = main(["resume", "--vault", str(vault), "--session-id", "sid-abc"])
    assert rc == 0
    fake_run.assert_called_once()
    cfg = fake_run.call_args[0][0]
    assert cfg.qa_resume_session_id == "sid-abc"
```

- [ ] **Step 2: Run, confirm fail**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_resume_cli.py -v
```

Expected: failure (no `resume` subparser).

- [ ] **Step 3: Modify `src/paper_distiller/chat/cli.py`**

In `build_parser()`, after the `ask` subparser, add:

```python
    resume = sub.add_parser("resume", help="Resume a paused/errored QA session")
    resume.add_argument("--vault", required=True)
    resume.add_argument("--session-id", required=True)
    resume.add_argument("--verbose", "-v", action="store_true")
    resume.add_argument("--model")
    resume.add_argument("--provider")
```

Add `_run_resume`:

```python
def _run_resume(args) -> int:
    from ..qa.state import read_state
    from pathlib import Path
    existing = read_state(Path(args.vault), args.session_id)
    if existing is None:
        print(f"Error: session {args.session_id!r} not found in {args.vault}", file=sys.stderr)
        return 2
    try:
        cfg = load_config_qa(
            vault_path=args.vault,
            question=existing.question,
            max_rounds=existing.config_snapshot.get("max_rounds", 5),
            max_articles=existing.config_snapshot.get("max_articles", 15),
            max_cost_cny=existing.config_snapshot.get("max_cost_cny", 20.0),
            confidence_threshold=existing.config_snapshot.get("confidence_threshold", 8),
            per_round=existing.config_snapshot.get("per_round", 2),
            source=existing.config_snapshot.get("source", "both"),
            interactive=False,
            resume_session_id=args.session_id,
            verbose=args.verbose,
            dry_run=False,
            model_override=args.model,
            provider_override=args.provider,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    try:
        summary = run_qa_loop(cfg)
    except Exception as e:
        print(f"\nError during resume: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    print()
    print(f"  Session:      {summary['session_id']} (resumed)")
    print(f"  Stop reason:  {summary['stop_reason']}")
    print(f"  Rounds:       {summary['rounds_completed']}")
    print(f"  Articles:     {summary['articles_distilled_count']}")
    print(f"  Cost:         CNY {summary['cost_cny']:.2f}")
    return 0
```

In `main`:

```python
    if args.subcommand == "resume":
        return _run_resume(args)
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/chat/test_resume_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run full suite + smoke**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
.venv\Scripts\paper-distiller-chat.exe resume --help
```

Expected: 150 passed (148 + 2).

- [ ] **Step 6: Commit**

```bash
git add src/paper_distiller/chat/cli.py tests/chat/test_resume_cli.py
git commit -m "feat(chat): paper-distiller-chat resume subcommand (continue paused QA session)"
```

---

## Task 7: End-to-end QA integration test

**Files:**
- Create: `tests/integration/test_ask_e2e.py`

Real on-disk verification: `paper-distiller-chat ask` writes articles + answer-survey to vault, persists state.json correctly.

- [ ] **Step 1: Write the test**

Create `tests/integration/test_ask_e2e.py`:

```python
"""End-to-end integration test for paper-distiller-chat ask — all subsystems mocked.

Tests the full QA loop: 2 rounds of distillation, then synthesis. Vault should
end up with N articles + 1 qa-...md survey + state.json under .paper_distiller/.
"""
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


def test_ask_e2e_writes_qa_survey(mocker, tmp_path, monkeypatch):
    monkeypatch.setenv("PD_API_KEY", "sk-test")
    monkeypatch.setenv("PD_BASE_URL", "https://x/v1")
    monkeypatch.setenv("PD_MODEL", "qwen-plus")

    # 2 rounds: round 1 not done (continue), round 2 not done (continue), round 3 reflection hits max_rounds
    reflections = [
        {"is_done": False, "confidence": 4, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q1",
         "next_query_rationale": "...", "suggest_stop": False},
        {"is_done": False, "confidence": 5, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q2",
         "next_query_rationale": "...", "suggest_stop": False},
        # third reflection triggers max_rounds check before continuing
        {"is_done": False, "confidence": 6, "what_we_know": "...",
         "what_is_missing": "...", "next_query": "q3",
         "next_query_rationale": "...", "suggest_stop": False},
    ]
    mocker.patch("paper_distiller.agents.reflector.reflect", side_effect=reflections)
    mocker.patch(
        "paper_distiller.agents.searchers.arxiv_search",
        side_effect=[[_paper(1), _paper(2)], [_paper(3), _paper(4)]],
    )
    mocker.patch("paper_distiller.agents.searchers.ss_search", return_value=[])
    mocker.patch(
        "paper_distiller.agents.curation.rank",
        side_effect=lambda candidates, topic, top_n, llm: candidates[:top_n],
    )
    mocker.patch(
        "paper_distiller.agents.processor.fetch_with_fallback",
        return_value="x" * 600,
    )
    mocker.patch(
        "paper_distiller.agents.processor.distill_article",
        side_effect=lambda paper, full_text, wiki_index, llm: ArticleResult(
            slug=f"a-{paper.arxiv_id}", title=f"T-{paper.arxiv_id}",
            body=f"body {paper.arxiv_id}", tags=["t"],
            refs=[f"arxiv:{paper.arxiv_id}"], depth="full-pdf",
        ),
    )
    mocker.patch(
        "paper_distiller.agents.processor.load_index",
        return_value=MagicMock(slugs=lambda: set()),
    )
    mocker.patch(
        "paper_distiller.agents.synthesizer.synthesize",
        return_value={
            "title": "QA: 答案", "body": "# answer\n\n...",
            "tags": ["qa"], "cited_slugs": ["a-2501.00001"],
        },
    )

    vault = tmp_path / "vault"
    vault.mkdir()

    from paper_distiller.chat.cli import main
    rc = main([
        "ask", "--vault", str(vault), "--question", "why diffusion?",
        "--max-rounds", "2", "--per-round", "2", "--max-cost-cny", "5",
    ])
    assert rc == 0

    # Articles distilled
    articles_dir = vault / "articles"
    assert len(list(articles_dir.glob("*.md"))) == 4

    # qa-* survey written
    surveys_dir = vault / "surveys"
    qa_surveys = list(surveys_dir.glob("qa-*.md"))
    assert len(qa_surveys) == 1

    # state.json present
    sessions = list((vault / ".paper_distiller" / "qa-sessions").iterdir())
    assert len(sessions) == 1
    state_path = sessions[0] / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["stop_reason"] == "max_rounds"
    assert data["is_done"] is True
```

- [ ] **Step 2: Run, confirm pass**

```bash
.venv\Scripts\python.exe -m pytest tests/integration/test_ask_e2e.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Run full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **151 passed** (150 + 1).

- [ ] **Step 4: Manual smoke (dry-run only — no API spend)**

```powershell
.\.venv\Scripts\paper-distiller-chat.exe ask `
    --vault "G:\Math research Agent\wiki" `
    --question "test" --max-rounds 1 --dry-run
```

Expected: prints `[DRY-RUN] Would run QA loop for 'test'`, exits 0.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_ask_e2e.py
git commit -m "test(chat): end-to-end integration test for ask subcommand"
```

---

## Task 8: Plan-2 wrap-up + push

- [ ] **Step 1: Full suite**

```bash
.venv\Scripts\python.exe -m pytest -q --tb=no
```

Expected: **151 passed** (128 baseline + 23 new).

- [ ] **Step 2: Verify all 5 CLIs respond**

```powershell
.\.venv\Scripts\paper-distiller.exe --help          # v0.5 single-pass (still alive)
.\.venv\Scripts\paper-distiller-qa.exe --help       # v0.5 QA (still alive)
.\.venv\Scripts\paper-distiller-chat.exe --help     # v1.0 chat — now has 3 subcommands
.\.venv\Scripts\paper-distiller-chat.exe distill --help
.\.venv\Scripts\paper-distiller-chat.exe ask --help
.\.venv\Scripts\paper-distiller-chat.exe resume --help
```

- [ ] **Step 3: Push to GitHub**

```bash
git push origin main
```

- [ ] **Step 4: Confirm CI green**

Open https://github.com/jesson-hh/paper-distiller/actions — verify the latest CI matrix passes on Python 3.10/3.11/3.12.

---

## Plan-2 success criteria

- [ ] All 8 tasks done
- [ ] 151 tests passing (128 baseline + 23 new)
- [ ] `paper-distiller-chat ask --question Y` runs end-to-end with rich Live table
- [ ] `paper-distiller-chat resume <sid>` continues a paused session
- [ ] Old `paper-distiller` and `paper-distiller-qa` CLIs unchanged + working
- [ ] CI green on all Python versions
- [ ] No new top-level deps beyond what Plan 1 added

Plan 3 (REPL + intent-router) builds on this. Plan 4 (cleanup + v1.0.0 tag) ships the release.
