# Phase 4: Wire Graph-Build into Distillation

> TDD, one commit per task, branch `feat/proof-graph-phase-1-2`.

**Goal:** Make the existing distill path (`distill_by_id` / `ask` / `research` / the `distill` subcommand) build the step-level proof graph, reusing the phase-3 `build_graph_for_paper`. **Gated by `PD_GRAPH_DEPTH` and default OFF** — so existing behavior and cost are unchanged unless the user opts in.

**Architecture:** A thin gating wrapper `maybe_build_graph(...)` reads `PD_GRAPH_DEPTH` and calls the phase-3 pipeline; `_DistillOne.run` calls it (in a thread) after the article + sidecar are produced. Purely additive + gated. The full "one read produces both article and graph" optimization from spec §5 is **out of scope here** (future) — for now the graph build is a separate, opt-in pass; note the extra cost in the wrapper docstring.

**Spec:** `docs/superpowers/specs/2026-05-21-deep-distill-proof-graph-design.md` §3 (the fusion: "distilling is building the graph") and §10 phase 4.

---

## File Structure
- **Modify** `src/paper_distiller/proofgraph/pipeline.py` — add `maybe_build_graph(proof_store, paper_arxiv_id, full_text, *, paper_slug=None, llm) -> CoverageReport | None`.
- **Modify** `src/paper_distiller/agents/processor.py` — in `_DistillOne.run`, after the article is appended and the sidecar ingested, call `maybe_build_graph` via `asyncio.to_thread`, guarded by `self._proof_store is not None`.
- **Create** `tests/proofgraph/test_integration.py` — gating tests for `maybe_build_graph`.
- **Modify** `tests/agents/` (or wherever processor tests live; check `tests/agents/test_processor*.py`) — a wiring test that `_DistillOne.run` invokes `maybe_build_graph`.

Run: `python -m pytest -q` (must stay green: 499 before this phase; the default-OFF gate means existing processor tests are unaffected).

---

## Task 4.1: `maybe_build_graph` gating wrapper

**Files:** Modify `pipeline.py`; Test `tests/proofgraph/test_integration.py`.

- [ ] **Test first** (monkeypatch `build_graph_for_paper` in the pipeline module with a recording stub; pass a dummy `proof_store` object and a stub `llm`):
  - `PD_GRAPH_DEPTH` unset → `maybe_build_graph(...)` returns `None` and the stub is NOT called.
  - `monkeypatch.setenv("PD_GRAPH_DEPTH","step")` → stub called once with `depth="step"`; returns the stub's `CoverageReport`.
  - `setenv("PD_GRAPH_DEPTH","theorem")` → called with `depth="theorem"`.
  - `setenv("PD_GRAPH_DEPTH","garbage")` → treated as off (returns `None`, stub not called).
  - `proof_store=None` → returns `None`, stub not called.
- [ ] **Run → fail.**
- [ ] **Implement:**
```python
import os
_VALID_DEPTHS = {"theorem", "step"}

def maybe_build_graph(proof_store, paper_arxiv_id, full_text, *, paper_slug=None, llm=None):
    """Build the proof graph for a just-distilled paper IF PD_GRAPH_DEPTH is set
    to 'theorem' or 'step' (default off). This is a SEPARATE LLM pass from the
    article distillation (extra cost) — opt-in. Returns the CoverageReport or None.
    Never raises: graph-build failures must not abort distillation."""
    depth = os.getenv("PD_GRAPH_DEPTH", "off").strip().lower()
    if proof_store is None or depth not in _VALID_DEPTHS or not (full_text or "").strip():
        return None
    try:
        return build_graph_for_paper(
            proof_store, paper_arxiv_id, full_text,
            paper_slug=paper_slug, llm=llm, depth=depth)
    except Exception:
        return None  # graph build is best-effort; never break the distill run
```
- [ ] **Run → pass.** Commit: `feat(proofgraph): PD_GRAPH_DEPTH-gated maybe_build_graph wrapper`.

## Task 4.2: wire into `_DistillOne.run`

**Files:** Modify `agents/processor.py`; Test in `tests/agents/`.

Context: `_DistillOne.run` (in `agents/processor.py`) currently: fetches `full_text` via `fetch_with_fallback`, does proof RAG, calls `distill_article(...)`, appends the article to `ctx.shared["articles"]`, and (if `self._proof_store` and the sidecar has theorems) calls `self._proof_store.ingest_sidecar(...)`. The `full_text` and `self._proof_store` and `ctx.llm` are all in scope there.

- [ ] **Test first** (in the processor tests dir; mirror existing processor-test mocking style): build a `_DistillOne` with a fake paper, monkeypatch `fetch_with_fallback` (return a fake full_text), `distill_article` (return a fake ArticleResult with an empty `proof_sidecar.theorems` so the existing ingest path is a no-op), provide a `ctx` with a fake `llm` and a real or fake `proof_store`. Monkeypatch `paper_distiller.proofgraph.pipeline.maybe_build_graph` (or the name imported into processor) with a recording stub. `await dist.run(ctx)`; assert the stub was called once with the fetched `full_text` and the paper's arxiv_id. Also assert that with the stub raising, `run` still completes (best-effort).
- [ ] **Run → fail.**
- [ ] **Implement:** import `maybe_build_graph` into `processor.py`; after the article-append + sidecar-ingest block in `_DistillOne.run`, add:
```python
        if self._proof_store is not None:
            try:
                await asyncio.to_thread(
                    maybe_build_graph,
                    self._proof_store, self._paper.arxiv_id or "", full_text,
                    paper_slug=getattr(article, "slug", None), llm=ctx.llm,
                )
            except Exception:
                pass  # never let graph build abort distillation
```
  Place it inside the existing `try` body after the sidecar ingest, or right after, so a failure can't break the run. Ensure `full_text` is still in scope (it is — it's the fetched text).
- [ ] **Run → pass**, and run the full processor test module to confirm no regression. Commit: `feat(agents): _DistillOne builds proof graph when PD_GRAPH_DEPTH set`.

## Task 4.3: full suite + ruff

- [ ] `python -m pytest -q` → expect 499 + the new tests, all green (existing processor/distill tests unaffected because `PD_GRAPH_DEPTH` is unset by default). `python -m ruff check src/paper_distiller/agents/processor.py src/paper_distiller/proofgraph/pipeline.py` → clean.

---

## Notes
- **Default OFF is the safety net**: existing tests don't set `PD_GRAPH_DEPTH`, so `maybe_build_graph` returns `None` and `_DistillOne` behavior is byte-for-byte unchanged. Use `monkeypatch.setenv` in new tests (auto-cleaned).
- Document `PD_GRAPH_DEPTH` in the README env-var table in a later docs pass (phase 7); not required here.
- Do NOT attempt the one-read-two-artifacts refactor of `distill_article` here — out of scope.
