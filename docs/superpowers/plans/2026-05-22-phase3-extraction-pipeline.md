# Phase 3: Extraction Pipeline Implementation Plan

> **For agentic workers:** implement task-by-task with TDD; one commit per task; checkbox (`- [ ]`) steps. Branch: `feat/proof-graph-phase-1-2`.

**Goal:** Turn a paper's segments (phase 2) into grounded graph nodes/edges (phase 1 store) via a per-segment LLM extraction loop with structured running memory, the grounding gate, self-check, and edge resolution. All LLM calls are mocked in tests.

**Architecture:** New modules under `src/paper_distiller/proofgraph/`. Reuse `LLMClient` (`complete(messages, temperature, response_format="json")`), `segment()` + `verify_quote()` (phase 2), and the `ProofStore` graph API (phase 1: `add_node`/`add_edge`/`delete_paper_graph`/`Node`/`Edge`). LLMLingua is an **optional** dependency.

**Tech Stack:** Python 3.10+, stdlib `json`, the existing `LLMClient`, pytest with the LLM mocked (monkeypatch `llm.complete`). No network in tests.

**Spec:** `docs/superpowers/specs/2026-05-21-deep-distill-proof-graph-design.md` §5 (read it — it is the authoritative narrative for this pipeline). This plan implements phase 3 of §10.

---

## File Structure

- **Create** `src/paper_distiller/proofgraph/compress.py` — `compress(text, instruction=None, target_ratio=0.5) -> str`; lazy-imports `llmlingua`, falls back to identity if unavailable. Never raises.
- **Create** `src/paper_distiller/proofgraph/memory.py` — `RunningMemory` dataclass + `update(nodes, resolved_labels)` + `render() -> str`.
- **Create** `src/paper_distiller/proofgraph/extraction_schema.py` — `ExtractedRef`, `ExtractedNode` dataclasses + `parse_extraction(raw: str | dict) -> list[ExtractedNode]` (robust to junk/missing fields).
- **Create** `src/paper_distiller/proofgraph/prompts/extract.md` — extraction prompt (uses `str.format()` params, mirroring how `prompts/*.md` work elsewhere).
- **Create** `src/paper_distiller/proofgraph/extractor.py` — `extract_segment(segment, memory, llm, depth="step") -> list[ExtractedNode]` (LLM + grounding gate); `self_check(segment, nodes, llm) -> list[ExtractedNode]`.
- **Create** `src/paper_distiller/proofgraph/pipeline.py` — `CoverageReport` dataclass + `build_graph_for_paper(store, paper_arxiv_id, full_text, *, paper_slug=None, llm, depth="step") -> CoverageReport`.
- **Modify** `pyproject.toml` — add an optional-dependency extra `compress = ["llmlingua>=0.2"]`.
- **Create** tests under `tests/proofgraph/`: `test_compress.py`, `test_memory.py`, `test_extraction_schema.py`, `test_extractor.py`, `test_pipeline.py`.

Run: `python -m pytest tests/proofgraph/ -q` (from `G:\paper-distiller`, default `python`).

---

## Data Contracts (define these exactly; later tasks depend on them)

**Extraction JSON the LLM must return (per segment):**
```json
{"nodes": [
  {"kind": "theorem|lemma|definition|assumption|proof_step|claim",
   "label": "Theorem 4.3",
   "text": "normalized assertion",
   "source_quote": "verbatim span copied from THIS segment",
   "techniques": ["Bernstein"],
   "refs": [{"rel": "depends_on|uses_lemma|uses_def|uses_assumption", "target": "Lemma 3.1"}]}
]}
```

**`ExtractedRef`**: `rel: str`, `target: str`.
**`ExtractedNode`**: `kind: str`, `text: str`, `source_quote: str`, `label: str | None = None`, `techniques: list[str] = []`, `refs: list[ExtractedRef] = []`, plus a mutable `status: str = "extracted"` (the gate/self-check set `unsupported`/`suspicious`).

**`RunningMemory`**: `notation: dict[str,str]`, `definitions: list[dict]`, `established: list[dict]` (theorems/lemmas seen), `obligations: list[str]` (referenced-but-unresolved labels). `render()` returns a compact text block (cap each list, e.g. last 20) for prompt injection.

**`CoverageReport`**: `segments_total: int`, `segments_processed: int`, `proof_blocks: int`, `nodes_by_kind: dict[str,int]`, `rejected_quotes: int`, `gaps: int`, `obligations: list[str]`.

---

## Task 3.1: `compress.py` (optional LLMLingua, identity fallback)

**Files:** Create `src/paper_distiller/proofgraph/compress.py`; Modify `pyproject.toml`; Test `tests/proofgraph/test_compress.py`.

- [ ] **Test first:**
```python
def test_compress_identity_fallback_when_llmlingua_absent(monkeypatch):
    import importlib, builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name.startswith("llmlingua"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    from paper_distiller.proofgraph import compress as c
    importlib.reload(c)
    out = c.compress("some long text " * 50, target_ratio=0.5)
    assert isinstance(out, str) and out  # never raises, returns a string
```
- [ ] **Run → fail** (`ModuleNotFoundError: proofgraph.compress`).
- [ ] **Implement:** `compress(text, instruction=None, target_ratio=0.5) -> str` that lazily tries `from llmlingua import PromptCompressor` (module-cached singleton); on `ImportError` (or any exception) returns `text` unchanged. Keep the LLMLingua call path simple; this task only needs the fallback to be solid and tested. Add to `pyproject.toml` under `[project.optional-dependencies]`: `compress = ["llmlingua>=0.2"]`. **Do not** add it to core `dependencies`.
- [ ] **Run → pass.** Commit: `feat(proofgraph): compress.py LLMLingua wrapper with identity fallback`.

## Task 3.2: `memory.py` RunningMemory

**Files:** Create `memory.py`; Test `test_memory.py`.

- [ ] **Test first:** construct `RunningMemory()`; call `update(...)` with a definition node and a theorem node plus an unresolved ref label; assert the definition lands in `definitions`, the theorem in `established`, the unresolved label in `obligations`; assert `render()` returns a non-empty `str` mentioning a known label and stays bounded when fed 100 items (cap check).
- [ ] **Run → fail.**
- [ ] **Implement** the dataclass + `update(nodes: list[ExtractedNode], resolved_labels: set[str])` (definitions→`definitions`, theorem/lemma→`established`, any ref target not in `resolved_labels`→`obligations` deduped) + `render()` (compact, capped).
- [ ] **Run → pass.** Commit: `feat(proofgraph): RunningMemory structured carry-forward state`.

## Task 3.3: `extraction_schema.py` parse

**Files:** Create `extraction_schema.py`; Test `test_extraction_schema.py`.

- [ ] **Test first:** `parse_extraction('{"nodes":[{"kind":"proof_step","text":"t","source_quote":"q","refs":[{"rel":"depends_on","target":"L1"}]}]}')` → one `ExtractedNode` with one `ExtractedRef`; `parse_extraction("garbage")` → `[]`; missing `text`/`source_quote` → that node skipped; non-list `refs`/`techniques` → coerced to `[]`.
- [ ] **Run → fail.**
- [ ] **Implement** dataclasses + `parse_extraction(raw)` accepting a JSON string or dict, tolerant of junk (try/except, type guards), dropping nodes lacking `kind`+`text`+`source_quote`.
- [ ] **Run → pass.** Commit: `feat(proofgraph): extraction JSON schema + tolerant parser`.

## Task 3.4: `extractor.extract_segment` (LLM + grounding gate)

**Files:** Create `extractor.py` + `prompts/extract.md`; Test `test_extractor.py`.

- [ ] **Test first** (mock the LLM): make a fake `llm` whose `.complete(...)` returns a canned JSON string with two nodes — one whose `source_quote` IS a verbatim substring of the segment, one whose `source_quote` is fabricated (not in the segment). Assert `extract_segment(seg, RunningMemory(), llm)` returns only the grounded node (the fabricated one dropped or marked `status="unsupported"` and excluded from the accepted list). Use a `Segment` from `segment(...)` or construct one directly.
- [ ] **Run → fail.**
- [ ] **Implement:** load `prompts/extract.md`, format with `memory.render()`, `segment.text`, `segment.kind_hint`, and `depth`; call `llm.complete([{role:user,content:prompt}], temperature=0.2, response_format="json")`; `parse_extraction`; for each node run `verify_quote(node.source_quote, segment.text)` — keep nodes that pass; on fail, retry the LLM **once** with a "your quote wasn't found verbatim, fix it" follow-up, then drop/mark `unsupported` if still failing. Return accepted nodes. Write `prompts/extract.md` instructing the model to emit ONLY the JSON contract above, copy `source_quote` verbatim from the provided segment, and abstain (`refs` empty, mark nothing) rather than invent. (See spec §5 for the contract + path-adherence rules.)
- [ ] **Run → pass.** Commit: `feat(proofgraph): extract_segment with grounding-gate enforcement`.

## Task 3.5: `extractor.self_check`

**Files:** Modify `extractor.py`; Test `test_extractor.py`.

- [ ] **Test first** (mock LLM returning a verdict JSON like `{"suspicious_labels":["Step 2"]}`): assert `self_check(seg, nodes, llm)` flips the matching node's `status` to `"suspicious"` and leaves others unchanged. Empty/garbled LLM output → all nodes unchanged (no crash).
- [ ] **Run → fail.**
- [ ] **Implement** a cheap LLM pass over (segment text + the accepted nodes) asking which nodes assert beyond the text; mark returned ones `suspicious`. Tolerate junk output.
- [ ] **Run → pass.** Commit: `feat(proofgraph): self_check pass marks over-reaching nodes suspicious`.

## Task 3.6: `pipeline.build_graph_for_paper` (orchestrate + edges + coverage)

**Files:** Create `pipeline.py`; Test `test_pipeline.py`.

- [ ] **Test first** (mock LLM end-to-end on a tiny fake paper): construct a `ProofStore(tmp_path/"proofs.db")`. Build a `full_text` with a heading, a `Theorem 1` segment, and a `Proof.` segment whose extraction (canned per-segment LLM JSON, dispatched by call count or segment content) yields: a theorem node `label="Theorem 1"`, a proof_step that `depends_on` target `"Theorem 1"` (resolvable → edge), and a proof_step that `depends_on` target `"Lemma 9"` (unresolvable → gap). Assert: nodes written to the store for the paper; exactly one `depends_on` edge created (step→Theorem 1); the dangling step has `status="gap"`; `CoverageReport.segments_processed == segments_total`, `report.gaps == 1`, `report.nodes_by_kind` sums to the node count. Re-running `build_graph_for_paper` for the same paper does NOT duplicate (idempotent via `delete_paper_graph`).
- [ ] **Run → fail.**
- [ ] **Implement** `build_graph_for_paper(store, paper_arxiv_id, full_text, *, paper_slug=None, llm, depth="step")`:
  1. `store.delete_paper_graph(paper_arxiv_id)` (idempotency).
  2. `segs = segment(full_text)`.
  3. `memory = RunningMemory()`; `label_to_id: dict[str,int] = {}`; `pending: list[tuple[int, list[ExtractedRef]]] = []`; counters.
  4. For each segment (skip detailed steps when `depth=="theorem"` and `seg.is_proof_block` — still extract the statement-level node if any): `nodes = extract_segment(seg, memory, llm, depth)`; `nodes = self_check(seg, nodes, llm)`; for each accepted node, `nid = store.add_node(Node(paper_arxiv_id=..., paper_slug=..., kind=..., label=..., text=..., source_quote=..., loc=json of {sec,char_start}, status=node.status, techniques=node.techniques))`; if `label`, record `label_to_id[label]=nid`; append `(nid, node.refs)` to `pending`. Update `memory`. Tally `nodes_by_kind`, `rejected_quotes` (from extract_segment drops — return a small stats tuple or read store), `segments_processed`.
  5. Resolve edges: for `(nid, refs)` in `pending`, for each `ref`: `target_id = label_to_id.get(ref.target)`; if found → `store.add_edge(Edge(src_id=nid, dst_id=target_id, rel=ref.rel))`; else → set that node's `status="gap"` (UPDATE nodes SET status='gap' WHERE id=nid) and append `ref.target` to obligations + increment `gaps`.
  6. Return `CoverageReport(...)`.
  Keep `pipeline.py` focused; it orchestrates, it doesn't re-implement extraction.
- [ ] **Run → pass.** Commit: `feat(proofgraph): build_graph_for_paper orchestration + edge resolution + coverage`.

## Task 3.7: full suite + ruff

- [ ] Run `python -m pytest -q` (expect prior 454 + the new phase-3 tests, all green). Run `python -m ruff check src/paper_distiller/proofgraph/`. Expected `All checks passed!`. No commit needed unless ruff fixes are required (then commit `style(proofgraph): ruff fixes`).

---

## Notes for the implementer
- **Mock the LLM** by passing a stub object with a `.complete(messages, temperature=..., response_format=...)` method returning canned strings. Do NOT call a real API. There is no network in tests.
- Follow existing prompt convention: `.md` files with `str.format()` named params, loaded via `Path(__file__).parent / "prompts" / "extract.md"`.
- `Node.loc` is a JSON string; store `{"sec": seg.section, "char_start": seg.char_start}`.
- Abstain-over-fabricate: never invent a `source_quote`; the gate enforces this, and the prompt must instruct it.
- Keep each new file focused (one responsibility). If `extractor.py` grows past ~200 lines, that's expected; flag if it exceeds ~300.
