# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.5.0] — 2026-05-18

### Added
- **Question-driven multi-round research loop (`paper-distiller-qa`).** Given a research question, the agent autonomously plans search queries, distills relevant papers across multiple rounds, and synthesizes a cited answer document written to `<vault>/surveys/qa-<slug>-<YYYYMMDD>.md`. Bounded by hard budget (rounds/articles/cost) + LLM "is_done" judgment + diminishing-returns detection.
- **Seven stop reasons** surfaced in the final survey footer + terminal summary: `max_rounds`, `llm_done`, `llm_brake`, `no_candidates`, `max_articles`, `max_cost`, `user_quit`.
- **`--interactive` mode** pauses after each round and prompts to continue (Y/n/q) — useful for prompt debugging and untrusted-question runs.
- **`--resume <session-id>` mode** picks up a paused or crashed session from disk-persisted state (`<vault>/.paper_distiller/qa-sessions/<sid>/state.json`).
- **Audit trail** rendered as a markdown table in every qa-survey doc: per-round query, LLM rationale, new articles, confidence.
- **`qa/state.py`** — `SessionState` + `RoundRecord` dataclasses with disk persistence.
- **Two new prompt templates** in `src/paper_distiller/qa/prompts/`: `reflect.md` (LLM judges loop progress) and `answer.md` (LLM synthesizes final cited answer).

### Changed
- **Pipeline helper promotion.** `pipeline._gather_candidates` and `pipeline._fetch_with_fallback` aliased to public names (`gather_candidates`, `fetch_with_fallback`) for qa-loop reuse. Old underscore names retained for v0.3 back-compat.
- **`Config` extended** with `qa_max_rounds`, `qa_max_articles`, `qa_max_cost_cny`, `qa_confidence_threshold`, `qa_per_round`, `qa_interactive`, `qa_resume_session_id`, `qa_question` (all defaults; v0.3 callers unaffected). New `load_config_qa()` validates qa-specific kwargs.

### Internal
- 16 new unit/integration tests (3 state + 3 reflection + 3 answer + 5 loop + 2 cli); total now **77** (was 61 in v0.3).
- No new runtime dependencies.

### Note on v0.4 gap
v0.4 was explored as a self-shipped LEANN-backed MCP server, then reverted in favor of recommending [vault-mcp](https://github.com/robbiemu/vault-mcp). See [docs/vault-mcp-recommendation.md](docs/vault-mcp-recommendation.md) and the README "Optional companion" section. No v0.4 tag exists; v0.5 is the next semantic-version bump.

## [0.3.0] — 2026-05-18

### Added
- **Semantic Scholar as second paper source.** New `sources/semantic_scholar.py` module exposes `search`, `lookup_by_arxiv_id`, `lookup_by_doi` returning the unified `Paper` dataclass.
- **`--source {arxiv,ss,both}` CLI flag** (default `both`). When `both`, pipeline searches both APIs in series, then `merge_candidates` dedupes by `arxiv_id` and `doi` (arxiv-sourced wins on conflict). When `arxiv` or `ss` solo, only that source is searched and errors propagate.
- **PDF fallback chain**: if a paper's primary PDF download fails AND the paper has an arxiv id or DOI, pipeline queries SS for `openAccessPdf` and tries that URL before falling back to abstract-only.
- **`VaultStore.find_by_doi`** mirrors `find_by_arxiv_id` semantics for DOI-based vault dedup.
- **`Config.source` and `Config.ss_api_key`** (the latter read from optional `PD_SS_API_KEY` env var).
- **`download_pdf_from_url(url, dest_dir, filename, timeout)`** as the URL-based primitive used by both arxiv direct fetch and SS fallback.

### Changed
- **`ArxivPaper` → unified `Paper` dataclass.** Adds `source`, `paper_id`, `arxiv_id`, `doi`, `ss_paper_id`, `venue`, `open_access_pdf_url` fields. `ArxivPaper` is kept as a module-level alias so v0.2 imports continue to work.
- **`distill/article.py` refs injection** now prefers arxiv id, falls back to DOI, then to SS paper id — fixes a NoneType bug that would have shipped if v0.3 hadn't generalized the dataclass.
- **Pipeline dedup** checks both `find_by_arxiv_id` and `find_by_doi` (in that priority order) before the slug-based fallback.

### Internal
- 10 new unit/integration tests; total now 61 (was 51 in v0.2).
- No new runtime dependencies (Semantic Scholar API is plain HTTPS via existing `httpx`).
- `.env.example` now documents the optional `PD_SS_API_KEY`.

## [0.2.0] — 2026-05-18

### Added
- `VaultStore.find_by_arxiv_id(arxiv_id)` — look up an article by its arxiv ref. Used by the pipeline for precise dedup.
- Pipeline: arxiv-id-based dedup runs ahead of the slug-based fallback. Prevents creating a sibling article (e.g. `cofindiff.md`) when one already exists for the same arxiv paper under a different slug (e.g. hand-written `cofindiff-controllable-financial-diffusion.md` with `refs: ["arxiv:2503.04164"]`).
- Verbose mode now logs which existing entry caused a dedup skip.

### Fixed
- `distill/article.py` now uses `len(full_text) > 500` as the threshold for "full-pdf" mode. v0.1's truthy check would tag a 50-byte garbage extraction from a scanned PDF as full-pdf and feed it to the LLM as the paper's content. Now such cases correctly fall back to abstract-only with the ⚠️ callout.

### Internal
- 6 new unit/integration tests; total now 51 (was 45 in v0.1.0). The 6 are: 3 vault (find_by_arxiv_id hit/miss/articles-only), 2 pipeline (arxiv-id dedup happy + force override), 1 article (short-extract fallback).
- No new runtime dependencies.

## [0.1.0] — 2026-05-18

### Added
- L2 single-pass search-and-distill pipeline (arxiv search → LLM filter → PDF fetch → text extract → LLM distill → vault save → optional session survey)
- arxiv source module: `search()` and `download_pdf()` (httpx streaming)
- PyMuPDF-based text extraction
- OpenAI-compatible LLM client (Aliyun Bailian default, supports DeepSeek/OpenRouter/Ollama/etc. via `PD_BASE_URL`)
- 3 markdown prompt templates (filter / article / survey) — user-editable, no Python changes needed
- VaultStore: Obsidian markdown CRUD with YAML frontmatter, path-traversal-safe
- Default 6-category schema (articles / techniques / directions / open-problems / authors / surveys)
- Crosslink index loader — feeds existing slugs to LLM, post-write scrub of hallucinated `[[wikilinks]]`
- CLI: `--topic` / `--author` / `--n` / `--pool` / `--force` / `--dry-run` / `--verbose` / `--model` / `--provider`
- Per-run JSONL log at `<vault>/.paper_distiller/runs.jsonl` (.dot-prefix keeps it out of Obsidian's default view)
- 45 unit tests + 3 integration tests
- Friendly error handling: arxiv/LLM exceptions wrapped, no raw stack traces (use `--verbose` for full traceback)

### Tested
- End-to-end smoke against Aliyun Bailian (`qwen3.5-plus`): distilled CoFinDiff paper (arxiv:2503.04164) with 4 valid `[[wikilinks]]` to existing entries, 24K in / 7K out tokens, ~¥0.15 per paper.
