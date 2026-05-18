# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
