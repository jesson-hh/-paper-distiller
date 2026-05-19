# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] — 2026-05-19

### Added
- **`OpenCLIOpenAlexSearcher` agent** — wraps the [OpenCLI](https://github.com/jackwener/OpenCLI) Node CLI's `openalex` adapter via subprocess. Returns Paper-shaped results with abstracts, DOIs, PDF URLs, citation counts, and venue metadata. OpenAlex offers **100 req/s** rate limit (vs Semantic Scholar's much stricter limits), covering ~250M scholarly works. No API key needed. arxiv-DOI papers are auto-detected (DOI `10.48550/arxiv.X` → `arxiv_id` populated).
- **`--source` flag accepts two new values**:
  - `openalex` — OpenAlex only (rate-limit-safe)
  - `all` — arxiv + ss + openalex parallel (NEW DEFAULT)
- **`CandidateMerger` merges 3 sources** via two-pass merge (arxiv beats ss beats openalex on tie).
- **REPL natural-language path** now defaults to `source="all"` for distill / ask / research.

### Changed
- **Default `--source` for new sessions is now `"all"`** (previously `"both"`). `"both"` is preserved as `arxiv + ss` for backward compatibility — old scripts and resumed v1.1.x sessions still work unchanged.
- **`CandidateMerger.deps`** now includes `"openalex-searcher"` alongside the original two searchers. Bypass mode (`shared["candidates_direct"]`) unchanged — still requires `merger.deps = []` instance override.

### Fixed
- **Searcher graceful degradation on HTTP 429** — if `arxiv-searcher` or `ss-searcher` hits a transient HTTP failure (429, timeout, connection reset), the agent now catches the error, returns an empty list, and logs a stderr warning. The other source(s) continue; the pipeline downstream sees just whichever source succeeded. Previously, one source's 429 would `AgentFailed` the entire DAG.

### External requirement (for `--source openalex` / `--source all`)
- Install Node 21+ and OpenCLI: `npm install -g @jackwener/opencli`. Verified working at OpenCLI v1.7.22. No Chrome extension needed for the `openalex` / `arxiv` adapters (`Browser: no` in their metadata).

### Internal
- 8 new tests (7 for OpenCLI agent + 1 for 3-source merger + 3 for searcher graceful degradation). Total: **212** (was 201).
- No new Python runtime deps. OpenCLI is an optional external tool — paper-distiller's PyPI install footprint unchanged.

## [1.1.0] — 2026-05-19

### Added
- **`paper-distiller-chat research`** — long-running (default 4h) autonomous "deep dive" loop on a single research question. Runs a 5-phase rolling cycle (seed → expand → structure → synthesize → gap-check) that produces distilled articles with structured frontmatter, theme-synthesis docs, and one top-level research report. State persisted per phase; resumable via `--resume <sid>`.
- **4 new agents** in `agents/`:
  - `CitationExplorer` — pulls references + cited-by from Semantic Scholar's `/paper/<id>/{references,citations}` endpoints, ranks candidates by Jaccard token-overlap on (question + seed title) vs (candidate title + abstract).
  - `ThemeClusterer` — single LLM call grouping articles into 2-5 themes; falls back to one "Mixed" bucket on JSON failure.
  - `TheoremExtractor` — extra LLM pass per article extracting `theorems` / `assumptions` / `convergence_rates` / `key_lemmas`. Parallel via `asyncio.gather`. Enables Dataview queries like "all papers assuming Lipschitz score".
  - `GapDetector` — LLM judges whether the research loop has remaining gaps; replaces the original iteration-counter heuristic. Returns `{should_continue, missing_aspects, next_query, rationale}`.
- **`ResearchState`** dataclass + disk persistence under `<vault>/.paper_distiller/research-sessions/<sid>/state.json`. Mirrors `SessionState` semantics.
- **`load_config_research()`** in `config.py` with research-mode fields: `research_max_papers` (default 30), `research_max_cost_cny` (default ¥30), `research_max_duration_sec` (default 4h), `research_resume_session_id`.
- **Interactive slot-filling in REPL** — natural-language input goes through `IntentRouter`, then for each missing param the REPL asks `name (default V): ` interactively, type-coerced (int/float/str/bool). Empty input uses default. Final preview + Y/n confirm before dispatch. Applies to all commands (distill / ask / resume / research).

### Changed
- **`CandidateMerger` bypass mode** — accepts `ctx.shared["candidates_direct"]` for direct candidate injection (used by research-mode Phase 2 to skip the searcher chain). Callers must override `merger.deps = []` on the instance to satisfy DAG validation.
- **Research-mode Phase 2 design** — uses the new `CandidateMerger` bypass + instance dep override, NOT a stub-searcher hack. Cleaner DAG.

### Internal
- 28 new unit + integration tests (4 citation-explorer + 3 theme-clusterer + 3 theorem-extractor + 5 gap-detector + 3 research-state + 5 research-runner + 2 research-cli + 1 e2e + 2 slot-filling). Total: **201** (was 173).
- No new runtime deps.

## [1.0.0] — 2026-05-19

**BREAKING CHANGE.** Major rewrite around a chat-first interface backed by an async sub-agent framework. The old `paper-distiller` (single-pass) and `paper-distiller-qa` (multi-round) console scripts are **removed** and replaced by a single `paper-distiller-chat` entry point with three subcommands (`distill` / `ask` / `resume`) and an interactive REPL when invoked without a subcommand.

### Added
- **`paper-distiller-chat` console script** with three one-shot subcommands and an interactive REPL:
  - `distill --topic X --n N` — single-pass mode (replaces v0.5 `paper-distiller`).
  - `ask --question Y` — multi-round QA loop (replaces v0.5 `paper-distiller-qa`).
  - `resume --session-id <sid>` — continue a paused/errored QA session.
  - No subcommand → opens interactive REPL with welcome banner, 10 slash commands, and natural-language input routed via an LLM intent-router.
- **`paper_distiller.agents/` package** — new async DAG framework. `Agent` protocol, `Context` dataclass, `Status` enum, `DAG` class (topology validation + topological levels), `Orchestrator` (asyncio executor with parallel-sibling scheduling), `FanoutAgent` protocol (runtime expansion into parallel sub-agents), `ConsoleRenderer` (rich live status table).
- **11 concrete agents** — `arxiv-searcher`, `ss-searcher`, `candidate-merger`, `candidate-dedup` (in-session dedup against seen_ids), `candidate-ranker`, `paper-processor` (fanout × N), `vault-writer`, `survey-composer`, `progress-reflector`, `answer-synthesizer`, `intent-router`.
- **REPL slash commands**: `/distill`, `/ask`, `/resume`, `/sessions`, `/vault`, `/provider`, `/agents`, `/show`, `/help`, `/quit`. All deterministic (no LLM call).
- **Natural-language routing**: any non-slash input goes through `IntentRouter` (one LLM call with the routing prompt), produces a classified command + extracted params + missing-params list, and asks the user to confirm before executing.
- **Live status table** (via the rich library) for all multi-step operations, showing per-agent status + elapsed time.

### Changed
- **Wire format unchanged.** Vault files, frontmatter, `[[wikilink]]` cross-references all unchanged from v0.5. Existing vaults open seamlessly with v1.0.
- **QA-mode now writes `answer_survey_slug`** to `<vault>/surveys/qa-<slug>-<date>.md` (was already true in v0.5 but the answer-survey is now produced by an agent rather than the procedural loop).
- **Documentation rewrite**: `README.md` and `docs/ARCHITECTURE.md` rewritten around the v1.0 chat-first interface.

### Removed (BREAKING)
- `paper-distiller` console script.
- `paper-distiller-qa` console script.
- `src/paper_distiller/cli.py`.
- `src/paper_distiller/qa/cli.py`.
- `src/paper_distiller/qa/loop.py` — its logic is preserved in `chat/qa_runner.py` + the 3 new QA agents.

### Migration

If you have scripts calling `paper-distiller --vault X --topic Y --n N`, replace with:

```bash
paper-distiller-chat distill --vault X --topic Y --n N
```

If you have scripts calling `paper-distiller-qa --vault X --question Y ...`, replace with:

```bash
paper-distiller-chat ask --vault X --question Y ...
```

The flag names, defaults, and behavior are otherwise preserved.

### Internal
- New runtime deps: `rich>=13` (status table), `prompt_toolkit>=3` (REPL input + tab completion).
- New dev dep: `pytest-asyncio>=0.23` (for the async test suite).
- 168 tests passing (was 78 at v0.5 start; +50 from Plan 1 framework + Plan 2 QA + Plan 3 REPL; -11 deleted with the removed CLIs).
- CI matrix: Python 3.10 / 3.11 / 3.12 on Ubuntu.

## [0.5.1] — 2026-05-19

First PyPI release. No code changes beyond engineering setup; the v0.5.0 feature set is unchanged.

### Engineering
- **GitHub Actions CI** — `pytest` on Python 3.10 / 3.11 / 3.12 (Ubuntu), triggered on push to main and on pull request.
- **GitHub Actions Release workflow** — on `v*` tag push (or manual dispatch): builds wheel + sdist, publishes to PyPI via OIDC trusted publishing (no token), attaches artifacts to a GitHub Release with auto-generated notes.
- **`docs/ARCHITECTURE.md`** — full module map, L2 single-pass and L3 multi-round data flow, the seven stop reasons explained, prompt-template locations, cost-accounting math.
- **README rewrite** — `paper-distiller-qa` now mentioned throughout (What it does / Quick start / How it works / CLI reference / Customizing prompts); PyPI install is now the primary path; status section reflects v0.5.x; badges switched to dynamic PyPI versions.
- **`pyproject.toml`** project URLs point at the renamed `github.com/jesson-hh/paper-distiller` repo.

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
