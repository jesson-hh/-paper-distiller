# Changelog

All notable changes documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.9.0] — 2026-05-21

### Added — three-way RAG retrieval for proof context

v1.8 introduced cross-paper RAG injection but only used a hardcoded ~50-name
keyword scan over the abstract. Many math papers don't mention specific
technique names in their abstracts, so RAG was firing on too few papers.
v1.9 adds two more retrieval strategies so coverage stays high as the
proof store grows.

- **Strategy A (extended) — store-augmented keyword scan**: in addition to
  the hardcoded list, scan the abstract against every technique name ever
  registered in the ProofStore. As the vault accumulates papers and the
  store learns more canonical technique names, the keyword scan auto-grows.
- **Strategy B — FTS5 text match**: `ProofStore.retrieve_by_text_match(text)`
  runs a BM25 query against `theorems_fts` (statement + proof_sketch +
  name) using tokens from the new paper's title + abstract. Catches
  relevant prior theorems when the new paper's vocabulary doesn't match any
  registered technique name. Stop-word filtered, deduplicated, capped.
- **Strategy C — LLM pre-extract**: a small dedicated LLM call before main
  distillation asks the model to list 5-10 specific math techniques the
  paper likely uses. Catches papers where the abstract is high-level
  ("conditional sampling") but the techniques are inferable (f-divergence,
  Pinsker, Fenchel duality). Opt-out via `PD_LLM_TECH_EXTRACT=0`. Adds
  ~¥0.005 + ~5-10s per paper.

`PaperProcessor._DistillOne.run` now combines all three sources:
1. `_gather_candidate_techniques(paper, store, llm)` → A + C union
2. `store.retrieve_relevant(candidates)` → theorems indexed by technique
3. `store.retrieve_by_text_match(title + abstract)` → theorems by FTS match
4. Merge + dedupe (technique-matches first), cap 12 theorems

The progress activity now reports `proof RAG: N candidates · M prior theorems
(technique:X + text:Y)` so you can see all 3 strategies' contributions live.

### Internal

- New helpers: `_llm_extract_techniques(paper, llm)`, `_gather_candidate_techniques`,
  `ProofStore.retrieve_by_text_match`, `ProofStore.list_canonical_technique_names`
- `_STOPWORDS` constant in store.py for FTS5 token filtering
- **+9 new tests** in `tests/proofs/test_three_way_retrieval.py`. Total: **401**.

### Backward compatibility

- `_extract_candidate_techniques` (the simple v1.8 helper) is kept and still
  used internally. v1.8 behavior is the lower bound of v1.9 retrieval.
- `PD_LLM_TECH_EXTRACT=0` disables Strategy C if cost matters.

## [1.8.0] — 2026-05-21

### Added — proof / technique knowledge base + cross-paper RAG

- **`paper_distiller.proofs` module**: SQLite + FTS5 store of theorems / techniques extracted during each paper's distillation. Stored per-vault at `<vault>/.proof_store/proofs.db`. Supports retrieval by technique name (LIKE on JSON), FTS5 full-text search over theorem statements + proof sketches, and dedup-aware multi-technique retrieval (`retrieve_relevant`).
- **`proof_sidecar` extraction**: the article distillation prompt now also produces a structured JSON sidecar alongside the markdown body, with three lists: `theorems` (name, statement, proof_sketch, techniques_used), `key_definitions` (name, statement), `key_techniques` (canonical short names like "Hölder", "Bernstein concentration"). LLMs are instructed to capture 3-8 numbered theorems per paper and 5-15 normalized technique names.
- **Cross-paper RAG during distillation**: before LLM distillation, `PaperProcessor` keyword-scans the paper's abstract + title for ~50 known technique names, queries the ProofStore for prior theorems using those techniques, and injects them as a "已知相关定理" markdown block at the top of the LLM prompt. The LLM can now reuse notation, cite prior work as "[[paper X]]'s Theorem 4.3", and flag duplicates / contradictions.
- **Idempotent per-paper ingest**: re-distilling a paper deletes its prior theorems and re-inserts cleanly — no duplicates.

### Internal

- `_extract_candidate_techniques(paper)` — cheap keyword scanner with ~50 hand-curated technique names (Hölder, Bernstein, Dudley chaining, Wasserstein, PAC-Bayes, sub-Gaussian, ReLU approximation, RKHS, etc.)
- `_proof_store_lock` (module-level asyncio.Lock) serializes ProofStore writes across the fanout sub-agents
- ProofStore.open_for_vault() — per-vault factory, isolates research projects from each other
- **+16 new tests** for proofs module (10 store tests + 6 distill-integration tests). Total: **392** (was 376).

### Backward compatibility

- Distillation works without a ProofStore — `proof_store=None` path returns sensible defaults and skips RAG injection.
- LLMs that don't produce `proof_sidecar` in their response → `ProofSidecar.from_json` returns empty defaults instead of crashing.
- Tests updated to handle the new `prior_theorems=None` parameter on `distill_article()`.

## [1.7.0] — 2026-05-21

### Added — deep distillation + bigger research defaults

- **12-section article template** (was 6) — every paper distillation now includes: TL;DR, motivation, setup/notation, core method (with sub-sections for ideas/algorithm/theory), key theorems, experimental setup (datasets/baselines/metrics/hardware), top results with concrete numbers, ablations, limitations/failure modes, wiki crosslinks, reproduction notes, my take, citation network. Length target raised from **600-1500 chars to 3000-6000 Chinese chars** per article — researcher-grade lab-notebook depth.
- **Paper text window: 120K → 250K chars** sent to LLM. qwen3.5-plus / qwen-plus support 128K tokens (~400K chars) of context; we now use most of it instead of aggressively truncating.
- **Fanout concurrency cap** via `asyncio.Semaphore` (default 5 concurrent LLM calls per fanout). Override with `PD_FANOUT_CONCURRENCY` env. Prevents Aliyun rate-limit cascades when distilling many papers in parallel during `research` mode.
- **`ask` defaults bumped**: max_rounds 3→5, per_round 2→3, max_articles 10→15, max_cost_cny ¥5→¥10.
- **`research` defaults bumped**: duration 2h→6h, max_papers 20→40, max_cost_cny ¥15→¥30. Reflects that each paper now costs ~2-3× more to distill at the new depth target.

### Internal

- 3 new fanout-concurrency tests (`test_fanout_respects_concurrency_cap`, etc). Total: **376** (was 373).
- Schema descriptions for `ask` and `research` mention the new depth + fanout concurrency so the LLM agent knows what behavior to expect.

### Backward compatibility

- Existing single-shot subcommands (`distill / browse / ask / research`) keep working.
- Schema parameter names + types unchanged; only defaults shifted.
- If a user explicitly passes `max_papers=20` or `duration="2h"`, old behavior is preserved.

## [1.6.1] — 2026-05-20

### Fixed
- **SQLite cross-thread crash** when `ArxivSearcher` ran via `asyncio.to_thread` — `Store` now opens connections with `check_same_thread=False` (safe under WAL with single-writer serialization). Without this fix, every `paper-distiller-chat` search after a non-empty local mirror would fail with `SQLite objects created in a thread can only be used in that same thread`.

### Changed
- **Bootstrap default source chain** swapped to `[oai_pmh, internet_archive, kaggle]` (was `[internet_archive, kaggle]`). The Internet Archive `arxiv-bulk-metadata` item only hosts 2017-2018 XML dumps (not the modern JSONL we expected); Kaggle requires API credentials. OAI-PMH is the only path that works out-of-the-box with no auth and current data.
- **`paper-distiller-arxiv bootstrap --since DATE`** added. Lets you bound the OAI-PMH harvest to a specific date range — e.g. `--since 2024-01-01` pulls only the last ~2 years (~600k papers, ~2h) instead of the full 1.7M catalog (~6-7h).

### Internal
- 3 new bootstrap tests covering the new chain order + `--since` threading + full-catalog default. Total: **~365** tests passing.

## [1.6.0] — 2026-05-20

### Added — local arxiv metadata mirror (bypass arxiv API rate limits)

- **`paper_distiller.arxiv_local` package** — SQLite + FTS5 store for the full arxiv metadata catalog (~1.7M papers, ~5 GB after bootstrap). Search queries no longer hit arxiv's live API — they run against the local index at <10 ms.
- **`paper-distiller-arxiv` CLI** with subcommands:
  - `bootstrap` — one-time bulk ingest from Internet Archive (no auth, default) / Kaggle / OAI-PMH from scratch. Auto-fallback chain.
  - `sync` — incremental OAI-PMH update since `last_sync` (~5 min when run daily).
  - `search` — local FTS5 query (BM25 ranking or `--sort date`).
  - `stats` — paper count, DB size, last sync time.
  - `doctor` — diagnose DB integrity + OAI-PMH reachability.
- **`LocalFirstFetcher`** — transparent local/live switch behind `ArxivSearcher`. When the local mirror is populated, the agent's `search` queries hit the local DB. Cold start (empty DB) transparently falls through to the v1.5 live-API path; existing users see no behavior change until they bootstrap.
- **`PD_ARXIV_LOCAL_ONLY=1`** env flag — disables live-API fallback for air-gapped / offline use.

### Internal

- New dep: `sickle>=0.7` (OAI-PMH client library).
- SQLite schema versioned via `meta.schema_version` for future migrations. FTS5 with `porter unicode61` tokenizer + BM25 ranking.
- `~/.paper-distiller/arxiv/arxiv.db` shared across all vaults / sessions (override via `PD_ARXIV_LOCAL_DIR`).
- Bootstrap source chain (Internet Archive → Kaggle → OAI-PMH) handles partial failures gracefully.
- **+45 new tests** across 6 modules (`store`, `bootstrap`, `incremental`, `search`, `fetcher`, `cli`). Total: **362** (was 317).

### Backward compatibility

- If user hasn't run `paper-distiller-arxiv bootstrap`, search behavior is identical to v1.5 (live arxiv API with throttle + cooldown).
- All existing CLI commands and tools unchanged.

## [1.5.0] — 2026-05-19

### Added — Claude Code-style UX + agent control

- **`ask_user` tool** — sixth LLM-callable tool. Lets the agent pause and ask the user a multi-choice question (2-4 options) for genuine ambiguity (which papers to distill, budget confirmation, vague-intent disambiguation). System prompt steers it away from trivial confirmations.
- **Slash commands** alongside natural language: `/clear /cost /help /show /history /exit /auto`. They bypass the LLM entirely — `/cost` shows session token + ¥ totals, `/show <slug>` reads a vault entry directly, `/auto` toggles plan-mode previews off.
- **Streaming text output** via `LLMClient.complete_with_tools_stream()`. Aliyun Bailian SSE chunks are accumulated into the assistant's reply incrementally. Tool calls assembled from chunked function-name + arguments. Streaming is on by default; the loop falls back to the non-streaming path for providers that don't implement it.
- **Session cost display** — every turn ends with `tokens in/out: X / Y  ·  ¥Z.ZZZZ`. New `llm/pricing.py` table covers qwen-plus / qwen-turbo / qwen-max / deepseek-chat / deepseek-reasoner; unknown models log a warning and use a conservative default. Override per-session via `PD_PRICE_IN_CNY_PER_M` / `PD_PRICE_OUT_CNY_PER_M`.
- **Plan mode** — when a tool's estimated cost exceeds `PD_PLAN_THRESHOLD_CNY` (default ¥10), the loop prints a structured preview card (tool name + arguments + estimated budget) and waits up to 5 seconds for the user to press Enter (proceed) or `q` (cancel). Auto-proceeds on timeout. Always-skip via `/auto` slash command.
- **Ctrl-C abort state machine** — single press cancels the currently running tool (the loop posts a cancelled-result to the LLM, conversation continues); two presses within 1.5s exits the REPL.

### Internal

- New files: `llm/pricing.py`, `chat/slash_commands.py`, `chat/cost_estimator.py`, `chat/plan_mode.py`, `chat/abort.py`.
- `LLMClient` gains `complete_with_tools_stream()`, `StreamChunk` dataclass, and `estimated_cost_cny` property.
- `AgentLoop.send()` rewritten to consume StreamChunks and accumulate tool calls across chunks. `_stream_one_response`, `_render_text_delta`, `_end_text_render`, `_execute_one_tool_call` split out for testability.
- **+48 new tests**: 5 pricing + 4 streaming + 11 slash + 6 ask_user + 12 plan-mode + 5 abort + 5 agent_loop integration. Total: **307** (was 259).

### Backward compatibility

- All 5 existing tools unchanged. `ask_user` is additive (6th tool).
- `LLMClient.complete_with_tools()` retained for non-streaming callers and tests.
- All single-shot CLI subcommands (`distill / browse / ask / resume / research / legacy-repl`) unchanged.

## [1.4.0] — 2026-05-19

### Added — paper-distiller is now a conversational agent

- **`paper-distiller-chat --vault X`** (no subcommand) now launches a **natural-language conversational agent**. The user only talks; the internal LLM decides which tools to call. No more slash commands, no more flag-juggling. Type `"帮我找几篇关于扩散模型的论文"` and the agent runs `search` → shows abstracts → asks if you want to distill any.
- **5 LLM-callable tools** exposed via OpenAI-compatible function-calling:
  - `search(topic, n=10, source="all")` — arxiv + SS + OpenAlex parallel search, returns ranked candidates with abstracts
  - `distill_by_id(ids, topic=...)` — fetch + distill specific papers into the vault
  - `show(slug, category="articles")` — read a saved vault entry back into the conversation
  - `ask(question, max_rounds=3, ...)` — multi-round QA loop
  - `research(question, duration="2h", ...)` — long-running autonomous deep-research mode
- **`LLMClient.complete_with_tools(messages, tools)`** — OpenAI-compatible function-calling against Aliyun Bailian's qwen-plus (and any other tool-calling-capable provider).
- **`AgentLoop`** — stateful conversation manager with message history, tool dispatch, per-turn tool-call safety cap (10), oversized-result truncation, on-tool-call hooks for status display.
- **Chinese system prompt** by default — tells the LLM about all 5 tools, when to use which, budget defaults, and to reply concisely in Chinese.

### Changed

- **`paper-distiller-chat --vault X`** (no subcommand) now launches the **agent loop**, not the slash-command REPL. Behavior change for users who relied on the REPL as the default.
- **Pre-v1.4 slash-command REPL** is still available as `paper-distiller-chat legacy-repl --vault X`. Same intent-router + slot-filling logic, just opt-in now.

### Internal

- **35 new tests** (16 for agent_tools, 15 for agent_loop, 4 for CLI dispatch). Total: **259** (was 224).
- `_parse_duration` extracted from `chat/cli.py` into shared `chat/_durations.py` to break a circular import between `cli.py` ↔ `agent_loop.py` ↔ `agent_tools.py`.
- `_NoDepsProcessor` subclass replaces the previous `processor.deps = []` instance-mutation hack in the two-phase distill flow.
- Agent loop's tool-call dispatcher catches all exceptions inside each wrapper and converts to `{"error": "<Type>: <msg>"}` — the LLM sees structured errors and can decide whether to retry.

### Backward compatibility

- All existing subcommands (`distill` / `browse` / `ask` / `resume` / `research`) work unchanged. Scripts and CI keep working.
- `paper-distiller` (the single-pass entry point) is unchanged.
- Old REPL accessible via `legacy-repl` subcommand.

## [1.3.0] — 2026-05-19

### Added
- **`paper-distiller-chat browse`** — search + LLM-rank, then **render N candidates with title / authors / year / abstract preview** and prompt the user to pick which to distill. Picked papers go through the full distill pipeline (PDF download + LLM distill + vault write); unpicked papers cost nothing further.
- Decouples search (cheap — single ranker LLM call ~¥0.05) from distillation (expensive — ~¥0.20/paper). Good for "I want to review before spending budget" workflows. Pick syntax: `1,3,5` / `1-5` / `1,3-5,7` / `all` / `q` to cancel.

### Internal
- 9 new tests covering pick-parser (`_parse_picks`) edge cases + argparse wiring. Total: **221** (was 212).
- No new runtime deps. Reuses existing agent framework — Phase 1 DAG stops after `CandidateRanker`; Phase 2 reuses `PaperProcessor` + `VaultWriter` + `SurveyComposer` with `processor.deps=[]` instance override (no candidate-ranker re-rank needed since user already picked).

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
