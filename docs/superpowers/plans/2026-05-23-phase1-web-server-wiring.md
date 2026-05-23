# Phase 1: Web Server + JSX Wiring

> Branch `feat/web-frontend-wiring`. Wires the designer's React prototype (at `D:/download/paper-design/`) to the Python backend. **Phase 1a scope only** — deferrals listed at the end.

**Goal:** A user runs `paper-distiller-web --vault X`, opens `http://localhost:8765`, sees the designer's UI, types natural language, and the real LLM agent does real search/distill/review/find_proof through SSE. Real data in: Welcome stats, Recent list, Article view, Graph view.

**Architecture:** New optional subpackage `paper_distiller.web` exposing a FastAPI app that (a) serves the static frontend (HTML/CSS/JSX/JS copied in) and (b) exposes a small REST + SSE API wrapping existing tools and vault store. **Conversation state is client-side** (frontend keeps full history, sends it with each turn) → stateless server. LLM env reused from `.env` exactly like the CLI.

**New dependency footprint:** lean — `[web]` extra only (`fastapi`, `uvicorn[standard]`, `sse-starlette`). Core install stays unchanged.

---

## File Structure
- **Create** `src/paper_distiller/web/__init__.py` — `__version__`, public exports.
- **Create** `src/paper_distiller/web/server.py` — `create_app(vault_path)` returns a FastAPI app; mounts `/static` for the frontend; CORS off (same-origin).
- **Create** `src/paper_distiller/web/cli.py` — `main()` parses `--vault`, `--host`, `--port` (default 127.0.0.1:8765); calls `uvicorn.run(create_app(...))`.
- **Create** `src/paper_distiller/web/routes/__init__.py`.
- **Create** `src/paper_distiller/web/routes/chat.py` — `POST /chat/stream` (SSE).
- **Create** `src/paper_distiller/web/routes/vault.py` — `GET /vault/stats`, `/vault/recent`, `/vault/article/{category}/{slug}`, `/vault/articles`, `/vault/graph/{paper_arxiv_id}`.
- **Create** `src/paper_distiller/web/routes/config.py` — `GET /config` (read-only minimum: model/base_url/permission_mode/graph_depth).
- **Create** `src/paper_distiller/web/agent_stream.py` — the async generator that drives the LLM agent and emits SSE events (re-uses `TOOL_SCHEMAS`/`execute_tool` from `chat/agent_tools.py`).
- **Create** `src/paper_distiller/web/static/` — copy `paper-distiller.html`, `paper-distiller.jsx`, `paper-distiller.css`, `deck-stage.js` from `D:/download/paper-design/`. **Modify the JSX in-place** to fetch from the API.
- **Modify** `pyproject.toml` — add `[web]` extra; add `paper-distiller-web = "paper_distiller.web.cli:main"` to `[project.scripts]`.

---

## API Contract

### `POST /chat/stream`  (Server-Sent Events)
Drives the agent loop. Stateless: client sends full history.

**Request body (JSON):**
```json
{
  "message": "找几篇 Transformer 注意力机制的核心论文,蒸馏 3 篇",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
  ],
  "vault_path": "/path/to/vault"
}
```

**Response:** `text/event-stream`. Each line `data: {JSON}\n\n`. Event types:
```
{"type": "text",            "delta": "..."}                            # streamed assistant text
{"type": "tool_call_start", "id": "tc_abc", "name": "search", "args": {...}}
{"type": "tool_call_done",  "id": "tc_abc", "result": {...}}           # full tool result JSON
{"type": "cost",            "tokens_in": N, "tokens_out": N, "cny": 0.12}
{"type": "done",            "history": [...]}                          # final history (for client to keep)
{"type": "error",           "message": "..."}
```

**Loop semantics (server side):**
1. Append `{role: "user", content: message}` to history.
2. Stream LLM with `complete_with_tools_stream(history, TOOL_SCHEMAS)`; emit `text` deltas; accumulate `tool_calls`.
3. If tool_calls: append assistant message with tool_calls; for each call, emit `tool_call_start` → run `execute_tool(name, args, vault_path=vault_path)` (in a thread, wrapped) → emit `tool_call_done` → append tool result message. Loop back to step 2.
4. If no tool_calls: append final assistant message; emit `cost` then `done`. Close stream.
5. Per-turn safety cap: max 10 tool calls (matches existing `AgentLoop`).
6. Any unhandled exception → emit `error` then close.

### `GET /vault/stats?vault_path=...`
```json
{"articles": 242, "surveys": 14, "proof_nodes": 3617, "proof_edges": 1284, "techniques": 76, "papers": 49}
```
Compute by: scanning `<vault>/articles/*.md`, `<vault>/surveys/*.md`, querying `<vault>/.proof_store/proofs.db` (counts on `nodes` / `edges` / `techniques` / distinct `paper_arxiv_id`).

### `GET /vault/recent?vault_path=...&limit=10`
```json
{"recent": [{"slug": "...", "title": "...", "category": "articles", "arxiv_id": "...", "updated": "ISO-8601"}, ...]}
```
Scan `articles/` + `surveys/` `.md` files, parse YAML frontmatter (`title`, `arxiv_id`, `updated`/`created`), sort by updated desc, take top N.

### `GET /vault/article/{category}/{slug}?vault_path=...`
```json
{
  "slug": "...", "category": "...", "title": "...",
  "tags": [...], "refs": [...], "arxiv_id": "...",
  "body": "...markdown body...",
  "frontmatter": {...all yaml...},
  "created": "...", "updated": "...",
  "proof_stats": {"nodes": 23, "suspicious": 3, "gap": 0}   // join the proof store
}
```
Read the `.md`, split frontmatter from body, query the proof store for the paper's node counts by status.

### `GET /vault/articles?vault_path=...&category=articles&q=...&tag=...&limit=50&offset=0`
For the (future) vault browser:
```json
{"total": 242, "items": [{"slug":"...","title":"...","tags":[...],"updated":"..."}, ...]}
```
Simple in-memory scan + filter is fine for Phase 1.

### `GET /vault/graph/{paper_arxiv_id}?vault_path=...`
Returns the full proof graph for one paper, ready for the SVG renderer:
```json
{
  "nodes": [
    {"id": 12, "kind": "theorem", "label": "Theorem 1", "text": "...", "source_quote": "...",
     "loc": "{\"sec\":\"3.2\"}", "status": "ok", "confidence": null, "techniques": ["Bernstein"]}
  ],
  "edges": [{"src_id": 12, "dst_id": 7, "rel": "depends_on", "cross_paper": 0, "justification": null}],
  "stats": {"by_kind": {...}, "by_status": {...}, "cross_paper_edges": N}
}
```

### `GET /config?vault_path=...`
Read-only for Phase 1:
```json
{"model": "qwen3.5-plus", "base_url": "...", "permission_mode": "default",
 "graph_depth": "off", "plan_threshold_cny": 10.0, "vault_path": "...", "version": "1.12.0"}
```

### Static
`GET /` → `paper-distiller.html`; `GET /paper-distiller.{css,jsx}`, `GET /deck-stage.js` → static files. Serve via `StaticFiles` mounted at `/`.

---

## JSX Modification Map

The implementer **modifies the JSX in-place** (after copying it into `web/static/`). What changes:

| In current JSX | Replace with |
|---|---|
| Hardcoded `ARTICLE` constant | `useEffect → fetch('/vault/article/articles/' + currentSlug)`; render the API response. Frontmatter `title/authors/venue/tags/arxiv_id` from API; body is markdown — render via a minimal markdown renderer (split on `## ` for section headings is enough for Phase 1; advanced rendering later) and run RichText/Equation on inline `$...$` segments. |
| Hardcoded `SEARCH_RESULTS` | comes from a `tool_call_done` SSE event for `name=="search"` — store on the corresponding `ToolCard` message. |
| Hardcoded `RECENT_ARTICLES` | `fetch('/vault/recent')` in WelcomeView's mount. |
| Vault stats `242 / 14 / 3.6k` in WelcomeView | `fetch('/vault/stats')`. |
| `sendText` scripted switch (keyword matching) | **DELETE** the scripted matching. New `sendText`: POST `/chat/stream` with `{message, history, vault_path}`, open EventSource (or fetch+ReadableStream), dispatch events into messages state. |
| `pickPaper(paper)` (direct distill call) | Synthesize a chat message: `sendText("蒸馏这篇:" + paper.arxiv + "(关于" + lastSearchTopic + ")")`. The agent will call `distill_by_id`. No new tool endpoint needed. |
| `GRAPH_NODES` / `GRAPH_EDGES` hardcoded | `fetch('/vault/graph/' + currentArxivId)`; lay out nodes — Phase 1: simple force-directed via a tiny client-side algorithm, OR a deterministic grid by kind (theorem at top, lemmas next row, steps below, definitions/assumptions sidebar). **Don't add a graph layout dep**; a fixed grid is fine for Phase 1 (improve later). |
| `PAPER_FOOTNOTES` + `PaperView` | **Show a placeholder**: "PDF 视图正在 Phase 2 中,先看 arXiv 原文 ↗" with link to `https://arxiv.org/abs/{arxiv_id}`. Keep the toolbar shell; don't try to render content. |
| `DashboardView` | **Show a placeholder**: "深度研究仪表盘 Phase 1b 接入"; keep the demo time-driven counters as visual placeholder, but mark the panel as MOCK. Real wiring (poll `/research/{sid}/state`) is Phase 1b. |
| `VerificationPanel` | **Keep as-is** (uses local state, no backend). Real wiring is Phase 4. |
| `cost` chip | Driven by `cost` SSE events accumulated client-side. |

**One vault path for the whole session**: read from a `<meta name="vault-path" content="...">` injected by `server.py` when serving `index.html`, OR a `/config` fetch on mount. Use the same path for every API call.

---

## Tasks (TDD where applicable)

### T1.1 — Scaffold `paper_distiller.web` package + `[web]` extra + console script
- Create `web/__init__.py`, `web/server.py` with a minimal `create_app(vault_path) -> FastAPI` that serves a hello route. Static mount at `/`. Create `web/cli.py` with argparse + uvicorn.run. Add `paper-distiller-web = "paper_distiller.web.cli:main"` to `[project.scripts]`. Add `[project.optional-dependencies]` `web = ["fastapi>=0.110", "uvicorn[standard]>=0.27", "sse-starlette>=2.0"]`.
- Test (`tests/web/test_server_starts.py`): import `create_app`, use `TestClient` from FastAPI, `GET /config?vault_path=tmp` returns 200 with the expected keys.
- Commit `feat(web): scaffold FastAPI app + [web] extra + paper-distiller-web cli`.

### T1.2 — Vault read endpoints
Implement `/vault/stats`, `/vault/recent`, `/vault/article/{category}/{slug}`, `/vault/articles`, `/vault/graph/{paper_arxiv_id}`. Tests use a `tmp_path` vault seeded with a couple `.md` files + a `ProofStore` with a few nodes. Each endpoint: test happy path + missing-file → 404 + bad vault → 400.
Commit `feat(web): vault stats/recent/article/articles/graph endpoints`.

### T1.3 — Chat stream endpoint (SSE) + agent driver
- `web/agent_stream.py`: `async def agent_event_stream(message, history, vault_path, llm) -> AsyncIterator[dict]` that yields SSE event dicts per the contract above. Uses `LLMClient.complete_with_tools_stream` (existing) for text/tool_call streaming; runs `execute_tool` in `asyncio.to_thread`.
- `web/routes/chat.py`: `POST /chat/stream` returns `EventSourceResponse` from `sse-starlette`.
- Test: with a **stub LLM** that yields one text delta + one tool_call (e.g. `search`), and `execute_tool` monkeypatched to return a canned result, drive the stream and assert event sequence (`text → tool_call_start → tool_call_done → done`).
- Commit `feat(web): SSE chat stream + agent driver`.

### T1.4 — Copy frontend into `web/static/` (verbatim first)
Just copy the 4 designer files from `D:/download/paper-design/` to `src/paper_distiller/web/static/`. Verify server can serve them. (No JSX modification yet.) Commit `feat(web): include designer's static frontend`.

### T1.5 — Wire JSX to backend (the big in-place edit)
Modify `web/static/paper-distiller.jsx` per the "JSX Modification Map" above:
- Add a `vaultPath` constant resolved from `<meta>` or `/config`.
- Add a `fetchVaultStats()` / `fetchRecent()` etc.
- Replace `sendText` body with SSE-driven flow.
- Replace `ArticleView`'s data source (the rendered article body should accept markdown; minimal renderer: split on `## ` headings, treat content as paragraphs with RichText `$...$`).
- Replace `GraphView`'s data source; lay out nodes by kind (deterministic, no layout lib).
- `PaperView` → placeholder.
- `DashboardView` → placeholder (keep visual demo, mark MOCK).
- Test:** the JSX has no automated tests. Manual smoke after this task.
Commit `feat(web): wire designer JSX to real backend (chat stream + vault APIs)`.

### T1.6 — Doctor / readiness
A `GET /healthz` endpoint that checks: vault exists, .env loaded (PD_API_KEY/BASE_URL/MODEL set), proof_store reachable. Returns `{ok: true|false, checks: [...]}`. Used by the frontend's startup banner. Test happy + sad paths. Commit `feat(web): /healthz readiness check`.

### T1.7 — Full pytest + ruff
Run `python -m pytest -q` and `python -m ruff check src/paper_distiller/web/`. Both green.

---

## Acceptance (after the implementer is done — I do the manual smoke)

1. `pip install -e ".[web]"` works (or the existing editable install picks up the new extras).
2. `paper-distiller-web --vault G:/pd-demo-vault --port 8765` starts; opening `http://localhost:8765` shows the designer's UI.
3. Welcome shows REAL counts (from `pd-demo-vault`: 2 articles · 1 survey · 334 proof nodes · ...).
4. Recent list shows the 2 distilled demo papers.
5. Typing "找几篇 Transformer attention 的论文,挑两篇蒸馏" — the agent really calls search → returns real arxiv candidates → ToolCard renders them. Clicking a candidate triggers a real distill_by_id call (synthesized chat message). Article view populates from the real `.md`.
6. Typing "对 DDGM 跑一次审查" — review_proof runs on `2110.05948` (already in vault); Graph view shows real nodes; flagged statuses are color-coded.
7. Cost chip increments with real `cost` SSE events.

---

## NOT in this plan (deferrals — explicit)

- **PaperView (Phase 2)** — keep-PDF + PDF.js + §↔page-anchor mapping. Big new feature.
- **DashboardView real wiring (Phase 1b)** — async research job + polling `/research/{sid}/state`. Trickier (long-running session). Placeholder UI for now.
- **VerificationPanel real backing (Phase 4)** — multi-step LLM verification + persistence. The component stays as a self-contained demo for now.
- **Settings page / permission modes / ask_user UI (Phase 3)** — needs designer补稿 before implementing.
- **Cross-paper graph visualization** — Phase 1 graph view is per-paper; cross-paper edges visible but not a "whole vault" view.
- **Markdown body of articles** — Phase 1 uses a minimal `## ` splitter. Full markdown rendering (lists, tables, code blocks, MathJax across paragraphs) can come later — articles already have an `.html` sibling we could render via iframe as a fallback.

---

## Notes for implementer
- Do NOT run a real LLM in tests. Stub `LLMClient` (a small class with `complete_with_tools_stream` that yields canned chunks) for the chat-stream test.
- Stateless server: client sends history each turn. Don't build session storage in Phase 1.
- The existing `LLMClient.complete_with_tools_stream` returns an iterator of `StreamChunk` (look at `llm/openai_compatible.py`); your SSE driver wraps it.
- `execute_tool` is synchronous and can block for minutes (distill, review). Wrap in `asyncio.to_thread` to not block the event loop. The user will see a "running" ToolCard for the duration — that's correct behavior for Phase 1.
- Don't change anything in `paper_distiller/chat/` (the CLI agent must keep working byte-for-byte). The web layer is purely additive.
- For per-turn safety cap: 10 tool calls max (matches `AgentLoop`).
- If `sse-starlette` ends up flaky, fall back to manual `StreamingResponse` with `text/event-stream` content type.
- If something in the JSX rewrite is unclear, **STOP and report** — do NOT silently change product behavior or strip designer styling.
