# Phase 2: PDF View (real original-paper rendering)

> Branch `feat/phase2-pdf-view`. Builds on Phase 1. Replaces the `PaperView` placeholder with a real PDF.js-backed viewer of the original arXiv PDF, with article-section → PDF-page jumping.

**Goal:** User clicks the "Paper" tab next to "Article"; the original paper PDF renders inside the workspace (page nav + zoom). Clicking a `§ N` button on the Article jumps the PDF to the right page (best-effort from existing `p. N` refs). The PDF persists in the vault so opening it next time is instant.

**Non-goals (Phase 2b later):** per-heading anchors (e.g. jump to `§ 3.2.1`'s exact paragraph), figure tooltips, footnote pop-out cards, region highlighting on jump. Phase 2a delivers the rendering + page-level jumping; Phase 2b adds the polish.

**Lean footprint:** PDF.js loaded from CDN (no new pip deps). Backend adds **one** module + **one** endpoint; cache lives at `<vault>/.pdfs/<arxiv_id>.pdf` (existing `.proof_store/` already establishes the "hidden vault subdir" convention).

---

## File Structure

- **Create** `src/paper_distiller/web/pdf_cache.py` — `get_or_download_pdf(arxiv_id, vault_path) -> Path`. Lazy fetch from `https://arxiv.org/pdf/<id>` if not cached; deterministic safe filename.
- **Create** `src/paper_distiller/web/routes/paper.py` — `GET /paper/{arxiv_id}.pdf?vault_path=...` returns the PDF as `application/pdf`.
- **Modify** `src/paper_distiller/web/server.py` — register the `paper` router.
- **Modify** `src/paper_distiller/web/static/paper-distiller.html` — add PDF.js CDN scripts (`pdf.min.js` + `pdf.worker.min.js`).
- **Modify** `src/paper_distiller/web/static/paper-distiller.jsx`:
  - Replace `PaperView` placeholder with a real PDF.js renderer (loads `/paper/{arxiv_id}.pdf`, renders pages onto canvases, supports prev/next/zoom/jump-to-page).
  - Plumb a `jumpToPage` ref/state from `App` → `PaperView`.
  - In `ArticleView`, when rendering section headings from markdown, detect `p\. ?\d+` patterns in nearby text; if present, render a tiny "↗ p.N" pill that calls `jumpToPaperPage(N)` and switches to the Paper tab.
  - Open-arxiv link in `PaperView` stays as a fallback for when PDF unavailable.
- **Create** `tests/web/test_paper_endpoint.py` — covers cache hit/miss, invalid id (path traversal), download failure (mocked httpx).
- **Modify** `pyproject.toml`? No new deps needed (httpx already in core; PDF.js is CDN).

---

## API Contract

### `GET /paper/{arxiv_id}.pdf?vault_path=...`
- **Validate `arxiv_id`** against `^[0-9]{4}\.[0-9]{4,6}(v[0-9]+)?$` (the modern arxiv ID format). Reject anything containing `/`, `\`, `..`, `%`. → 400 on invalid.
- Resolve cache path = `(<vault>/.pdfs/<arxiv_id>.pdf).resolve()`. Assert `is_relative_to(<vault>/.pdfs)` after resolve (defense in depth).
- If file exists → `FileResponse(path, media_type="application/pdf")`.
- Else: HTTP GET `https://arxiv.org/pdf/{arxiv_id}` with httpx (follow redirects, 30s timeout). On 200 → write to cache, then return. On non-200 / network error → 502 with `{"detail": "could not fetch PDF"}`.
- **Send `Cache-Control: public, max-age=86400`** on success so browser caches.
- Don't stream: PDFs are small (~1-3 MB); `FileResponse` is fine.

---

## Frontend integration (the part that requires care)

PDF.js loaded via:
```html
<script src="https://cdn.jsdelivr.net/npm/pdfjs-dist@4.7.76/build/pdf.min.mjs" type="module"></script>
```
(Use the modern ESM build; set `pdfjsLib.GlobalWorkerOptions.workerSrc` to the matching worker URL.)

`PaperView` (new, replacing the placeholder):
- Accepts props: `arxivId`, `jumpToPage` (number | null), `onPageChange` (optional).
- Internal state: `pdf` (the loaded PdfDocument), `currentPage`, `zoom`, `numPages`.
- On `arxivId` change: `pdfjsLib.getDocument(`/paper/${arxivId}.pdf?vault_path=…`).promise → setPdf`; reset page to 1.
- On `jumpToPage` change: set `currentPage` to that value (clamped).
- Render: a toolbar (prev/next/page-input/zoom −/+ /open-in-arxiv) + a `<canvas>` for the current page. Render the current page on `currentPage`/`zoom`/`pdf` change. (Single-canvas, page-at-a-time; no virtualized multi-page scroll — keep it simple for Phase 2a.)
- Empty/error states: "PDF 加载中…" / "PDF 加载失败 — arxiv.org/abs/{id} ↗".

`App`:
- New state `paperJump: number | null`.
- Pass `jumpToPage={paperJump}` into `PaperView`.
- `jumpToPaperPage(n)` helper sets `paperJump=n` + `setTab("paper")`.

`ArticleView`:
- When rendering markdown, when a heading line contains a `p. N` ref, render an inline pill `↗ p.N` after the heading, `onClick={() => jumpToPaperPage(N)}`. Pattern: `p\.\s*(\d+)` (first match only). Heuristic is fine; users can always click the toolbar to navigate.

---

## Tasks (TDD where applicable)

### T2.1 — `pdf_cache.py` module + tests
`get_or_download_pdf(arxiv_id: str, vault_path: Path) -> Path`. Validates id; ensures `<vault>/.pdfs/` exists; returns cached path if present; else downloads via httpx and writes. Raises `ValueError` on bad id, `RuntimeError` on download failure.
Tests (mock httpx with `respx` or `httpx.MockTransport`): id-validation rejects `../`, `1234`, `abc`; cache hit returns existing path; cache miss + 200 downloads; cache miss + 404 raises.
Commit `feat(web): pdf_cache module — lazy download + cache PDFs per vault`.

### T2.2 — `/paper/{arxiv_id}.pdf` endpoint + tests
`routes/paper.py`. Wraps `get_or_download_pdf`. Returns `FileResponse` with `Cache-Control`. 400 on invalid id; 502 on download error; 200 + `application/pdf` on success. Tests: happy (cache pre-seeded with a tiny dummy PDF byte string), 400 (bad id), 502 (mock download fail).
Commit `feat(web): /paper/{arxiv_id}.pdf endpoint`.

### T2.3 — HTML: add PDF.js CDN
Modify `paper-distiller.html`: add `<script src="…pdf.min.mjs" type="module">` + worker config. Make sure existing static assets still load.
Commit `feat(web): load PDF.js from CDN in index html`.

### T2.4 — JSX: real PaperView with PDF.js
Replace the placeholder `PaperView`. State machine: loading → ready → error. Renders one page at a time on canvas. Toolbar: ‹ N/M › | zoom − % + | ↗ arxiv.
Manual smoke (no automated test for the JSX). Commit `feat(web): PaperView renders real PDF via PDF.js`.

### T2.5 — JSX: section-ref → page jumping
In `App`, add `paperJump` state + `jumpToPaperPage(n)`. Pass to `PaperView`. In `ArticleView` markdown renderer, after each heading, scan the following paragraph for `p\.\s*(\d+)`; if found, append an inline pill button `↗ p.N` that triggers `jumpToPaperPage(N)`.
Manual smoke. Commit `feat(web): jump from article section to PDF page (p. N refs)`.

### T2.6 — full pytest + ruff
Should remain green; `tests/web/` grows by ~6-10 tests. Commit any necessary cleanups.

---

## Acceptance (I'll do the smoke after the implementer)

1. `python -m pytest -q` stays green (was 641 + new tests).
2. `paper-distiller-web --vault G:/pd-demo-vault --port 8765` boots; opening `http://localhost:8765/paper/2110.05948.pdf?vault_path=G:/pd-demo-vault` downloads + serves the real DDGM PDF (~1MB) on first hit; instant on second.
3. In the browser UI, open the DDGM article → click the "Paper" tab → real PDF.js viewer renders page 1; prev/next/zoom work; jump-to-page works.
4. When the article body has a `p. 3` ref, clicking the `↗ p.3` pill switches to the Paper tab and lands on page 3.
5. `<vault>/.pdfs/2110.05948.pdf` exists after first fetch.

---

## NOT in this plan

- Per-heading anchors (`§ 3.2.1` → exact paragraph inside page) — Phase 2b.
- Region highlighting on jump (the "flash" effect from the designer's mock) — Phase 2b.
- Figure / footnote pop-out cards — Phase 2b.
- Pre-cache PDFs at distill time (instead of lazy on first view) — optimization, Phase 2c.

---

## Notes for implementer

- **Don't add new pip deps.** `httpx` is already core; PDF.js is CDN.
- **Don't change anything under `paper_distiller/chat/`** (the CLI must keep working).
- **Mock the arxiv download in tests** (`httpx.MockTransport` is the cleanest). No real network calls.
- **The PDF cache is per-vault** at `<vault>/.pdfs/`. Add it to `.gitignore`-style if needed (it's inside the user's vault, not the repo; the `.proof_store/` precedent applies).
- **PDF.js worker** must use a matching version's worker file (`pdf.worker.min.mjs`). Use the same CDN version for both.
- **Path traversal**: arxiv ID is validated by regex AND the resolved cache path is checked to be under `.pdfs/`.
- **arxiv URL pattern**: `https://arxiv.org/pdf/{id}` (no `.pdf` suffix needed; arxiv handles it). Follow redirects.
- If PDF.js's ESM module loading via CDN is flaky (it sometimes is), fall back to the UMD build `pdf.min.js` + global `pdfjsLib`. Either works.
