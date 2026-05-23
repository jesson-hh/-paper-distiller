"""Vault read endpoints.

GET /vault/stats
GET /vault/recent
GET /vault/article/{category}/{slug}
GET /vault/articles
GET /vault/graph/{paper_arxiv_id}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/vault")

# ── Frontmatter helpers ──────────────────────────────────────────────────────

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_LIST_RE = re.compile(r"\[([^\]]*)\]")


def _parse_fm_value(raw: str):
    """Parse a single YAML-lite frontmatter value (list, bool, int, str)."""
    raw = raw.strip()
    if not raw:
        return ""
    m = _LIST_RE.match(raw)
    if m:
        inner = m.group(1).strip()
        if not inner:
            return []
        items = [s.strip().strip('"').strip("'") for s in inner.split(",")]
        return [x for x in items if x]
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body. Returns (meta, body)."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        meta[key.strip()] = _parse_fm_value(val)
    body = text[m.end():]
    return meta, body


def _arxiv_id_from_meta(meta: dict) -> str:
    """Return arxiv id from frontmatter.

    Tries ``arxiv_id`` key first; falls back to the first entry in ``refs``
    that starts with ``"arxiv:"`` (strips the prefix).
    """
    direct = meta.get("arxiv_id", "")
    if direct:
        return str(direct)
    refs = meta.get("refs", [])
    if isinstance(refs, str):
        refs = [r.strip() for r in refs.split(",") if r.strip()]
    if isinstance(refs, list):
        for ref in refs:
            s = str(ref).strip()
            if s.startswith("arxiv:"):
                return s[len("arxiv:"):]
    return ""


def _vault_path_from(request: Request, vault_path: str) -> str:
    return vault_path or getattr(request.app.state, "vault_path", "")


def _require_vault(vault_path: str) -> Path:
    """Resolve vault path; raise 400 if blank, 400 if not a directory."""
    if not vault_path:
        raise HTTPException(status_code=400, detail="vault_path is required")
    p = Path(vault_path)
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"vault not found: {vault_path}")
    return p


# ── /vault/stats ─────────────────────────────────────────────────────────────


@router.get("/stats")
async def vault_stats(request: Request, vault_path: str = Query(default="")):
    """Return aggregate vault statistics."""
    vp = _vault_path_from(request, vault_path)
    root = _require_vault(vp)

    articles_dir = root / "articles"
    surveys_dir = root / "surveys"

    articles = len(list(articles_dir.glob("*.md"))) if articles_dir.is_dir() else 0
    surveys = len(list(surveys_dir.glob("*.md"))) if surveys_dir.is_dir() else 0

    # Proof store stats
    proof_nodes = 0
    proof_edges = 0
    techniques = 0
    papers = 0
    db_path = root / ".proof_store" / "proofs.db"
    if db_path.exists():
        try:
            import sqlite3  # noqa: PLC0415
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                proof_nodes = (conn.execute("SELECT COUNT(*) FROM nodes").fetchone() or [0])[0]
                proof_edges = (conn.execute("SELECT COUNT(*) FROM edges").fetchone() or [0])[0]
                techniques = (conn.execute("SELECT COUNT(*) FROM techniques").fetchone() or [0])[0]
                papers = (conn.execute(
                    "SELECT COUNT(DISTINCT paper_arxiv_id) FROM nodes"
                ).fetchone() or [0])[0]
            finally:
                conn.close()
        except Exception:
            pass

    return {
        "articles": articles,
        "surveys": surveys,
        "proof_nodes": proof_nodes,
        "proof_edges": proof_edges,
        "techniques": techniques,
        "papers": papers,
    }


# ── /vault/recent ─────────────────────────────────────────────────────────────


@router.get("/recent")
async def vault_recent(
    request: Request,
    vault_path: str = Query(default=""),
    limit: int = Query(default=10, ge=1, le=100),
):
    """Return recently updated articles + surveys, sorted by updated desc."""
    vp = _vault_path_from(request, vault_path)
    root = _require_vault(vp)

    items = []
    for cat in ("articles", "surveys"):
        cat_dir = root / cat
        if not cat_dir.is_dir():
            continue
        for md in cat_dir.glob("*.md"):
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
                meta, _ = _parse_frontmatter(text)
                title = meta.get("title", md.stem)
                arxiv_id = _arxiv_id_from_meta(meta)
                updated = meta.get("updated") or meta.get("created") or ""
                slug = md.stem
                items.append({
                    "slug": slug,
                    "title": str(title),
                    "category": cat,
                    "arxiv_id": arxiv_id,
                    "updated": str(updated),
                })
            except Exception:
                continue

    items.sort(key=lambda x: x["updated"], reverse=True)
    return {"recent": items[:limit]}


# ── /vault/article/{category}/{slug} ─────────────────────────────────────────

_VALID_CATS = {"articles", "surveys", "techniques", "directions", "open-problems", "authors"}


@router.get("/article/{category}/{slug}")
async def vault_article(
    category: str,
    slug: str,
    request: Request,
    vault_path: str = Query(default=""),
):
    """Read a single vault article by category + slug."""
    if category not in _VALID_CATS:
        raise HTTPException(status_code=400, detail=f"unknown category: {category}")

    vp = _vault_path_from(request, vault_path)
    root = _require_vault(vp)

    md_path = (root / category / f"{slug}.md").resolve()
    if not md_path.is_relative_to(root.resolve()):
        raise HTTPException(status_code=400, detail="invalid slug")
    if not md_path.exists():
        raise HTTPException(status_code=404, detail=f"{category}/{slug} not found")

    text = md_path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_frontmatter(text)

    # Proof stats from proof store
    proof_stats = {"nodes": 0, "suspicious": 0, "gap": 0}
    arxiv_id = _arxiv_id_from_meta(meta)
    db_path = root / ".proof_store" / "proofs.db"
    if arxiv_id and db_path.exists():
        try:
            import sqlite3  # noqa: PLC0415
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                proof_stats["nodes"] = (conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE paper_arxiv_id=?", (arxiv_id,)
                ).fetchone() or [0])[0]
                proof_stats["suspicious"] = (conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE paper_arxiv_id=? AND status='suspicious'",
                    (arxiv_id,),
                ).fetchone() or [0])[0]
                proof_stats["gap"] = (conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE paper_arxiv_id=? AND status='gap'",
                    (arxiv_id,),
                ).fetchone() or [0])[0]
            finally:
                conn.close()
        except Exception:
            pass

    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    refs = meta.get("refs", [])
    if isinstance(refs, str):
        refs = [r.strip() for r in refs.split(",") if r.strip()]

    return {
        "slug": slug,
        "category": category,
        "title": str(meta.get("title", slug)),
        "tags": tags if isinstance(tags, list) else [],
        "refs": refs if isinstance(refs, list) else [],
        "arxiv_id": arxiv_id,
        "body": body,
        "frontmatter": {k: v for k, v in meta.items()},
        "created": str(meta.get("created", "")),
        "updated": str(meta.get("updated", "")),
        "proof_stats": proof_stats,
    }


# ── /vault/articles ──────────────────────────────────────────────────────────


@router.get("/articles")
async def vault_articles(
    request: Request,
    vault_path: str = Query(default=""),
    category: str = Query(default="articles"),
    q: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List and filter vault articles. Simple in-memory scan."""
    if category not in _VALID_CATS:
        raise HTTPException(status_code=400, detail=f"unknown category: {category}")

    vp = _vault_path_from(request, vault_path)
    root = _require_vault(vp)

    cat_dir = root / category
    if not cat_dir.is_dir():
        return {"total": 0, "items": []}

    items = []
    q_lower = q.lower() if q else None
    tag_lower = tag.lower() if tag else None

    for md in sorted(cat_dir.glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
            meta, _ = _parse_frontmatter(text)
            title = str(meta.get("title", md.stem))
            tags_raw = meta.get("tags", [])
            if isinstance(tags_raw, str):
                tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]
            tags_list = tags_raw if isinstance(tags_raw, list) else []
            updated = str(meta.get("updated") or meta.get("created") or "")

            if q_lower and q_lower not in title.lower() and q_lower not in " ".join(str(t) for t in tags_list).lower():
                continue
            if tag_lower and not any(tag_lower in str(t).lower() for t in tags_list):
                continue

            items.append({
                "slug": md.stem,
                "title": title,
                "tags": [str(t) for t in tags_list],
                "updated": updated,
            })
        except Exception:
            continue

    total = len(items)
    items.sort(key=lambda x: x["updated"], reverse=True)
    return {"total": total, "items": items[offset : offset + limit]}


# ── /vault/graph/{paper_arxiv_id} ────────────────────────────────────────────

_KIND_ROW: dict[str, int] = {
    "assumption": 0,
    "definition": 0,
    "axiom": 0,
    "lemma": 1,
    "step": 1,
    "claim": 1,
    "theorem": 2,
    "proposition": 2,
    "corollary": 2,
}

_COL_STRIDE = 160
_ROW_STRIDE = 120
_X_OFFSET = 30
_Y_OFFSET = 30


def _layout_nodes(nodes: list[dict]) -> list[dict]:
    """Assign x/y positions by kind-row (deterministic grid, no layout lib)."""
    rows: dict[int, list] = {}
    for n in nodes:
        row = _KIND_ROW.get(n.get("kind", "step"), 1)
        rows.setdefault(row, []).append(n)

    result = []
    for row_idx in sorted(rows):
        cols = rows[row_idx]
        for col_idx, n in enumerate(cols):
            n = dict(n)
            n["x"] = _X_OFFSET + col_idx * _COL_STRIDE
            n["y"] = _Y_OFFSET + row_idx * _ROW_STRIDE
            result.append(n)
    return result


@router.get("/graph/{paper_arxiv_id}")
async def vault_graph(
    paper_arxiv_id: str,
    request: Request,
    vault_path: str = Query(default=""),
):
    """Return full proof graph for one paper, ready for SVG renderer."""
    # Defense-in-depth: reject IDs that look like path components
    if ".." in paper_arxiv_id or "/" in paper_arxiv_id or "\\" in paper_arxiv_id:
        raise HTTPException(status_code=400, detail="invalid paper_arxiv_id")

    vp = _vault_path_from(request, vault_path)
    root = _require_vault(vp)

    db_path = root / ".proof_store" / "proofs.db"
    if not db_path.exists():
        return {"nodes": [], "edges": [], "stats": {"by_kind": {}, "by_status": {}, "cross_paper_edges": 0}}

    try:
        import sqlite3  # noqa: PLC0415
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            node_rows = conn.execute(
                "SELECT n.id, n.kind, n.label, n.text, n.source_quote, n.loc, "
                "n.status, n.confidence "
                "FROM nodes n WHERE n.paper_arxiv_id=? ORDER BY n.id",
                (paper_arxiv_id,),
            ).fetchall()

            if not node_rows:
                return {"nodes": [], "edges": [], "stats": {"by_kind": {}, "by_status": {}, "cross_paper_edges": 0}}

            node_ids = [r["id"] for r in node_rows]

            # Techniques per node
            tech_map: dict[int, list] = {nid: [] for nid in node_ids}
            for nid in node_ids:
                rows_t = conn.execute(
                    "SELECT technique FROM node_techniques WHERE node_id=?", (nid,)
                ).fetchall()
                tech_map[nid] = [r["technique"] for r in rows_t]

            nodes_out = []
            by_kind: dict = {}
            by_status: dict = {}
            for r in node_rows:
                nid = r["id"]
                kind = r["kind"] or "step"
                status = r["status"] or "extracted"
                by_kind[kind] = by_kind.get(kind, 0) + 1
                by_status[status] = by_status.get(status, 0) + 1
                nodes_out.append({
                    "id": nid,
                    "kind": kind,
                    "label": r["label"] or "",
                    "text": r["text"] or "",
                    "source_quote": r["source_quote"] or "",
                    "loc": r["loc"] or "",
                    "status": status,
                    "confidence": r["confidence"],
                    "techniques": tech_map.get(nid, []),
                })

            # Lay out nodes by kind
            nodes_out = _layout_nodes(nodes_out)

            # Edges involving any node in this paper
            id_placeholders = ",".join("?" * len(node_ids))
            edge_rows = conn.execute(
                f"SELECT src_id, dst_id, rel, cross_paper, justification "
                f"FROM edges WHERE src_id IN ({id_placeholders}) "
                f"OR dst_id IN ({id_placeholders})",
                node_ids + node_ids,
            ).fetchall()

            edges_out = []
            cross_paper_count = 0
            for e in edge_rows:
                cp = int(e["cross_paper"] or 0)
                cross_paper_count += cp
                edges_out.append({
                    "src_id": e["src_id"],
                    "dst_id": e["dst_id"],
                    "rel": e["rel"],
                    "cross_paper": cp,
                    "justification": e["justification"],
                })

            return {
                "nodes": nodes_out,
                "edges": edges_out,
                "stats": {
                    "by_kind": by_kind,
                    "by_status": by_status,
                    "cross_paper_edges": cross_paper_count,
                },
            }
        finally:
            conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
