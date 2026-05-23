"""T1.2 — Vault endpoint tests.

Tests use tmp_path with a seeded vault (a couple .md files + a proof SQLite DB).
Each endpoint: happy path + missing → 404 + bad vault → 400.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paper_distiller.web.server import create_app


# ── Fixtures ────────────────────────────────────────────────────────────────

_FM_TEMPLATE = """\
---
title: "{title}"
arxiv_id: "{arxiv_id}"
tags: [{tags}]
refs: []
created: "2024-01-01T00:00:00"
updated: "{updated}"
---

{body}
"""


def _write_article(vault: Path, cat: str, slug: str, title: str, arxiv_id: str, updated: str, body: str = "Test body.", tags: str = "") -> Path:
    (vault / cat).mkdir(parents=True, exist_ok=True)
    md = vault / cat / f"{slug}.md"
    md.write_text(_FM_TEMPLATE.format(
        title=title,
        arxiv_id=arxiv_id,
        tags=tags,
        updated=updated,
        body=body,
    ), encoding="utf-8")
    return md


def _create_proof_db(vault: Path, arxiv_id: str = "2101.00001") -> Path:
    db_dir = vault / ".proof_store"
    db_dir.mkdir(exist_ok=True)
    db = db_dir / "proofs.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_arxiv_id TEXT NOT NULL,
            paper_slug TEXT,
            kind TEXT NOT NULL,
            label TEXT,
            text TEXT NOT NULL,
            source_quote TEXT,
            loc TEXT,
            status TEXT NOT NULL DEFAULT 'extracted',
            confidence REAL,
            parent_id INTEGER,
            ord INTEGER,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_id INTEGER NOT NULL,
            dst_id INTEGER NOT NULL,
            rel TEXT NOT NULL,
            justification TEXT,
            cross_paper INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(src_id, dst_id, rel)
        );
        CREATE TABLE IF NOT EXISTS techniques (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL DEFAULT '',
            first_seen_arxiv_id TEXT
        );
        CREATE TABLE IF NOT EXISTS node_techniques (
            node_id INTEGER NOT NULL,
            technique TEXT NOT NULL,
            PRIMARY KEY (node_id, technique)
        );
        CREATE TABLE IF NOT EXISTS theorems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_arxiv_id TEXT NOT NULL,
            paper_slug TEXT,
            name TEXT NOT NULL,
            statement TEXT NOT NULL,
            proof_sketch TEXT,
            techniques_used TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    now = "2024-01-01T00:00:00"
    # Insert 3 nodes for the arxiv_id
    conn.execute(
        "INSERT INTO nodes(paper_arxiv_id, kind, text, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (arxiv_id, "theorem", "Theorem A", "ok", now),
    )
    node1_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO nodes(paper_arxiv_id, kind, text, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (arxiv_id, "lemma", "Lemma B", "suspicious", now),
    )
    node2_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO nodes(paper_arxiv_id, kind, text, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (arxiv_id, "step", "Step C", "gap", now),
    )
    node3_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Edge: node1 → node2
    conn.execute(
        "INSERT OR IGNORE INTO edges(src_id, dst_id, rel, cross_paper, created_at) VALUES (?, ?, ?, ?, ?)",
        (node1_id, node2_id, "depends_on", 0, now),
    )
    # Technique
    conn.execute("INSERT OR IGNORE INTO techniques(name) VALUES (?)", ("Cauchy-Schwarz",))
    conn.execute("INSERT OR IGNORE INTO node_techniques(node_id, technique) VALUES (?, ?)", (node1_id, "Cauchy-Schwarz"))

    conn.commit()
    conn.close()
    return db


@pytest.fixture
def seeded_vault(tmp_path):
    """Vault with 2 articles + 1 survey + proof store."""
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_article(vault, "articles", "paper-alpha", "Paper Alpha", "2101.00001", "2024-03-01T00:00:00", tags='"diffusion", "score"')
    _write_article(vault, "articles", "paper-beta", "Paper Beta", "2101.00002", "2024-02-01T00:00:00", tags='"attention"')
    _write_article(vault, "surveys", "survey-one", "Survey One", "2101.00099", "2024-01-01T00:00:00")
    _create_proof_db(vault, "2101.00001")
    return vault


@pytest.fixture
def client(seeded_vault):
    app = create_app(str(seeded_vault))
    return TestClient(app, raise_server_exceptions=True)


# ── /vault/stats ──────────────────────────────────────────────────────────────

class TestVaultStats:
    def test_happy_path(self, client):
        r = client.get("/vault/stats")
        assert r.status_code == 200
        d = r.json()
        assert d["articles"] == 2
        assert d["surveys"] == 1
        assert d["proof_nodes"] == 3
        assert d["proof_edges"] == 1
        assert d["techniques"] == 1

    def test_bad_vault_400(self, client):
        r = client.get("/vault/stats?vault_path=/nonexistent/vault/xyz")
        assert r.status_code == 400

    def test_empty_vault_path_uses_app_state(self, client):
        """Empty vault_path should fall back to app.state.vault_path."""
        r = client.get("/vault/stats")
        assert r.status_code == 200

    def test_explicit_vault_path(self, seeded_vault):
        app = create_app("/tmp/unused")
        tc = TestClient(app)
        r = tc.get(f"/vault/stats?vault_path={seeded_vault}")
        assert r.status_code == 200
        assert r.json()["articles"] == 2


# ── /vault/recent ─────────────────────────────────────────────────────────────

class TestVaultRecent:
    def test_happy_path(self, client):
        r = client.get("/vault/recent")
        assert r.status_code == 200
        data = r.json()
        assert "recent" in data
        # 3 total (2 articles + 1 survey)
        assert len(data["recent"]) == 3

    def test_sorted_by_updated_desc(self, client):
        r = client.get("/vault/recent")
        items = r.json()["recent"]
        dates = [i["updated"] for i in items]
        assert dates == sorted(dates, reverse=True)

    def test_limit_respected(self, client):
        r = client.get("/vault/recent?limit=1")
        assert r.status_code == 200
        assert len(r.json()["recent"]) == 1

    def test_bad_vault_400(self, client):
        r = client.get("/vault/recent?vault_path=/nonexistent/xyz")
        assert r.status_code == 400

    def test_items_have_required_fields(self, client):
        r = client.get("/vault/recent")
        for item in r.json()["recent"]:
            assert "slug" in item
            assert "title" in item
            assert "category" in item
            assert "arxiv_id" in item
            assert "updated" in item


# ── /vault/article/{category}/{slug} ─────────────────────────────────────────

class TestVaultArticle:
    def test_happy_path(self, client):
        r = client.get("/vault/article/articles/paper-alpha")
        assert r.status_code == 200
        d = r.json()
        assert d["slug"] == "paper-alpha"
        assert d["title"] == "Paper Alpha"
        assert d["arxiv_id"] == "2101.00001"
        assert "body" in d
        assert "frontmatter" in d

    def test_proof_stats_included(self, client):
        r = client.get("/vault/article/articles/paper-alpha")
        d = r.json()
        stats = d["proof_stats"]
        assert stats["nodes"] == 3
        assert stats["suspicious"] == 1
        assert stats["gap"] == 1

    def test_missing_article_404(self, client):
        r = client.get("/vault/article/articles/nonexistent-slug")
        assert r.status_code == 404

    def test_bad_category_400(self, client):
        r = client.get("/vault/article/invalid-cat/some-slug")
        assert r.status_code == 400

    def test_bad_vault_400(self, client):
        r = client.get("/vault/article/articles/paper-alpha?vault_path=/nonexistent/xyz")
        assert r.status_code == 400

    def test_body_is_string(self, client):
        r = client.get("/vault/article/articles/paper-alpha")
        assert isinstance(r.json()["body"], str)

    def test_tags_parsed(self, client):
        r = client.get("/vault/article/articles/paper-alpha")
        tags = r.json()["tags"]
        assert isinstance(tags, list)
        assert "diffusion" in tags or any("diffusion" in str(t) for t in tags)


# ── /vault/articles ──────────────────────────────────────────────────────────

class TestVaultArticles:
    def test_happy_path(self, client):
        r = client.get("/vault/articles")
        assert r.status_code == 200
        d = r.json()
        assert d["total"] == 2  # articles only by default
        assert len(d["items"]) == 2

    def test_survey_category(self, client):
        r = client.get("/vault/articles?category=surveys")
        assert r.status_code == 200
        assert r.json()["total"] == 1

    def test_filter_by_q(self, client):
        r = client.get("/vault/articles?q=Alpha")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["slug"] == "paper-alpha"

    def test_filter_by_tag(self, client):
        r = client.get("/vault/articles?tag=attention")
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) == 1
        assert items[0]["slug"] == "paper-beta"

    def test_offset_and_limit(self, client):
        r = client.get("/vault/articles?limit=1&offset=1")
        assert r.status_code == 200
        d = r.json()
        assert d["total"] == 2
        assert len(d["items"]) == 1

    def test_bad_vault_400(self, client):
        r = client.get("/vault/articles?vault_path=/nonexistent/xyz")
        assert r.status_code == 400

    def test_bad_category_400(self, client):
        r = client.get("/vault/articles?category=not-a-cat")
        assert r.status_code == 400


# ── /vault/graph/{paper_arxiv_id} ────────────────────────────────────────────

class TestVaultGraph:
    def test_happy_path(self, client):
        r = client.get("/vault/graph/2101.00001")
        assert r.status_code == 200
        d = r.json()
        assert len(d["nodes"]) == 3
        assert len(d["edges"]) == 1
        assert "stats" in d

    def test_nodes_have_xy(self, client):
        """Nodes should have layout x/y coordinates assigned."""
        r = client.get("/vault/graph/2101.00001")
        for n in r.json()["nodes"]:
            assert "x" in n
            assert "y" in n

    def test_stats_by_kind(self, client):
        r = client.get("/vault/graph/2101.00001")
        stats = r.json()["stats"]
        assert "by_kind" in stats
        assert "theorem" in stats["by_kind"]

    def test_stats_by_status(self, client):
        r = client.get("/vault/graph/2101.00001")
        stats = r.json()["stats"]
        assert "by_status" in stats
        assert "ok" in stats["by_status"]

    def test_unknown_paper_returns_empty(self, client):
        r = client.get("/vault/graph/9999.99999")
        assert r.status_code == 200
        d = r.json()
        assert d["nodes"] == []
        assert d["edges"] == []

    def test_bad_vault_400(self, client):
        r = client.get("/vault/graph/2101.00001?vault_path=/nonexistent/xyz")
        assert r.status_code == 400
