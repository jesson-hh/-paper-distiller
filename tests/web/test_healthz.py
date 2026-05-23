"""T1.6 — /healthz readiness check tests.

Happy path: vault exists, env set, proof_store reachable.
Sad paths: missing vault, missing env, corrupt db.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paper_distiller.web.server import create_app


@pytest.fixture
def bare_vault(tmp_path):
    """Vault directory with no proof store yet."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def vault_with_proof_store(bare_vault):
    """Vault with a minimal working proof_store."""
    db_dir = bare_vault / ".proof_store"
    db_dir.mkdir()
    conn = sqlite3.connect(str(db_dir / "proofs.db"))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_arxiv_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'extracted',
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()
    return bare_vault


@pytest.fixture
def client_with_vault(vault_with_proof_store):
    app = create_app(str(vault_with_proof_store))
    return TestClient(app)


@pytest.fixture
def client_bare(bare_vault):
    app = create_app(str(bare_vault))
    return TestClient(app)


# ── Happy path ────────────────────────────────────────────────────────────────

class TestHealthzHappy:
    def test_returns_200(self, client_with_vault, monkeypatch):
        monkeypatch.setenv("PD_API_KEY", "fake-key")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")
        r = client_with_vault.get("/healthz")
        assert r.status_code == 200

    def test_ok_true_when_all_checks_pass(self, client_with_vault, monkeypatch):
        monkeypatch.setenv("PD_API_KEY", "fake-key")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")
        r = client_with_vault.get("/healthz")
        data = r.json()
        assert data["ok"] is True

    def test_has_checks_list(self, client_with_vault, monkeypatch):
        monkeypatch.setenv("PD_API_KEY", "fake-key")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")
        r = client_with_vault.get("/healthz")
        data = r.json()
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) >= 3

    def test_each_check_has_name_ok_detail(self, client_with_vault, monkeypatch):
        monkeypatch.setenv("PD_API_KEY", "fake-key")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")
        r = client_with_vault.get("/healthz")
        for check in r.json()["checks"]:
            assert "name" in check
            assert "ok" in check
            assert "detail" in check

    def test_fresh_vault_without_proof_store_is_still_ok(self, client_bare, monkeypatch):
        """A fresh vault (no proof store yet) should not fail healthz."""
        monkeypatch.setenv("PD_API_KEY", "fake-key")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")
        r = client_bare.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        # proof_store check should still pass (no store = fresh vault, not fatal)
        proof_check = next(c for c in data["checks"] if c["name"] == "proof_store")
        assert proof_check["ok"] is True


# ── Sad paths ─────────────────────────────────────────────────────────────────

class TestHealthzSad:
    def test_missing_vault_makes_ok_false(self, monkeypatch):
        monkeypatch.setenv("PD_API_KEY", "fake-key")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")
        app = create_app("/nonexistent/vault/path")
        client = TestClient(app)
        r = client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is False
        vault_check = next(c for c in data["checks"] if c["name"] == "vault_exists")
        assert vault_check["ok"] is False

    def test_missing_api_key_makes_ok_false(self, client_with_vault, monkeypatch):
        monkeypatch.delenv("PD_API_KEY", raising=False)
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.setenv("PD_MODEL", "fake-model")
        r = client_with_vault.get("/healthz")
        data = r.json()
        assert data["ok"] is False
        env_check = next(c for c in data["checks"] if c["name"] == "env_vars")
        assert env_check["ok"] is False
        assert "PD_API_KEY" in env_check["detail"]

    def test_missing_base_url_makes_ok_false(self, client_with_vault, monkeypatch):
        monkeypatch.setenv("PD_API_KEY", "key")
        monkeypatch.delenv("PD_BASE_URL", raising=False)
        monkeypatch.setenv("PD_MODEL", "model")
        r = client_with_vault.get("/healthz")
        data = r.json()
        assert data["ok"] is False

    def test_missing_model_makes_ok_false(self, client_with_vault, monkeypatch):
        monkeypatch.setenv("PD_API_KEY", "key")
        monkeypatch.setenv("PD_BASE_URL", "http://fake.local")
        monkeypatch.delenv("PD_MODEL", raising=False)
        r = client_with_vault.get("/healthz")
        data = r.json()
        assert data["ok"] is False

    def test_vault_path_query_param_override(self, monkeypatch, tmp_path):
        """vault_path query param should override app.state.vault_path."""
        monkeypatch.setenv("PD_API_KEY", "k")
        monkeypatch.setenv("PD_BASE_URL", "http://x.local")
        monkeypatch.setenv("PD_MODEL", "m")
        # app.state points at a nonexistent vault
        app = create_app("/nonexistent/xyz")
        client = TestClient(app)
        # but query param points at tmp_path which exists
        r = client.get(f"/healthz?vault_path={tmp_path}")
        data = r.json()
        vault_check = next(c for c in data["checks"] if c["name"] == "vault_exists")
        assert vault_check["ok"] is True
