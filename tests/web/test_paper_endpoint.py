"""T2.2 — /paper/{arxiv_id}.pdf endpoint tests.

All arXiv downloads are mocked via httpx.MockTransport injected through
monkeypatching get_or_download_pdf; no real network calls are made.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from paper_distiller.web.server import create_app

DUMMY_PDF = (
    b"%PDF-1.4\n1 0 obj\n<</Type /Catalog>>\nendobj\n"
    b"xref\n0 1\n0000000000 65535 f \n"
    b"trailer\n<</Size 1/Root 1 0 R>>\nstartxref\n9\n%%EOF"
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def vault(tmp_path) -> Path:
    """A bare vault directory (no .pdfs yet)."""
    v = tmp_path / "vault"
    v.mkdir()
    return v


@pytest.fixture
def client(vault) -> TestClient:
    app = create_app(str(vault))
    return TestClient(app)


@pytest.fixture
def seeded_vault(tmp_path) -> Path:
    """A vault that already has a PDF cached."""
    v = tmp_path / "vault"
    pdfs = v / ".pdfs"
    pdfs.mkdir(parents=True)
    (pdfs / "2110.05948.pdf").write_bytes(DUMMY_PDF)
    return v


@pytest.fixture
def client_seeded(seeded_vault) -> TestClient:
    app = create_app(str(seeded_vault))
    return TestClient(app)


# ── cache-hit (pre-seeded) ────────────────────────────────────────────────────

class TestCacheHit:
    def test_returns_200(self, client_seeded, seeded_vault):
        r = client_seeded.get(f"/paper/2110.05948.pdf?vault_path={seeded_vault}")
        assert r.status_code == 200

    def test_content_type_is_pdf(self, client_seeded, seeded_vault):
        r = client_seeded.get(f"/paper/2110.05948.pdf?vault_path={seeded_vault}")
        assert "application/pdf" in r.headers["content-type"]

    def test_cache_control_header(self, client_seeded, seeded_vault):
        r = client_seeded.get(f"/paper/2110.05948.pdf?vault_path={seeded_vault}")
        assert r.headers.get("cache-control") == "public, max-age=86400"

    def test_body_is_the_dummy_pdf(self, client_seeded, seeded_vault):
        r = client_seeded.get(f"/paper/2110.05948.pdf?vault_path={seeded_vault}")
        assert r.content == DUMMY_PDF


# ── cache-miss + successful download ─────────────────────────────────────────

class TestCacheMissDownload:
    def test_returns_200_after_download(self, vault, monkeypatch):
        """Monkeypatch get_or_download_pdf to pretend the file was downloaded."""
        pdfs_dir = vault / ".pdfs"
        pdfs_dir.mkdir()
        dest = pdfs_dir / "2110.05948.pdf"
        dest.write_bytes(DUMMY_PDF)  # pre-write so FileResponse finds it

        import paper_distiller.web.routes.paper as paper_mod

        def _fake_download(arxiv_id, vault_path, _client=None):
            return dest

        monkeypatch.setattr(paper_mod, "get_or_download_pdf", _fake_download)
        app = create_app(str(vault))
        client = TestClient(app)
        r = client.get(f"/paper/2110.05948.pdf?vault_path={vault}")
        assert r.status_code == 200
        assert r.content == DUMMY_PDF


# ── invalid arxiv_id → 400 ────────────────────────────────────────────────────

class TestInvalidId:
    @pytest.mark.parametrize("bad_id", [
        "1234",
        "abc.defgh",
        "2110.12",
    ])
    def test_bad_id_returns_400(self, client, vault, bad_id):
        """IDs that reach the handler but fail regex validation → 400."""
        r = client.get(f"/paper/{bad_id}.pdf?vault_path={vault}")
        assert r.status_code == 400

    @pytest.mark.parametrize("bad_id", [
        "../etc/passwd",
        "2110/05948",
        "2110%2F05948",
    ])
    def test_path_traversal_id_rejected_at_router_or_handler(self, client, vault, bad_id):
        """IDs with path-special chars (/,%) are intercepted by Starlette's router
        (404) before ever reaching our code — that is still a safe rejection."""
        r = client.get(f"/paper/{bad_id}.pdf?vault_path={vault}")
        assert r.status_code in (400, 404)

    def test_400_body_is_json(self, client, vault):
        r = client.get(f"/paper/bad.pdf?vault_path={vault}")
        assert r.status_code == 400
        data = r.json()
        assert "detail" in data


# ── download failure → 502 ────────────────────────────────────────────────────

class TestDownloadFailure:
    def test_404_from_arxiv_returns_502(self, vault, monkeypatch):
        import paper_distiller.web.routes.paper as paper_mod

        def _raise_runtime(arxiv_id, vault_path, _client=None):
            raise RuntimeError("arXiv returned HTTP 404")

        monkeypatch.setattr(paper_mod, "get_or_download_pdf", _raise_runtime)
        app = create_app(str(vault))
        client = TestClient(app)
        r = client.get(f"/paper/2110.05948.pdf?vault_path={vault}")
        assert r.status_code == 502

    def test_502_body_contains_detail(self, vault, monkeypatch):
        import paper_distiller.web.routes.paper as paper_mod

        def _raise_runtime(arxiv_id, vault_path, _client=None):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(paper_mod, "get_or_download_pdf", _raise_runtime)
        app = create_app(str(vault))
        client = TestClient(app)
        r = client.get(f"/paper/2110.05948.pdf?vault_path={vault}")
        assert r.status_code == 502
        data = r.json()
        assert data.get("detail") == "could not fetch PDF"


# ── vault_path from app.state (no query param) ───────────────────────────────

class TestVaultPathFromState:
    def test_uses_app_state_when_no_query_param(self, seeded_vault):
        """vault_path from app.state.vault_path should work without a query param."""
        app = create_app(str(seeded_vault))
        client = TestClient(app)
        r = client.get("/paper/2110.05948.pdf")
        assert r.status_code == 200
