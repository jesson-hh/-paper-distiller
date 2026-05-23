"""T1.1 — Scaffold tests.

Verify that create_app returns a FastAPI app and GET /config returns 200
with the expected keys.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from paper_distiller.web.server import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(str(tmp_path))
    return TestClient(app)


def test_create_app_returns_app(tmp_path):
    """create_app should return a FastAPI application object."""
    from fastapi import FastAPI

    app = create_app(str(tmp_path))
    assert isinstance(app, FastAPI)


def test_config_endpoint_returns_200(client):
    """GET /config should return HTTP 200."""
    r = client.get("/config")
    assert r.status_code == 200


def test_config_endpoint_has_expected_keys(client, tmp_path):
    """GET /config should return all required keys."""
    r = client.get(f"/config?vault_path={tmp_path}")
    data = r.json()
    required = {"model", "base_url", "permission_mode", "graph_depth", "plan_threshold_cny", "vault_path", "version"}
    assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"


def test_config_vault_path_echoed(client, tmp_path):
    """GET /config?vault_path=... should echo the vault_path back."""
    r = client.get(f"/config?vault_path={tmp_path}")
    data = r.json()
    assert str(tmp_path) in data["vault_path"]


def test_config_plan_threshold_is_float(client):
    """plan_threshold_cny should be a float."""
    r = client.get("/config")
    data = r.json()
    assert isinstance(data["plan_threshold_cny"], float)


def test_root_returns_200(client):
    """GET / should return 200 (even if static files not yet installed)."""
    r = client.get("/")
    assert r.status_code == 200
