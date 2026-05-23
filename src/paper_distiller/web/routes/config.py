"""GET /config — read-only server configuration."""

from __future__ import annotations

import os

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/config")
async def get_config(request: Request, vault_path: str = ""):
    """Return read-only server configuration used by the frontend on mount."""
    vp = vault_path or getattr(request.app.state, "vault_path", "")

    # paper-distiller version
    try:
        from importlib.metadata import version as _v  # noqa: PLC0415
        pd_version = _v("paper-distiller")
    except Exception:
        pd_version = "unknown"

    return {
        "model": os.getenv("PD_MODEL", ""),
        "base_url": os.getenv("PD_BASE_URL", ""),
        "permission_mode": os.getenv("PD_PERMISSION_MODE", "default"),
        "graph_depth": os.getenv("PD_GRAPH_DEPTH", "off"),
        "plan_threshold_cny": float(os.getenv("PD_PLAN_THRESHOLD_CNY", "10.0")),
        "vault_path": vp,
        "version": pd_version,
    }
