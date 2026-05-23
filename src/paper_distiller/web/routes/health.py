"""GET /healthz — readiness check."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request, vault_path: str = ""):
    """Readiness check: vault exists, env set, proof_store reachable."""
    vp = vault_path or getattr(request.app.state, "vault_path", "")
    checks = []
    ok = True

    # 1. Vault directory
    vault_root = Path(vp) if vp else None
    vault_ok = bool(vault_root and vault_root.is_dir())
    checks.append({"name": "vault_exists", "ok": vault_ok, "detail": str(vp) if vp else "no vault_path"})
    if not vault_ok:
        ok = False

    # 2. Env vars
    api_key_ok = bool(os.getenv("PD_API_KEY"))
    base_url_ok = bool(os.getenv("PD_BASE_URL"))
    model_ok = bool(os.getenv("PD_MODEL"))
    env_ok = api_key_ok and base_url_ok and model_ok
    missing = [k for k, v in [("PD_API_KEY", api_key_ok), ("PD_BASE_URL", base_url_ok), ("PD_MODEL", model_ok)] if not v]
    checks.append({
        "name": "env_vars",
        "ok": env_ok,
        "detail": "all set" if env_ok else f"missing: {', '.join(missing)}",
    })
    if not env_ok:
        ok = False

    # 3. Proof store
    proof_store_ok = False
    proof_detail = "not found"
    if vault_ok:
        db_path = vault_root / ".proof_store" / "proofs.db"
        if db_path.exists():
            try:
                import sqlite3  # noqa: PLC0415
                conn = sqlite3.connect(str(db_path), check_same_thread=False)
                conn.execute("SELECT 1 FROM nodes LIMIT 1")
                conn.close()
                proof_store_ok = True
                proof_detail = "reachable"
            except Exception as e:
                proof_detail = str(e)
        else:
            proof_detail = "proof_store not yet created (distill first)"
            proof_store_ok = True  # not a fatal error — fresh vault is ok

    checks.append({"name": "proof_store", "ok": proof_store_ok, "detail": proof_detail})
    if not proof_store_ok:
        ok = False

    return {"ok": ok, "checks": checks}
