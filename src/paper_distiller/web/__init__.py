"""paper_distiller.web — optional FastAPI web frontend for paper-distiller.

Install with: pip install "paper-distiller[web]"
Run with: paper-distiller-web --vault /path/to/vault
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(vault_path: str):  # noqa: ANN201  (import is conditional)
    """Create and return a configured FastAPI application.

    Import is lazy so that the core package remains importable without
    fastapi installed (only the [web] extra installs it).
    """
    from .server import create_app as _create  # noqa: PLC0415

    return _create(vault_path)
